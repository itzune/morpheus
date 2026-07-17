"""
Morpheus v2 Mamba-2 training script.

Trains a Mamba-2 language model on pre-tokenized Basque text for
predictive autocomplete. Supports Small (91M), Base (200M), and
Large (370M) configurations via YAML config files.

Usage:
    # Train with config file
    python train.py --config config/small.yaml

    # Override specific settings
    python train.py --config config/small.yaml --batch-size 64 --seq-len 256

    # Resume from checkpoint
    python train.py --config config/small.yaml --resume checkpoints/step_5000.pt

Full implementation: Morpheus_v2_Mamba.md §5
"""

import argparse
import json
import math
import os
import random
import time
import warnings
import yaml
import numpy as np
import sentencepiece as spm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path

from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from mamba_ssm.models.config_mamba import MambaConfig

from src.dataset import MemmapTokenDataset
from src.eval_utils import compute_autocomplete_metrics


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    # Architecture
    "d_model": 960,
    "n_layer": 32,
    "ssm_layer": "Mamba2",
    "d_state": 64,
    "d_conv": 4,
    "expand": 2,
    "headdim": 64,
    "chunk_size": 256,
    # Vocabulary
    "vocab_size": 4000,
    "pad_vocab_size_multiple": 16,
    # Regularization
    "residual_in_fp32": True,
    "fused_add_norm": True,
    # Training
    "seq_len": 512,
    "batch_size": 128,
    "gradient_accumulation": 1,
    # Optimization
    "learning_rate": 2e-3,
    "min_lr": 1e-5,
    "weight_decay": 0.1,
    "beta1": 0.9,
    "beta2": 0.95,
    "grad_clip": 1.0,
    # Schedule
    "warmup_tokens": 50_000_000,
    "total_tokens": 10_000_000_000,
    # Precision
    "dtype": "bfloat16",  # "bfloat16" or "float16"
    "compile": False,
    # Reproducibility
    "seed": 42,
    # Logging
    "log_interval": 50,
    "eval_interval": 1000,
    "save_interval": 5000,
    # Paths
    "train_data": "data/train_tokens.npy",
    "valid_data": "data/valid_tokens.npy",
    "output_dir": "checkpoints",
}


def load_config(config_path: str, overrides: dict = None) -> dict:
    """Load YAML config and apply CLI overrides."""
    cfg = DEFAULT_CONFIG.copy()

    if config_path:
        with open(config_path) as f:
            yaml_cfg = yaml.safe_load(f)
        cfg.update(yaml_cfg)

    if overrides:
        cfg.update({k: v for k, v in overrides.items() if v is not None})

    return cfg


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_model(cfg: dict, device: torch.device) -> MambaLMHeadModel:
    """Build Mamba-2 model from config."""
    dtype = torch.bfloat16 if cfg["dtype"] == "bfloat16" else torch.float16

    config = MambaConfig(
        d_model=cfg["d_model"],
        n_layer=cfg["n_layer"],
        vocab_size=cfg.get("padded_vocab_size",
                          _pad_vocab(cfg["vocab_size"], cfg["pad_vocab_size_multiple"])),
        ssm_cfg={
            "layer": cfg["ssm_layer"],
            "d_state": cfg["d_state"],
            "d_conv": cfg["d_conv"],
            "expand": cfg["expand"],
            "headdim": cfg["headdim"],
            "chunk_size": cfg["chunk_size"],
        },
        rms_norm=True,
        residual_in_fp32=cfg["residual_in_fp32"],
        fused_add_norm=cfg["fused_add_norm"],
    )

    model = MambaLMHeadModel(config, device=device, dtype=dtype)

    if cfg.get("compile", False):
        try:
            print("Compiling model with torch.compile...")
            model = torch.compile(model, mode="reduce-overhead")
        except Exception as e:
            print(f"torch.compile failed: {e}. Continuing without compile.")

    return model


def _pad_vocab(vocab_size: int, multiple: int) -> int:
    """Pad vocab size to multiple for hardware alignment."""
    return ((vocab_size + multiple - 1) // multiple) * multiple


# ---------------------------------------------------------------------------
# LR Schedule
# ---------------------------------------------------------------------------

def get_lr_scheduler(optimizer, cfg, total_steps):
    """Create linear warmup + cosine decay LR schedule."""
    accum = cfg.get("gradient_accumulation", 1)
    effective_batch = cfg["seq_len"] * cfg["batch_size"] * accum
    warmup_steps = cfg["warmup_tokens"] // effective_batch
    warmup_steps = min(warmup_steps, total_steps // 10)  # Cap warmup at 10%

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        progress = min(progress, 1.0)
        return (cfg["min_lr"] + 0.5 * (cfg["learning_rate"] - cfg["min_lr"]) *
                (1 + math.cos(math.pi * progress))) / cfg["learning_rate"]

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, loader, device, max_batches=100):
    """Run validation and return average loss."""
    model.eval()
    total_loss = 0.0
    n = 0

    dtype = torch.bfloat16
    if hasattr(model, 'config') and hasattr(model.config, 'residual_in_fp32'):
        dtype = torch.bfloat16 if model.config.residual_in_fp32 else torch.float16

    for i, (x, y) in enumerate(loader):
        if i >= max_batches:
            break
        x, y = x.to(device), y.to(device)

        with torch.amp.autocast("cuda", dtype=dtype):
            output = model(x)
            loss = F.cross_entropy(
                output.logits.view(-1, output.logits.size(-1)),
                y.view(-1),
                # ignore_index=0 (<unk>): safety net for any stray unknown chars.
                # Separators are </s> (id=2), NOT id=0, so they ARE included in loss.
                ignore_index=0,
            )

        total_loss += loss.item()
        n += 1

    model.train()
    return total_loss / n



# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(model, optimizer, step, cfg, path, valid_loss=None):
    """Save training checkpoint (atomic: write to .tmp, then os.replace).

    os.replace is atomic on POSIX, so the checkpoint file only appears at
    its final path once fully written. This prevents corruption if a stop
    monitor (or crash) interrupts the write.
    """
    tmp_path = str(path) + ".tmp"
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "config": cfg,
            "valid_loss": valid_loss,
        },
        tmp_path,
    )
    os.replace(tmp_path, path)


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------

def train(cfg: dict, checkpoint_path: str = None, pretrained_path: str = None):
    """Run the full training loop.
    
    Args:
        cfg: Training configuration dictionary.
        checkpoint_path: If provided, full resume (model + optimizer + step).
        pretrained_path: If provided, load model weights only (fresh optimizer,
            step=0). Used for continued pretraining (e.g. Phase 6 FIM).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Seed for reproducibility
    seed = cfg.get("seed", 42)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    print(f"Seed: {seed}")

    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Data
    print(f"Loading data from {cfg['train_data']}...")
    train_ds = MemmapTokenDataset(cfg["train_data"], seq_len=cfg["seq_len"])
    valid_ds = MemmapTokenDataset(cfg["valid_data"], seq_len=cfg["seq_len"])
    print(f"  Train: {len(train_ds):,} sequences")
    print(f"  Valid: {len(valid_ds):,} sequences")

    # num_workers=2: server has 30 GB RAM; each worker is a full Python process (~3-4 GB).
    # 4 workers + main process caused OOM. 2 workers + main fits in 30 GB.
    train_loader = DataLoader(
        train_ds, batch_size=cfg["batch_size"],
        shuffle=True, num_workers=2, pin_memory=True, drop_last=True,
    )
    valid_loader = DataLoader(
        valid_ds, batch_size=cfg["batch_size"],
        shuffle=False, num_workers=1, pin_memory=True,
    )

    # Model
    model = build_model(cfg, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} parameters (~{n_params / 1e6:.0f}M)")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["learning_rate"],
        betas=(cfg["beta1"], cfg["beta2"]),
        weight_decay=cfg["weight_decay"],
    )

    # LR Schedule
    accum = cfg.get("gradient_accumulation", 1)
    effective_batch = cfg["seq_len"] * cfg["batch_size"] * accum
    total_steps = cfg["total_tokens"] // effective_batch
    scheduler = get_lr_scheduler(optimizer, cfg, total_steps)
    print(f"Total steps: {total_steps:,} (accum={accum}, effective_batch={effective_batch})")

    # Resume from checkpoint if provided
    dtype = torch.bfloat16 if cfg["dtype"] == "bfloat16" else torch.float16
    global_step = 0
    tokens_seen = 0
    best_valid_loss = float("inf")

    if checkpoint_path:
        print(f"Resuming from checkpoint: {checkpoint_path}")
        # Load on CPU to avoid doubling GPU VRAM
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        global_step = ckpt["step"]
        del ckpt  # Free CPU memory
        # Advance scheduler to match global_step
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for _ in range(global_step):
                scheduler.step()
        print(f"  Resumed at step {global_step:,}, LR: {optimizer.param_groups[0]['lr']:.2e}")
        tokens_seen = global_step * effective_batch
        # Restore best_valid_loss from best.pt if it has one
        best_path = output_dir / "best.pt"
        if best_path.exists():
            best_ckpt = torch.load(best_path, map_location='cpu', weights_only=False)
            best_valid_loss = best_ckpt.get("valid_loss", float("inf"))
            print(f"  Found best.pt (step {best_ckpt.get('step', '?')}), best_valid_loss={best_valid_loss:.4f}")
    elif pretrained_path:
        print(f"Loading pretrained model weights: {pretrained_path}")
        ckpt = torch.load(pretrained_path, map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt["model"])
        src_step = ckpt.get("step", "?")
        del ckpt  # Free CPU memory
        print(f"  Loaded model from step {src_step}")
        print(f"  Fresh optimizer, starting from step 0 (continued pretraining)")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.2e}")

    tokens_at_start = tokens_seen
    t0 = time.time()

    # Load autocomplete eval targets for CSR/MorphAcc logging
    csr_tests = []
    morphacc_tests = []
    targets_path = Path(cfg.get("eval_targets", "eval/targets.json"))
    if not targets_path.exists():
        raise FileNotFoundError(f"Eval targets file not found: {targets_path}")
    with open(targets_path) as f:
        targets = json.load(f)
    # targets.json nests tests under strategies
    for strategy in targets.get("strategies", []):
        if strategy.get("name") == "csr":
            csr_tests.extend(strategy.get("tests", []))
        elif strategy.get("name") == "morphacc":
            morphacc_tests.extend(strategy.get("tests", []))
    print(f"Loaded eval targets: {len(csr_tests)} CSR, {len(morphacc_tests)} MorphAcc")

    # Load SentencePiece model for tokenization during eval
    sp_model_path = cfg.get("sp_model", "tokenizer/basque_unigram_4000.model")
    sp = spm.SentencePieceProcessor()
    sp.Load(sp_model_path)
    print(f"Loaded tokenizer: {sp_model_path}")

    # Initialize W&B
    try:
        import wandb
        wandb_run_id = os.environ.get("WANDB_RUN_ID")
        wandb_resume = os.environ.get("WANDB_RESUME", "allow")
        if wandb_run_id:
            print(f"Resuming W&B run: {wandb_run_id}")
        wandb.init(
            project="morpheus-v2-mamba",
            config=cfg,
            id=wandb_run_id or None,
            resume=wandb_resume if wandb_run_id else None,
        )
        use_wandb = True
    except ImportError:
        print("wandb not installed — skipping experiment tracking")
        use_wandb = False

    if checkpoint_path:
        print(f"\nResuming training from step {global_step:,}/{total_steps:,} "
              f"(~{cfg['total_tokens']/1e9:.0f}B tokens)\n")
    elif pretrained_path:
        print(f"\nContinued pretraining from pretrained weights "
              f"({total_steps:,} steps, ~{cfg['total_tokens']/1e9:.1f}B tokens)\n")
    else:
        print(f"\nStarting training... ({total_steps:,} steps, ~{cfg['total_tokens']/1e9:.0f}B tokens)\n")

    # Training loop
    for epoch in range(100):
        model.train()
        micro_step = 0

        for x, y in train_loader:
            if tokens_seen >= cfg["total_tokens"]:
                break

            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            # Forward
            with torch.amp.autocast("cuda", dtype=dtype):
                output = model(x)
                logits = output.logits
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    y.view(-1),
                    # ignore_index=0 (<unk>): safety net for stray unknown chars.
                    # Separators are </s> (id=2), NOT id=0, so they ARE in loss.
                    ignore_index=0,
                ) / accum  # Scale loss for gradient accumulation

            # Backward
            loss.backward()
            micro_step += 1

            # Only step optimizer after accumulation steps
            if micro_step % accum == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                tokens_seen += x.numel() * accum
                global_step += 1

                # Logging
                if global_step % cfg["log_interval"] == 0:
                    ppl = math.exp(min(loss.item() * accum, 20))
                    elapsed = time.time() - t0
                    tps = (tokens_seen - tokens_at_start) / elapsed if elapsed > 0 else 0
                    lr = optimizer.param_groups[0]["lr"]

                    msg = (f"step={global_step:>6d}  loss={loss.item() * accum:.4f}  ppl={ppl:.1f}  "
                           f"lr={lr:.2e}  grad_norm={grad_norm:.2f}  tok/s={tps:.0f}")
                    print(msg)

                    if use_wandb:
                        wandb.log({
                            "train/loss": loss.item() * accum,
                            "train/ppl": ppl,
                            "train/lr": lr,
                            "train/grad_norm": grad_norm,
                            "train/tokens_per_sec": tps,
                            "train/tokens_seen": tokens_seen,
                            "step": global_step,
                        })

                # Validation
                if global_step % cfg["eval_interval"] == 0:
                    valid_loss = evaluate(model, valid_loader, device)
                    valid_ppl = math.exp(min(valid_loss, 20))
                    print(f"  [VALID] step={global_step}  loss={valid_loss:.4f}  ppl={valid_ppl:.1f}")

                    wandb_metrics = {
                        "valid/loss": valid_loss,
                        "valid/ppl": valid_ppl,
                        "step": global_step,
                    }

                    # Add CSR and MorphAcc if tests are available
                    if csr_tests and sp is not None:
                        ac_metrics = compute_autocomplete_metrics(
                            model, sp, csr_tests, morphacc_tests, device
                        )
                        wandb_metrics.update(ac_metrics)
                        print(f"  [AUTOCOMPLETE] CSR={ac_metrics['valid/csr']:.3f}  "
                              f"MorphAcc={ac_metrics['valid/morphacc']:.3f}")
                        print(f"  [NEXT-WORD]    NW-CSR={ac_metrics['valid/nw_csr']:.3f}  "
                              f"Top1={ac_metrics['valid/nw_top1_accuracy']:.3f}  "
                              f"Top3={ac_metrics['valid/nw_top3_accuracy']:.3f}  "
                              f"Top5={ac_metrics['valid/nw_top5_accuracy']:.3f}  "
                              f"Accept={ac_metrics['valid/nw_acceptance_rate']:.3f}  "
                              f"AvgPrefix={ac_metrics['valid/nw_avg_prefix_before_accept']:.1f}  "
                              f"AvgConf={ac_metrics['valid/nw_avg_confidence']:.3f}")

                    if use_wandb:
                        wandb.log(wandb_metrics)

                    if valid_loss < best_valid_loss:
                        best_valid_loss = valid_loss
                        save_checkpoint(model, optimizer, global_step, cfg,
                                        output_dir / "best.pt", valid_loss=valid_loss)
                        print(f"  [BEST] New best valid loss: {valid_loss:.4f} (ppl: {valid_ppl:.1f})")

                # Checkpoint
                if global_step % cfg["save_interval"] == 0:
                    ckpt_path = output_dir / f"step_{global_step:07d}.pt"
                    save_checkpoint(model, optimizer, global_step, cfg, ckpt_path)
                    print(f"  Saved: {ckpt_path}")

        if tokens_seen >= cfg["total_tokens"]:
            break

    elapsed = time.time() - t0
    print(f"\nTraining complete: {elapsed/3600:.1f} hours, {tokens_seen:,} tokens")
    print(f"Best valid loss: {best_valid_loss:.4f} (ppl: {math.exp(min(best_valid_loss, 20)):.1f})")

    if use_wandb:
        wandb.finish()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train Morpheus v2 Mamba-2")
    parser.add_argument("--config", default="config/small.yaml", help="YAML config file")
    parser.add_argument("--batch-size", type=int, help="Override batch size")
    parser.add_argument("--seq-len", type=int, help="Override sequence length")
    parser.add_argument("--learning-rate", type=float, help="Override learning rate")
    parser.add_argument("--total-tokens", type=int, help="Override total training tokens")
    parser.add_argument("--train-data", help="Override training data path")
    parser.add_argument("--valid-data", help="Override validation data path")
    parser.add_argument("--output-dir", help="Override checkpoint output dir")
    parser.add_argument("--resume", help="Path to checkpoint to resume from (full: model + optimizer + step)")
    parser.add_argument("--pretrained", help="Path to pretrained checkpoint (model weights only, fresh optimizer)")
    parser.add_argument("--no-compile", action="store_true", help="Disable torch.compile")
    args = parser.parse_args()

    # Load config
    overrides = {
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "learning_rate": args.learning_rate,
        "total_tokens": args.total_tokens,
        "train_data": args.train_data,
        "valid_data": args.valid_data,
        "output_dir": args.output_dir,
    }
    cfg = load_config(args.config, overrides)

    if args.no_compile:
        cfg["compile"] = False

    # Print effective config
    print("Configuration:")
    for k, v in sorted(cfg.items()):
        print(f"  {k}: {v}")
    print()

    # --pretrained can come from CLI or config (cfg["pretrained"])
    pretrained = args.pretrained or cfg.get("pretrained")
    train(cfg, checkpoint_path=args.resume, pretrained_path=pretrained)


if __name__ == "__main__":
    main()
