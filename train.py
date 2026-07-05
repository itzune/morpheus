"""
Morpheus v2 Mamba-2 training script.

Trains a Mamba-2 language model on pre-tokenized Basque text for
predictive autocomplete. Supports Small (130M), Base (200M), and
Large (370M) configurations via YAML config files.

Usage:
    # Train with config file
    python train.py --config config/base.yaml

    # Override specific settings
    python train.py --config config/small.yaml --batch-size 64 --seq-len 256

    # Resume from checkpoint
    python train.py --config config/base.yaml --resume checkpoints/step_5000.pt

Full implementation: Morpheus_v2_Mamba.md §5
"""

import argparse
import math
import time
import yaml
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path

from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from mamba_ssm.models.config_mamba import MambaConfig

from src.dataset import MemmapTokenDataset


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    # Architecture
    "d_model": 960,
    "n_layer": 32,
    "d_state": 64,
    "d_conv": 4,
    "expand": 2,
    "headdim": 64,
    "chunk_size": 256,
    # Vocabulary
    "vocab_size": 32000,
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
    "compile": True,
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

    if cfg["compile"]:
        print("Compiling model with torch.compile...")
        model = torch.compile(model)

    return model


def _pad_vocab(vocab_size: int, multiple: int) -> int:
    """Pad vocab size to multiple for hardware alignment."""
    return ((vocab_size + multiple - 1) // multiple) * multiple


# ---------------------------------------------------------------------------
# LR Schedule
# ---------------------------------------------------------------------------

def get_lr_scheduler(optimizer, cfg, total_steps):
    """Create linear warmup + cosine decay LR schedule."""
    warmup_steps = cfg["warmup_tokens"] // (cfg["seq_len"] * cfg["batch_size"])
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
                ignore_index=0,
            )

        total_loss += loss.item()
        n += 1

    model.train()
    return total_loss / n


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(model, optimizer, step, cfg, path):
    """Save training checkpoint."""
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "config": cfg,
        },
        path,
    )


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------

def train(cfg: dict):
    """Run the full training loop."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Data
    print(f"Loading data from {cfg['train_data']}...")
    train_ds = MemmapTokenDataset(cfg["train_data"], seq_len=cfg["seq_len"])
    valid_ds = MemmapTokenDataset(cfg["valid_data"], seq_len=cfg["seq_len"])
    print(f"  Train: {len(train_ds):,} sequences")
    print(f"  Valid: {len(valid_ds):,} sequences")

    train_loader = DataLoader(
        train_ds, batch_size=cfg["batch_size"],
        shuffle=True, num_workers=4, pin_memory=True, drop_last=True,
    )
    valid_loader = DataLoader(
        valid_ds, batch_size=cfg["batch_size"],
        shuffle=False, num_workers=2, pin_memory=True,
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
    total_steps = cfg["total_tokens"] // (cfg["seq_len"] * cfg["batch_size"])
    scheduler = get_lr_scheduler(optimizer, cfg, total_steps)
    print(f"Total steps: {total_steps:,}")

    # Tracking
    dtype = torch.bfloat16 if cfg["dtype"] == "bfloat16" else torch.float16
    global_step = 0
    tokens_seen = 0
    best_valid_loss = float("inf")
    t0 = time.time()

    # Initialize W&B
    try:
        import wandb
        wandb.init(project="morpheus-v2-mamba", config=cfg)
        use_wandb = True
    except ImportError:
        print("wandb not installed — skipping experiment tracking")
        use_wandb = False

    print(f"\nStarting training... ({total_steps:,} steps, ~{cfg['total_tokens']/1e9:.0f}B tokens)\n")

    # Training loop
    for epoch in range(100):
        model.train()
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
                    ignore_index=0,
                )

            # Backward
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            tokens_seen += x.numel()
            global_step += 1

            # Logging
            if global_step % cfg["log_interval"] == 0:
                ppl = math.exp(min(loss.item(), 20))
                elapsed = time.time() - t0
                tps = tokens_seen / elapsed if elapsed > 0 else 0
                lr = optimizer.param_groups[0]["lr"]

                msg = (f"step={global_step:>6d}  loss={loss.item():.4f}  ppl={ppl:.1f}  "
                       f"lr={lr:.2e}  grad_norm={grad_norm:.2f}  tok/s={tps:.0f}")
                print(msg)

                if use_wandb:
                    wandb.log({
                        "train/loss": loss.item(),
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

                if use_wandb:
                    wandb.log({
                        "valid/loss": valid_loss,
                        "valid/ppl": valid_ppl,
                        "step": global_step,
                    })

                if valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss
                    save_checkpoint(model, optimizer, global_step, cfg,
                                    output_dir / "best.pt")
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
    parser.add_argument("--config", default="config/base.yaml", help="YAML config file")
    parser.add_argument("--batch-size", type=int, help="Override batch size")
    parser.add_argument("--seq-len", type=int, help="Override sequence length")
    parser.add_argument("--learning-rate", type=float, help="Override learning rate")
    parser.add_argument("--total-tokens", type=int, help="Override total training tokens")
    parser.add_argument("--train-data", help="Override training data path")
    parser.add_argument("--valid-data", help="Override validation data path")
    parser.add_argument("--output-dir", help="Override checkpoint output dir")
    parser.add_argument("--resume", help="Path to checkpoint to resume from")
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

    train(cfg)


if __name__ == "__main__":
    main()
