#!/usr/bin/env python3
"""
eval_baselines.py — Cross-model evaluation of Basque language models.

Evaluates HuggingFace models (GPT-2 eus-euscrawl, Latxa-Qwen3.5-2B) and
optionally our Mamba-2 model on the SAME metrics, enabling fair cross-model
comparison.

═══════════════════════════════════════════════════════════════════════
  WHY BPC, NOT PPL
═══════════════════════════════════════════════════════════════════════

  PPL is tokenizer-dependent. Our model has a 4K Unigram vocab, GPT-2 has
  a 50K BPE vocab, and Qwen3.5 has ~150K tokens. A "token" means different
  things across these tokenizers — a 4K token covers ~3-4 characters, while
  a 150K token covers ~5-6 characters. Per-token PPL is NOT comparable.

  BPC (bits per character) is tokenizer-independent:
    BPC = total_negative_log_likelihood_in_bits / total_characters

  This lets us compare models fairly regardless of vocabulary size.

═══════════════════════════════════════════════════════════════════════
  METRICS
═══════════════════════════════════════════════════════════════════════

  1. BPC (bits per character) — PRIMARY cross-model metric
     Computed on eval/real_corpus/ raw text files.
     Each model tokenizes with its own tokenizer.
     BPC = sum(-log2(P(token_i))) / total_characters

  2. Next-Word CSR (simplified) — SECONDARY metric
     Tokenizer-agnostic keyboard simulation WITHOUT inference engineering.
     (No retokenization fallback, sticky merge, digit repair, etc.)
     This is a FAIR raw-model-ability comparison. Our model's inference
     engineering (§5.5) is shown separately as an improvement on top.

  3. Greedy completions — qualitative
     Greedy continuations on CSR test prompts for side-by-side inspection.

═══════════════════════════════════════════════════════════════════════
  USAGE
═══════════════════════════════════════════════════════════════════════

  # HuggingFace model (GPT-2, Latxa, etc.)
  python3 scripts/eval_baselines.py \\
      --hf-model "HiTZ/gpt2-eus-euscrawl" \\
      --corpus-dir eval/real_corpus \\
      --targets eval/targets.json \\
      --output-dir eval/baselines/gpt2-eus-euscrawl

  # Our Mamba model (for BPC on the same footing)
  python3 scripts/eval_baselines.py \\
      --morpheus-checkpoint checkpoints/best.pt \\
      --tokenizer tokenizer/basque_unigram_4000.model \\
      --corpus-dir eval/real_corpus \\
      --targets eval/targets.json \\
      --output-dir eval/baselines/morpheus-v2

  # Skip BPC (fast, CSR + completions only)
  python3 scripts/eval_baselines.py --hf-model "HiTZ/gpt2-eus-euscrawl" \\
      --targets eval/targets.json --skip-bpc

  # Skip CSR (BPC + completions only)
  python3 scripts/eval_baselines.py --hf-model "HiTZ/Latxa-Qwen3.5-2B" \\
      --corpus-dir eval/real_corpus --skip-csr
"""

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════
#  Model Wrapper: common interface for HF and Mamba models
# ═══════════════════════════════════════════════════════════════════════

class ModelWrapper:
    """Common interface for model evaluation.

    Subclasses must implement:
      - encode(text) -> list[int]
      - decode(ids) -> str
      - forward_logits(ids) -> torch.Tensor  (logits for LAST position)
      - generate_greedy(ids, max_tokens) -> (list[int], list[float])
      - vocab_size -> int
      - n_params -> int
      - model_name -> str
    """

    def encode(self, text):
        raise NotImplementedError

    def decode(self, ids):
        raise NotImplementedError

    @torch.no_grad()
    def forward_logits(self, ids):
        raise NotImplementedError

    @torch.no_grad()
    def generate_greedy(self, ids, max_tokens=10):
        """Greedy decode up to max_tokens. Returns (token_ids, logprobs)."""
        raise NotImplementedError

    @property
    def vocab_size(self):
        raise NotImplementedError

    @property
    def n_params(self):
        raise NotImplementedError

    @property
    def model_name(self):
        raise NotImplementedError


class HFModelWrapper(ModelWrapper):
    """Wrapper for HuggingFace models (GPT-2, Qwen, etc.)."""

    def __init__(self, hf_model_name, device="cuda", dtype="bf16"):
        from transformers import AutoTokenizer, AutoConfig

        self._model_name = hf_model_name
        self.device = device
        torch_dtype = torch.bfloat16 if dtype == "bf16" else torch.float32

        print(f"Loading tokenizer: {hf_model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"Loading model: {hf_model_name}...")
        config = AutoConfig.from_pretrained(hf_model_name)
        self._vocab_size = config.vocab_size

        # Try AutoModelForCausalLM first; fall back to multimodal for VLMs
        try:
            from transformers import AutoModelForCausalLM
            self.model = AutoModelForCausalLM.from_pretrained(
                hf_model_name,
                torch_dtype=torch_dtype,
                device_map=device if device == "cuda" else None,
                trust_remote_code=True,
            )
            self._is_multimodal = False
        except Exception as e:
            print(f"  AutoModelForCausalLM failed ({e}), trying AutoModelForMultimodalLM...")
            from transformers import AutoProcessor, AutoModelForMultimodalLM
            processor = AutoProcessor.from_pretrained(hf_model_name, trust_remote_code=True)
            self.tokenizer = processor.tokenizer
            self.model = AutoModelForMultimodalLM.from_pretrained(
                hf_model_name,
                torch_dtype=torch_dtype,
                device_map=device if device == "cuda" else None,
                trust_remote_code=True,
            )
            self._is_multimodal = True

        self.model.eval()
        self._n_params = sum(p.numel() for p in self.model.parameters())
        print(f"  Vocab size: {self._vocab_size}")
        print(f"  Parameters: {self._n_params / 1e6:.1f}M")
        print(f"  Multimodal: {self._is_multimodal}")

    def encode(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False)

    def decode(self, ids):
        if isinstance(ids, list):
            return self.tokenizer.decode(ids, skip_special_tokens=True)
        return self.tokenizer.decode([ids], skip_special_tokens=True)

    @torch.no_grad()
    def forward_logits(self, ids):
        """Return logits for the LAST token position."""
        import torch as T
        x = T.tensor([ids], device=self.device)
        with T.amp.autocast("cuda", dtype=torch.bfloat16):
            outputs = self.model(x)
        return outputs.logits[0, -1, :].float()

    @torch.no_grad()
    def generate_greedy(self, ids, max_tokens=10):
        """Greedy decode. Returns (token_ids, logprobs)."""
        generated = []
        logprobs = []
        current = list(ids)

        for _ in range(max_tokens):
            x = torch.tensor([current], device=self.device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = self.model(x)
            logits = outputs.logits[0, -1, :].float()
            log_probs = torch.log_softmax(logits, dim=-1)
            next_id = int(torch.argmax(logits).item())
            lp = float(log_probs[next_id].item())

            # Stop at EOS
            if next_id == self.tokenizer.eos_token_id:
                break

            generated.append(next_id)
            logprobs.append(lp)
            current.append(next_id)

        return generated, logprobs

    @property
    def vocab_size(self):
        return self._vocab_size

    @property
    def n_params(self):
        return self._n_params

    @property
    def model_name(self):
        return self._model_name


class MambaModelWrapper(ModelWrapper):
    """Wrapper for our Mamba-2 checkpoint."""

    def __init__(self, checkpoint_path, tokenizer_path, device="cuda"):
        import sentencepiece as spm
        from mamba_ssm.models.config_mamba import MambaConfig
        from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

        self._model_name = f"Morpheus v2 ({Path(checkpoint_path).name})"
        self.device = device

        # Load tokenizer
        self.sp = spm.SentencePieceProcessor(model_file=tokenizer_path)
        self._vocab_size = self.sp.get_piece_size()
        self._eos_id = 2  # </s> = id=2

        # Load checkpoint
        print(f"Loading checkpoint: {checkpoint_path}...")
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        raw_cfg = ckpt["config"]

        pad_multiple = raw_cfg.get("pad_vocab_size_multiple", 16)
        vocab_size = raw_cfg.get("padded_vocab_size",
                                 ((raw_cfg["vocab_size"] + pad_multiple - 1) // pad_multiple) * pad_multiple)

        config = MambaConfig(
            d_model=raw_cfg["d_model"],
            n_layer=raw_cfg["n_layer"],
            vocab_size=vocab_size,
            ssm_cfg={
                "layer": raw_cfg.get("ssm_layer", "Mamba2"),
                "d_state": raw_cfg["d_state"],
                "d_conv": raw_cfg["d_conv"],
                "expand": raw_cfg["expand"],
                "headdim": raw_cfg["headdim"],
                "chunk_size": raw_cfg.get("chunk_size", 256),
            },
            residual_in_fp32=raw_cfg.get("residual_in_fp32", True),
            fused_add_norm=raw_cfg.get("fused_add_norm", True),
            rms_norm=True,
        )

        self.model = MambaLMHeadModel(config, device=device, dtype=torch.bfloat16)
        self.model.load_state_dict(ckpt["model"], strict=True)
        self.model.eval()
        self.step = ckpt.get("step", "?")
        self.valid_loss = ckpt.get("valid_loss", None)
        self._n_params = sum(p.numel() for p in self.model.parameters())
        del ckpt

        print(f"  Step: {self.step}")
        print(f"  Vocab size: {self._vocab_size} (padded: {vocab_size})")
        print(f"  Parameters: {self._n_params / 1e6:.1f}M")
        if self.valid_loss:
            print(f"  Valid loss: {self.valid_loss} (PPL: {math.exp(self.valid_loss):.2f})")

    def encode(self, text):
        return self.sp.encode(text, out_type=int)

    def decode(self, ids):
        if isinstance(ids, list):
            return self.sp.decode(ids)
        return self.sp.decode([ids])

    @torch.no_grad()
    def forward_logits(self, ids):
        """Return logits for the LAST token position."""
        x = torch.tensor([ids], device=self.device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = self.model(x)
        return out.logits[0, -1, :].float()

    @torch.no_grad()
    def generate_greedy(self, ids, max_tokens=10):
        """Greedy decode. Returns (token_ids, logprobs)."""
        generated = []
        logprobs = []
        current = list(ids)

        for _ in range(max_tokens):
            x = torch.tensor([current], device=self.device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = self.model(x)
            logits = out.logits[0, -1, :].float()
            log_probs = torch.log_softmax(logits, dim=-1)
            next_id = int(torch.argmax(logits).item())
            lp = float(log_probs[next_id].item())

            if next_id == 0 or next_id == self._eos_id:
                break

            generated.append(next_id)
            logprobs.append(lp)
            current.append(next_id)

        return generated, logprobs

    @property
    def vocab_size(self):
        return self._vocab_size

    @property
    def n_params(self):
        return self._n_params

    @property
    def model_name(self):
        return self._model_name


# ═══════════════════════════════════════════════════════════════════════
#  Metric 1: BPC (Bits Per Character)
# ═══════════════════════════════════════════════════════════════════════

SEQ_LEN = 1024  # same as training


def count_chars(corpus_dir):
    """Count total characters in all .txt files in corpus_dir."""
    total = 0
    for txt_file in sorted(Path(corpus_dir).glob("*.txt")):
        with open(txt_file, encoding="utf-8") as f:
            total += len(f.read())
    return total


@torch.no_grad()
def compute_bpc_hf(wrapper, corpus_dir, device="cuda", seq_len=SEQ_LEN):
    """Compute BPC for a HuggingFace model on raw text corpus.

    Tokenizes each file with the model's own tokenizer, runs forward passes
    in seq_len windows, computes total NLL in bits, divides by character count.
    """
    from transformers import AutoTokenizer  # already loaded in wrapper

    total_nll_bits = 0.0
    total_tokens = 0
    total_chars = 0
    file_results = []

    for txt_file in sorted(Path(corpus_dir).glob("*.txt")):
        with open(txt_file, encoding="utf-8") as f:
            text = f.read()

        chars = len(text)
        total_chars += chars

        # Tokenize: raw text, no special tokens (base LM evaluation)
        token_ids = wrapper.encode(text)
        n_tokens = len(token_ids)
        total_tokens += n_tokens

        # Process in seq_len windows (same as training)
        file_nll = 0.0
        file_tok = 0
        n_windows = max(1, (n_tokens - 1) // seq_len)

        for i in range(n_windows + 1):
            start = i * seq_len
            end = min(start + seq_len + 1, n_tokens)
            if end - start < 2:
                break

            chunk = token_ids[start:end]
            x = torch.tensor([chunk[:-1]], device=device)
            y = torch.tensor([chunk[1:]], device=device)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = wrapper.model(x)
                logits = outputs.logits[0]  # [seq_len, vocab]

            # Compute per-token NLL in bits
            log_probs = torch.log_softmax(logits.float(), dim=-1)
            # Gather log probs for target tokens
            target_log_probs = log_probs.gather(1, y.unsqueeze(1)).squeeze(1)
            nll_bits = -target_log_probs.sum().item() / math.log(2)  # nats → bits

            file_nll += nll_bits
            file_tok += len(chunk) - 1

        file_bpc = file_nll / chars if chars > 0 else float("inf")
        file_results.append({
            "file": txt_file.name,
            "chars": chars,
            "tokens": n_tokens,
            "nll_bits": file_nll,
            "bpc": round(file_bpc, 6),
            "tokens_per_char": round(n_tokens / chars, 4) if chars > 0 else 0,
        })
        total_nll_bits += file_nll

        print(f"  {txt_file.name:<32s}  chars={chars:>7,}  tokens={n_tokens:>7,}  "
              f"tok/char={n_tokens/chars:.3f}  BPC={file_bpc:.4f}")

    overall_bpc = total_nll_bits / total_chars if total_chars > 0 else float("inf")
    overall_ppl = math.exp(total_nll_bits * math.log(2) / total_tokens) if total_tokens > 0 else float("inf")

    return {
        "bpc": round(overall_bpc, 6),
        "ppl_per_token": round(overall_ppl, 4),
        "total_nll_bits": total_nll_bits,
        "total_tokens": total_tokens,
        "total_chars": total_chars,
        "tokens_per_char": round(total_tokens / total_chars, 4) if total_chars > 0 else 0,
        "files": file_results,
    }


@torch.no_grad()
def compute_bpc_mamba(wrapper, corpus_dir, device="cuda", seq_len=SEQ_LEN):
    """Compute BPC for our Mamba-2 model on raw text corpus.

    Same logic as compute_bpc_hf but uses SentencePiece tokenizer with
    </s> (id=2) line separators, matching training semantics exactly.
    """
    import sentencepiece as spm

    total_nll_bits = 0.0
    total_tokens = 0
    total_chars = 0
    file_results = []

    eos_id = 2  # </s>

    for txt_file in sorted(Path(corpus_dir).glob("*.txt")):
        with open(txt_file, encoding="utf-8") as f:
            text = f.read()

        chars = len(text)
        total_chars += chars

        # Tokenize line-by-line with </s> separators (same as pretokenize.py)
        token_ids = []
        for line in text.split("\n"):
            line = line.strip()
            if not line:
                continue
            ids = wrapper.sp.encode(line, out_type=int)
            token_ids.extend(ids)
            token_ids.append(eos_id)

        n_tokens = len(token_ids)
        total_tokens += n_tokens

        # Process in seq_len windows
        file_nll = 0.0
        file_tok = 0
        n_windows = max(1, (n_tokens - 1) // seq_len)

        for i in range(n_windows + 1):
            start = i * seq_len
            end = min(start + seq_len + 1, n_tokens)
            if end - start < 2:
                break

            chunk = token_ids[start:end]
            x = torch.tensor([chunk[:-1]], device=device)
            y = torch.tensor([chunk[1:]], device=device)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                output = wrapper.model(x)
                logits = output.logits[0]

            # Sum CE (reduction=sum for token-weighted aggregation)
            ce_sum = F.cross_entropy(
                logits.float(),
                y,
                ignore_index=0,  # <unk>
                reduction="sum",
            )
            nll_bits = ce_sum.item() / math.log(2)  # nats → bits

            mask = y != 0
            count = mask.sum().item()
            file_nll += nll_bits
            file_tok += count

        file_bpc = file_nll / chars if chars > 0 else float("inf")
        file_results.append({
            "file": txt_file.name,
            "chars": chars,
            "tokens": n_tokens,
            "nll_bits": file_nll,
            "bpc": round(file_bpc, 6),
            "tokens_per_char": round(n_tokens / chars, 4) if chars > 0 else 0,
        })
        total_nll_bits += file_nll

        print(f"  {txt_file.name:<32s}  chars={chars:>7,}  tokens={n_tokens:>7,}  "
              f"tok/char={n_tokens/chars:.3f}  BPC={file_bpc:.4f}")

    overall_bpc = total_nll_bits / total_chars if total_chars > 0 else float("inf")
    overall_ppl = math.exp(total_nll_bits * math.log(2) / total_tokens) if total_tokens > 0 else float("inf")

    return {
        "bpc": round(overall_bpc, 6),
        "ppl_per_token": round(overall_ppl, 4),
        "total_nll_bits": total_nll_bits,
        "total_tokens": total_tokens,
        "total_chars": total_chars,
        "tokens_per_char": round(total_tokens / total_chars, 4) if total_chars > 0 else 0,
        "files": file_results,
    }


def compute_bpc(wrapper, corpus_dir, device="cuda"):
    """Dispatch to the correct BPC computation based on model type."""
    if isinstance(wrapper, MambaModelWrapper):
        return compute_bpc_mamba(wrapper, corpus_dir, device)
    else:
        return compute_bpc_hf(wrapper, corpus_dir, device)


# ═══════════════════════════════════════════════════════════════════════
#  Metric 2: Next-Word CSR (Simplified, Tokenizer-Agnostic)
# ═══════════════════════════════════════════════════════════════════════
#
#  Simulates word-by-word typing with greedy next-word prediction.
#  NO inference engineering (no retokenization fallback, no sticky merge,
#  no digit repair, no byte-fallback detection). This is a FAIR raw-model
#  comparison — all models are evaluated on the same simple algorithm.
#
#  Algorithm per test (prompt, target):
#    For each word in target:
#      1. Add space before word (1 keystroke)
#      2. At prefix_len=0 (no chars typed): generate greedy, extract first
#         word. If it matches target word → accept (1 Tab keystroke), done.
#      3. If not: type 1 character (1 keystroke), try again at prefix_len=1.
#      4. Continue until word is predicted or fully typed manually.
#
#  CSR = 1 - (total_keystrokes / total_chars)

_PUNCT = '.!,?;:()[]{}"\''


def _clean_word(w):
    """Lowercase and strip punctuation for word comparison."""
    return w.lower().strip(_PUNCT)


def _extract_first_word(text):
    """Extract the first word from generated text. Handles leading whitespace."""
    if not text:
        return ""
    text = text.lstrip()
    if not text:
        return ""
    word = ""
    for c in text:
        if c.isspace():
            break
        word += c
    return word.rstrip(_PUNCT)


def _has_garbage(text):
    """Check for byte-fallback garbage (chars above U+00FF)."""
    return any(ord(c) > 0xFF for c in text)


def simplified_nw_csr(wrapper, tests, device="cuda", max_gen_tokens=10):
    """Tokenizer-agnostic next-word CSR without inference engineering.

    Returns list of per-test result dicts.
    """
    wrapper.model.eval()
    results = []

    for ti, t in enumerate(tests):
        prompt = t["input"]
        target = t["target_completion"]
        words = target.split()

        text = prompt
        keystrokes = 0
        word_results = []

        for wi, target_word in enumerate(words):
            target_c = _clean_word(target_word)

            # Add space before word (unless first word and prompt doesn't end with space)
            if wi > 0 or (text and not text[-1].isspace()):
                text += " "
                keystrokes += 1

            accepted = False
            best_prediction = ""

            for prefix_len in range(len(target_word) + 1):
                prefix = target_word[:prefix_len]
                current = text + prefix

                # Encode and generate
                ids = wrapper.encode(current)
                if len(ids) > 1024:
                    ids = ids[-1024:]

                gen_ids, gen_probs = wrapper.generate_greedy(ids, max_tokens=max_gen_tokens)
                gen_text = wrapper.decode(gen_ids) if gen_ids else ""

                # Word completion path: prefix + generated text
                if prefix_len > 0:
                    full_word_candidate = prefix + gen_text.lstrip()
                    candidate_word = _extract_first_word(full_word_candidate)
                else:
                    # Next-word prediction path
                    candidate_word = _extract_first_word(gen_text)

                if candidate_word and not _has_garbage(candidate_word):
                    best_prediction = candidate_word

                candidate_c = _clean_word(candidate_word)

                if candidate_c and candidate_c == target_c:
                    # Accept! 1 keystroke (Tab)
                    keystrokes += 1
                    text = current  # word is "typed"
                    word_results.append({
                        "target": target_word,
                        "typed_prefix": prefix,
                        "predicted": candidate_word,
                        "accepted": True,
                        "prefix_len": prefix_len,
                        "confidence": round(math.exp(gen_probs[0]), 4) if gen_probs else 0.0,
                    })
                    accepted = True
                    break

                # Type next character if not at end of word
                if prefix_len < len(target_word):
                    keystrokes += 1

            if not accepted:
                # Fully typed manually
                text = text + target_word
                word_results.append({
                    "target": target_word,
                    "typed_prefix": target_word,
                    "predicted": best_prediction if best_prediction else "(none)",
                    "accepted": False,
                    "prefix_len": len(target_word),
                    "confidence": 0.0,
                })

        total_chars = len(target)
        csr = 1.0 - (keystrokes / total_chars) if total_chars > 0 else 0.0
        n_accepted = sum(1 for w in word_results if w["accepted"])
        prefix_lens = [w["prefix_len"] for w in word_results if w["accepted"]]

        results.append({
            "prompt": prompt,
            "target": target,
            "csr": round(csr, 4),
            "keystrokes": keystrokes,
            "total_chars": total_chars,
            "n_words": len(words),
            "n_accepted": n_accepted,
            "word_accuracy": round(n_accepted / len(words), 4) if words else 0.0,
            "avg_prefix_before_accept": round(sum(prefix_lens) / len(prefix_lens), 2) if prefix_lens else 0.0,
            "word_results": word_results,
        })

        if (ti + 1) % 10 == 0:
            print(f"  ... {ti + 1}/{len(tests)} tests done")

    return results


def bootstrap_mean_ci(values, n_bootstrap=1000, confidence=0.95, seed=42):
    """Bootstrap confidence interval for the mean."""
    arr = np.array(values, dtype=float)
    n = len(arr)
    if n == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.RandomState(seed)
    means = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sample = rng.choice(arr, size=n, replace=True)
        means[i] = sample.mean()
    alpha = (1 - confidence) / 2
    lower = float(np.percentile(means, alpha * 100))
    upper = float(np.percentile(means, (1 - alpha) * 100))
    point = float(arr.mean())
    return point, lower, upper


# ═══════════════════════════════════════════════════════════════════════
#  Metric 3: Greedy Completions (Qualitative)
# ═══════════════════════════════════════════════════════════════════════

def generate_completions(wrapper, tests, device="cuda", max_tokens=20):
    """Generate greedy completions for each test prompt."""
    wrapper.model.eval()
    results = []

    for t in tests:
        prompt = t["input"]
        target = t["target_completion"]

        ids = wrapper.encode(prompt)
        if len(ids) > 1024:
            ids = ids[-1024:]

        gen_ids, gen_probs = wrapper.generate_greedy(ids, max_tokens=max_tokens)
        gen_text = wrapper.decode(gen_ids) if gen_ids else ""

        results.append({
            "prompt": prompt,
            "target": target,
            "generated": gen_text,
            "generated_tokens": gen_ids,
            "logprobs": [round(lp, 4) for lp in gen_probs],
        })

    return results


# ═══════════════════════════════════════════════════════════════════════
#  Summary & Output
# ═══════════════════════════════════════════════════════════════════════

def print_summary(wrapper, bpc_results, csr_results, completion_results):
    """Print a formatted summary of all metrics."""
    print(f"\n{'=' * 80}")
    print(f"  EVALUATION SUMMARY: {wrapper.model_name}")
    print(f"{'=' * 80}")
    print(f"  Parameters:  {wrapper.n_params / 1e6:.1f}M")
    print(f"  Vocab size:  {wrapper.vocab_size}")

    if bpc_results:
        print(f"\n  ── BPC (Bits Per Character) ──")
        print(f"  BPC:              {bpc_results['bpc']:.4f}")
        print(f"  PPL (per-token):  {bpc_results['ppl_per_token']:.2f}")
        print(f"  Total chars:      {bpc_results['total_chars']:,}")
        print(f"  Total tokens:     {bpc_results['total_tokens']:,}")
        print(f"  Tokens/char:      {bpc_results['tokens_per_char']:.4f}")
        print(f"  ⚠ Corpus may be contaminated (Wikipedia/Berria in training data)")
        print(f"  ⚠ Absolute BPC is optimistic; relative comparison is valid")

    if csr_results:
        macro_csrs = [r["csr"] for r in csr_results]
        total_chars = sum(r["total_chars"] for r in csr_results)
        total_ks = sum(r["keystrokes"] for r in csr_results)
        total_words = sum(r["n_words"] for r in csr_results)
        total_accepted = sum(r["n_accepted"] for r in csr_results)
        point, lo, hi = bootstrap_mean_ci(macro_csrs)

        print(f"\n  ── Next-Word CSR (Simplified, no inference engineering) ──")
        print(f"  CSR (micro):      {1 - total_ks / total_chars:.4f}" if total_chars else "")
        print(f"  CSR (macro):      {point:.4f}  CI [{lo:.4f}, {hi:.4f}]")
        print(f"  Word accuracy:    {total_accepted / total_words:.4f}  ({total_accepted}/{total_words})")
        print(f"  N tests:          {len(csr_results)}")

    if completion_results:
        print(f"\n  ── Greedy Completions (first 5) ──")
        for c in completion_results[:5]:
            print(f"  Prompt:  {c['prompt'][:60]}")
            print(f"  Target:  {c['target'][:60]}")
            print(f"  Gen:     {c['generated'][:60]}")
            print()

    print(f"{'=' * 80}")


def main():
    parser = argparse.ArgumentParser(
        description="Cross-model evaluation of Basque LMs (BPC + CSR + completions)"
    )
    # Model selection (mutually exclusive)
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--hf-model", help="HuggingFace model name (e.g., HiTZ/gpt2-eus-euscrawl)")
    g.add_argument("--morpheus-checkpoint", help="Path to Morpheus .pt checkpoint")

    parser.add_argument("--tokenizer", default="tokenizer/basque_unigram_4000.model",
                        help="SentencePiece model (for --morpheus-checkpoint)")
    parser.add_argument("--corpus-dir", default="eval/real_corpus",
                        help="Directory with .txt files for BPC computation")
    parser.add_argument("--targets", default="eval/targets.json",
                        help="JSON file with CSR test targets")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory for JSON results")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-bpc", action="store_true", help="Skip BPC computation")
    parser.add_argument("--skip-csr", action="store_true", help="Skip CSR computation")
    parser.add_argument("--skip-completions", action="store_true", help="Skip greedy completions")
    parser.add_argument("--max-completions", type=int, default=30,
                        help="Max number of completions to generate")
    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"

    # Output directory
    if args.output_dir:
        out_dir = args.output_dir
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if args.hf_model:
            model_slug = args.hf_model.replace("/", "_")
        else:
            model_slug = "morpheus_" + Path(args.morpheus_checkpoint).stem
        out_dir = f"eval/baselines/{model_slug}/{timestamp}"
    os.makedirs(out_dir, exist_ok=True)

    # ── Load model ──
    print(f"\n{'=' * 80}")
    print(f"  Loading model...")
    print(f"{'=' * 80}")
    t0 = time.time()

    if args.hf_model:
        wrapper = HFModelWrapper(args.hf_model, device=device)
    else:
        wrapper = MambaModelWrapper(args.morpheus_checkpoint, args.tokenizer, device=device)

    load_time = time.time() - t0
    print(f"  Loaded in {load_time:.1f}s")

    all_results = {
        "model_name": wrapper.model_name,
        "n_params": wrapper.n_params,
        "vocab_size": wrapper.vocab_size,
        "load_time_s": round(load_time, 1),
        "timestamp": datetime.now().isoformat(),
    }

    # ── Metric 1: BPC ──
    if not args.skip_bpc:
        print(f"\n{'=' * 80}")
        print(f"  Computing BPC on {args.corpus_dir}/")
        print(f"  (Each model tokenizes with its own tokenizer)")
        print(f"{'=' * 80}")
        t0 = time.time()
        bpc_results = compute_bpc(wrapper, args.corpus_dir, device)
        eval_time = time.time() - t0

        print(f"\n  ┌─ BPC:          {bpc_results['bpc']:.4f}")
        print(f"  ├─ PPL (token):  {bpc_results['ppl_per_token']:.2f}")
        print(f"  ├─ Tokens/char:  {bpc_results['tokens_per_char']:.4f}")
        print(f"  └─ Eval time:    {eval_time:.1f}s")

        bpc_results["eval_time_s"] = round(eval_time, 1)
        all_results["bpc"] = bpc_results

        with open(os.path.join(out_dir, "bpc_results.json"), "w") as f:
            json.dump(bpc_results, f, indent=2, ensure_ascii=False)

    # ── Metric 2: Next-Word CSR ──
    if not args.skip_csr:
        # Load targets
        with open(args.targets) as f:
            targets_data = json.load(f)
        csr_tests = None
        for s in targets_data.get("strategies", []):
            if s["name"] == "csr":
                csr_tests = s["tests"]
                break

        if csr_tests:
            print(f"\n{'=' * 80}")
            print(f"  Simplified Next-Word CSR ({len(csr_tests)} tests)")
            print(f"  (No inference engineering — fair raw model comparison)")
            print(f"{'=' * 80}")
            t0 = time.time()
            csr_results = simplified_nw_csr(wrapper, csr_tests, device)
            eval_time = time.time() - t0

            # Aggregate
            macro_csrs = [r["csr"] for r in csr_results]
            total_chars = sum(r["total_chars"] for r in csr_results)
            total_ks = sum(r["keystrokes"] for r in csr_results)
            total_words = sum(r["n_words"] for r in csr_results)
            total_accepted = sum(r["n_accepted"] for r in csr_results)
            point, lo, hi = bootstrap_mean_ci(macro_csrs)

            csr_summary = {
                "csr_micro": round(1 - total_ks / total_chars, 4) if total_chars else 0.0,
                "csr_macro": round(point, 4),
                "csr_ci_lower": round(lo, 4),
                "csr_ci_upper": round(hi, 4),
                "word_accuracy": round(total_accepted / total_words, 4) if total_words else 0.0,
                "n_tests": len(csr_results),
                "n_words": total_words,
                "n_accepted": total_accepted,
                "eval_time_s": round(eval_time, 1),
            }

            print(f"\n  ┌─ CSR (micro):  {csr_summary['csr_micro']:.4f}")
            print(f"  ├─ CSR (macro):  {csr_summary['csr_macro']:.4f}  CI [{csr_summary['csr_ci_lower']:.4f}, {csr_summary['csr_ci_upper']:.4f}]")
            print(f"  ├─ Word acc:     {csr_summary['word_accuracy']:.4f}  ({total_accepted}/{total_words})")
            print(f"  └─ Eval time:    {eval_time:.1f}s")

            all_results["csr"] = csr_summary
            all_results["csr_details"] = csr_results

            with open(os.path.join(out_dir, "csr_results.json"), "w") as f:
                json.dump({"summary": csr_summary, "details": csr_results}, f,
                          indent=2, ensure_ascii=False)

    # ── Metric 3: Greedy Completions ──
    if not args.skip_completions:
        with open(args.targets) as f:
            targets_data = json.load(f)
        csr_tests = None
        for s in targets_data.get("strategies", []):
            if s["name"] == "csr":
                csr_tests = s["tests"]
                break

        if csr_tests:
            tests_for_completion = csr_tests[:args.max_completions]
            print(f"\n{'=' * 80}")
            print(f"  Greedy Completions ({len(tests_for_completion)} prompts)")
            print(f"{'=' * 80}")
            t0 = time.time()
            completion_results = generate_completions(wrapper, tests_for_completion, device)
            eval_time = time.time() - t0

            print(f"  Completed in {eval_time:.1f}s")
            for c in completion_results[:5]:
                print(f"\n  Prompt:  {c['prompt'][:70]}")
                print(f"  Target:  {c['target'][:70]}")
                print(f"  Gen:     {c['generated'][:70]}")

            all_results["completions"] = completion_results
            all_results["completions_eval_time_s"] = round(eval_time, 1)

            with open(os.path.join(out_dir, "completions.json"), "w") as f:
                json.dump(completion_results, f, indent=2, ensure_ascii=False)

    # ── Final summary ──
    print_summary(wrapper,
                  all_results.get("bpc"),
                  all_results.get("csr_details"),
                  all_results.get("completions"))

    # Save full summary
    # Remove csr_details from summary (it's in separate file)
    summary_out = {k: v for k, v in all_results.items() if k != "csr_details"}
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary_out, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n  Results saved to: {out_dir}/")


if __name__ == "__main__":
    main()
