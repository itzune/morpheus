# DATA.md — Corpus Strategy and Data Pipeline for Morpheus v2

## 1. Overview

Morpheus v2 reuses the **entire data pipeline output from v1**. We do not recompute tokenizers, data splits, or preprocessing. This document describes what's reused, what's different, and how to prepare data for Mamba-2 training.

---

## 2. Reused Assets from v1

| Asset | v1 Path | v2 Path | Status |
|-------|---------|---------|--------|
| SentencePiece model | `tokenizer/basque_unigram_32k.model` | `tokenizer/` (symlink or copy) | ✅ Reused |
| Train split | `data/splits/train/` | `data/splits/train/` (symlink) | ✅ Reused |
| Valid split | `data/splits/valid/` | `data/splits/valid/` (symlink) | ✅ Reused |
| Test split | `data/splits/test/` | `data/splits/test/` (symlink) | ✅ Reused |
| Corpus strategy | `DATA.md` in v1 repo | Reference only | ✅ Referenced |
| MWE tokens | 1,000 injected entries | Reuse (included in 32K vocab) | ✅ Reused |

**Why not recompute?** The v1 tokenizer was trained on 5.45M domain-filtered documents and achieves fertility 1.71 (target ≤ 2.0). Retraining on the same data would produce an identical model. The 32K Unigram vocabulary is architecture-agnostic — it works for Mamba exactly as it works for LSTM.

---

## 3. What's Different from v1

### 3.1 No Morfessor, No Pre-segmentation

v1's ADR-013 already dropped Morfessor pre-segmentation — the Unigram tokenizer alone achieved fertility 1.705 without it. v2 continues this approach.

### 3.2 Pre-tokenization Format

v1 used a custom binary format (`uint16 .bin` files). v2 uses **memory-mapped numpy arrays** (`.npy`) for training. This is a different on-disk format but contains the same tokenized data.

```bash
# Convert v1 data to numpy format for Mamba training
python scripts/pretokenize.py \
    --sp-model tokenizer/basque_unigram_32k.model \
    --input-dir data/splits/train \
    --output data/train_tokens.npy

python scripts/pretokenize.py \
    --sp-model tokenizer/basque_unigram_32k.model \
    --input-dir data/splits/valid \
    --output data/valid_tokens.npy
```

### 3.3 Sequence Length

v1 trained at `seq_len=128`. v2 trains at **`seq_len=512`** — Mamba can handle longer contexts efficiently (O(n) inference, not O(n²) like Transformers), and longer context windows improve suffix agreement prediction for Basque's recursive morphology.

### 3.4 No BoW Context Encoding

v1 required a separate context encoder that pre-computed context vectors from prior sentences. Mamba processes the full token context natively through its recurrent state. The data pipeline is simpler: just tokenize, pack into sequences, and feed to the model.

---

## 4. Data Pipeline (v2-specific)

### 4.1 Pre-tokenization Script

```python
# scripts/pretokenize.py
"""Pre-tokenize corpus into memory-mapped numpy array for Mamba training."""
import numpy as np
import sentencepiece as spm
from pathlib import Path
from tqdm import tqdm
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sp-model", required=True)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    sp = spm.SentencePieceProcessor(model_file=args.sp_model)

    all_tokens = []
    input_dir = Path(args.input_dir)

    for txt_file in sorted(input_dir.glob("*.txt")):
        with open(txt_file) as f:
            for line in tqdm(f, desc=txt_file.name):
                line = line.strip()
                if not line:
                    continue
                tokens = sp.encode(line, out_type=int)
                all_tokens.extend(tokens)
                all_tokens.append(0)  # <eos> / document separator

    arr = np.array(all_tokens, dtype=np.uint16)
    np.save(args.output, arr)
    print(f"Saved {len(arr):,} tokens to {args.output}")
    print(f"Size: {arr.nbytes / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
```

### 4.2 Dataset Class

```python
# src/dataset.py
"""Memory-mapped dataset for Mamba causal LM training."""
import numpy as np
import torch
from torch.utils.data import Dataset


class MemmapTokenDataset(Dataset):
    """Loads pre-tokenized data from a numpy file via memory mapping."""

    def __init__(self, path: str, seq_len: int = 512):
        self.data = np.load(path, mmap_mode='r')
        self.seq_len = seq_len
        self.n_sequences = (len(self.data) - 1) // seq_len

    def __len__(self):
        return self.n_sequences

    def __getitem__(self, idx):
        start = idx * self.seq_len
        end = start + self.seq_len + 1  # +1 for target shift

        chunk = self.data[start:end].astype(np.int64)
        x = torch.from_numpy(chunk[:-1])   # input:  tokens[0..seq_len-1]
        y = torch.from_numpy(chunk[1:])    # target: tokens[1..seq_len]
        return x, y
```

### 4.3 Corpus Size Estimate

Based on v1's corpus (5.45M documents, ~1.17B words):

| Metric | Estimate |
|--------|----------|
| Total word tokens | ~1.17B |
| Total subword tokens (fertility ~1.7) | ~2.0B |
| Sequences (seq_len=512) | ~3.9M |
| Epochs to 10B tokens (target) | ~5 |
| Train .npy size (uint16) | ~4 GB |

---

## 5. Corpus Reference

For the full dataset registry (EusCrawl, Latxa, BERnaT BSM, EuskañolDS) and preprocessing details, see the v1 repo's [`DATA.md`](../morpheus/DATA.md). All decisions around domain filtering (ADR-011), deduplication (ADR-012), and code-switching inclusion (ADR-009) carry forward unchanged.

---

## 6. Data Validation

After pre-tokenization, verify the dataset:

```bash
python -c "
import numpy as np

train = np.load('data/train_tokens.npy', mmap_mode='r')
valid = np.load('data/valid_tokens.npy', mmap_mode='r')

print(f'Train: {len(train):,} tokens ({train.nbytes / 1e9:.2f} GB)')
print(f'Valid: {len(valid):,} tokens ({valid.nbytes / 1e9:.2f} GB)')

# Check token range (must be < 32000)
print(f'Token range: [{train.min()}, {train.max()}]')

# Check for NaN/invalid
assert train.max() < 32000, f'Invalid token ID: {train.max()}'
assert train.min() >= 0, f'Negative token ID: {train.min()}'
print('Validation OK')
"
```
