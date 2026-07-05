"""
Pre-tokenize corpus into a memory-mapped numpy array for Mamba training.

Reads raw text files from the v1 data splits, tokenizes them with the
SentencePiece Unigram model, and stores the token sequence as a uint16
.npy file for efficient random access during training.

Usage:
    python scripts/pretokenize.py \\
        --sp-model tokenizer/basque_unigram_32k.model \\
        --input-dir data/splits/train \\
        --output data/train_tokens.npy

Full implementation: Morpheus_v2_Mamba.md §4.2
"""

import argparse
import numpy as np
import sentencepiece as spm
from pathlib import Path
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(
        description="Pre-tokenize corpus into numpy array for Mamba training"
    )
    parser.add_argument("--sp-model", required=True, help="Path to SentencePiece .model file")
    parser.add_argument("--input-dir", required=True, help="Directory containing .txt files to tokenize")
    parser.add_argument("--output", required=True, help="Output .npy file path")
    args = parser.parse_args()

    sp = spm.SentencePieceProcessor(model_file=args.sp_model)

    all_tokens = []
    input_dir = Path(args.input_dir)
    seq_sep_token = 0  # <eos> token used as document separator

    for txt_file in sorted(input_dir.glob("*.txt")):
        with open(txt_file) as f:
            for line in tqdm(f, desc=txt_file.name):
                line = line.strip()
                if not line:
                    continue
                tokens = sp.encode(line, out_type=int)
                all_tokens.extend(tokens)
                all_tokens.append(seq_sep_token)

    # Write as memory-mapped uint16 (vocab 32000 < 65535)
    arr = np.array(all_tokens, dtype=np.uint16)
    np.save(args.output, arr)

    print(f"Saved {len(arr):,} tokens to {args.output}")
    print(f"Size: {arr.nbytes / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
