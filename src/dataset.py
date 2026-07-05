"""
Memory-mapped dataset for Mamba causal language model training.

Loads pre-tokenized data from a numpy file via memory mapping, enabling
efficient random access without loading the entire file into RAM.

Usage:
    from src.dataset import MemmapTokenDataset

    ds = MemmapTokenDataset("data/train_tokens.npy", seq_len=512)
    loader = DataLoader(ds, batch_size=128, shuffle=True)

Full implementation: Morpheus_v2_Mamba.md §4.3
"""

import numpy as np
import torch
from torch.utils.data import Dataset


class MemmapTokenDataset(Dataset):
    """Loads pre-tokenized data from a numpy file via memory mapping.

    The data file should be a 1D uint16 numpy array of token IDs.
    Each sequence is seq_len tokens, with an additional token for
    the target (shifted by 1).
    """

    def __init__(self, path: str, seq_len: int = 512):
        self.data = np.load(path, mmap_mode='r')
        self.seq_len = seq_len
        # Number of non-overlapping sequences
        self.n_sequences = (len(self.data) - 1) // seq_len

    def __len__(self) -> int:
        return self.n_sequences

    def __getitem__(self, idx: int):
        start = idx * self.seq_len
        end = start + self.seq_len + 1  # +1 for target shift

        chunk = self.data[start:end].astype(np.int64)

        x = torch.from_numpy(chunk[:-1])   # input:  tokens[0..seq_len-1]
        y = torch.from_numpy(chunk[1:])    # target: tokens[1..seq_len]
        return x, y
