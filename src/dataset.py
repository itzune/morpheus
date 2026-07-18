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


class PackedFimDataset(Dataset):
    """Packed dataset that preserves example boundaries for FIM training.

    Examples in the flat token array are delimited by </s> (eos_id=2).
    This dataset greedily packs whole examples into fixed (seq_len+1) windows,
    ensuring FIM structures ([PRE] prefix [SUF] suffix [MID] middle [EOT])
    are never split across chunk boundaries.

    Without packing, fixed chunk boundaries can sever the [PRE]...[MID]
    structural context, producing broken FIM training signal where the model
    sees [MID] middle [EOT] with no prefix/suffix context — actively harmful.

    The remainder of each window is padded with pad_id (0, <unk>), which is
    ignored in loss via ignore_index=0. Examples longer than seq_len+1 are
    truncated (rare: <0.01% of examples in our corpus).

    This matches the BigCode/StarCoder sequence-packing approach: tokenize
    first, pack whole examples, never split structural context.
    """

    def __init__(self, path: str, seq_len: int = 1024, eos_id: int = 2,
                 pad_id: int = 0, verbose: bool = True):
        self.data = np.load(path, mmap_mode='r')
        self.seq_len = seq_len
        self.eos_id = eos_id
        self.pad_id = pad_id
        self.window = seq_len + 1  # +1 for target shift
        self._fallback_flat = False

        if verbose:
            print(f"  [Packed] Scanning {path}: {len(self.data):,} tokens...")

        # ── Step 1: Find example boundaries by scanning for </s> ──
        eos_positions = self._find_eos_positions()
        n_examples = len(eos_positions)

        if n_examples == 0:
            # No separators: fall back to flat chunking (plain AR data)
            if verbose:
                print(f"  [Packed] No </s> separators found — using flat chunking")
            self._fallback_flat = True
            self.n_sequences = max((len(self.data) - 1) // seq_len, 0)
            self.chunk_starts = None
            self.chunk_lens = None
            return

        # Example i spans [start_i, end_i) where end_i includes the </s>
        starts = np.empty(n_examples, dtype=np.int64)
        starts[0] = 0
        if n_examples > 1:
            starts[1:] = eos_positions[:-1] + 1
        ends = eos_positions + 1  # exclusive end (includes </s>)

        # ── Step 2: Greedy packing — fill windows with whole examples ──
        chunk_starts = []
        chunk_lens = []
        n_truncated = 0
        examples_per_window = []

        cur_start = int(starts[0])
        cur_len = 0
        cur_count = 0

        for i in range(n_examples):
            ex_start = int(starts[i])
            ex_len = int(ends[i]) - ex_start

            if ex_len > self.window:
                # Example too long: truncate (rare). Emit current window first.
                if cur_len > 0:
                    chunk_starts.append(cur_start)
                    chunk_lens.append(cur_len)
                    examples_per_window.append(cur_count)
                    cur_len = 0
                    cur_count = 0
                chunk_starts.append(ex_start)
                chunk_lens.append(self.window)
                examples_per_window.append(1)
                n_truncated += 1
                cur_start = int(ends[i])  # skip remainder, start at next example
            elif cur_len + ex_len > self.window:
                # Doesn't fit in current window: close and start new
                chunk_starts.append(cur_start)
                chunk_lens.append(cur_len)
                examples_per_window.append(cur_count)
                cur_start = ex_start
                cur_len = ex_len
                cur_count = 1
            else:
                # Fits: append to current window
                if cur_len == 0:
                    cur_start = ex_start
                cur_len += ex_len
                cur_count += 1

        # Emit last window
        if cur_len > 0:
            chunk_starts.append(cur_start)
            chunk_lens.append(cur_len)
            examples_per_window.append(cur_count)

        self.chunk_starts = np.array(chunk_starts, dtype=np.int64)
        self.chunk_lens = np.array(chunk_lens, dtype=np.int64)
        self.n_sequences = len(self.chunk_starts)

        if verbose:
            total_data = int(self.chunk_lens.sum())
            total_window = self.n_sequences * self.window
            padding = total_window - total_data
            mean_ex = np.mean(examples_per_window) if examples_per_window else 0
            print(f"  [Packed] {n_examples:,} examples → {self.n_sequences:,} windows")
            print(f"  [Packed] Mean examples/window: {mean_ex:.1f}")
            print(f"  [Packed] Padding: {padding:,} / {total_window:,} tokens "
                  f"({100 * padding / total_window:.1f}%)")
            if n_truncated > 0:
                print(f"  [Packed] Truncated (>{self.window} tokens): {n_truncated}")

    def _find_eos_positions(self):
        """Scan for eos_id positions in chunks to limit memory usage."""
        positions = []
        chunk_size = 50_000_000  # 50M tokens per scan batch
        for start in range(0, len(self.data), chunk_size):
            end = min(start + chunk_size, len(self.data))
            idxs = np.flatnonzero(self.data[start:end] == self.eos_id) + start
            if len(idxs) > 0:
                positions.append(idxs)
        if positions:
            return np.concatenate(positions)
        return np.array([], dtype=np.int64)

    def __len__(self) -> int:
        return self.n_sequences

    def __getitem__(self, idx: int):
        if self._fallback_flat:
            start = idx * self.seq_len
            end = start + self.window
            chunk = self.data[start:end].astype(np.int64)
            if len(chunk) < self.window:
                padded = np.full(self.window, self.pad_id, dtype=np.int64)
                padded[:len(chunk)] = chunk
                chunk = padded
            return torch.from_numpy(chunk[:-1]), torch.from_numpy(chunk[1:])

        start = int(self.chunk_starts[idx])
        length = int(self.chunk_lens[idx])

        chunk = np.full(self.window, self.pad_id, dtype=np.int64)
        chunk[:length] = self.data[start:start + length]

        x = torch.from_numpy(chunk[:-1].copy())
        y = torch.from_numpy(chunk[1:].copy())
        return x, y
