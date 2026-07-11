#!/usr/bin/env python3
"""
Quick-win cleaning strategies for Morpheus training data.

Applies strategies 1-4 from the cleaning plan:
  1. URL stripping
  2. Mention & hashtag filtering
  3. Email address removal
  4. Repeated character normalization

Operates line-by-line on raw text files. Output is clean lines -- the
pretokenize.py script then tokenizes from these cleaned lines.

Usage:
    # Single file
    python scripts/pipeline/clean_quick.py -i data/splits/train/part_000.txt -o data/clean/part_000.txt

    # Directory
    python scripts/pipeline/clean_quick.py -i data/splits/train/ -o data/clean/train/

    # Stream from stdin (pipeline)
    cat data/splits/train/*.txt | python scripts/pipeline/clean_quick.py --stdin > data/clean.txt
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Tuple


# ---------------------------------------------------------------------------
# Patterns (compiled once)
# ---------------------------------------------------------------------------

URL_RE = re.compile(r"https?://\S+")
EMAIL_RE = re.compile(r"\S+@\S+\.\S+")
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#(\w+)")

# Emoji: U+2500-27BF (misc symbols, dingbats, play button ▶, heart ❤, star ★)
# + U+1F000-1FFFF (emoticons, transport, symbols, supplemental)
# + U+FE0F (variation selector-16, part of emoji presentation sequences)
_EMOJI_BASE = re.compile(
    r"[\U00002500-\U000027BF"
    r"\U0001F000-\U0001FFFF"
    r"]"
)
# VS16 alone is not emoji, it's the modifier. Use a unified emoji detector:
# Match an emoji base char optionally followed by VS16 + any modifiers.
EMOJI_SEQ_RE = re.compile(
    r"[\U00002500-\U000027BF"
    r"\U0001F000-\U0001FFFF"
    r"]"
    r"(?:\uFE0F"                        # optional VS16
    r"|[\U0001F3FB-\U0001F3FF]"         # or skin-tone modifier
    r"|\u200D[\U0001F000-\U0001FFFF]"   # or ZWJ sequence
    r")*"
)

# For repeated-emoji collapsing, we need a different approach:
# detect emoji sequences (possibly multi-codepoint) and collapse runs.
# Strategy: strip VS16 before doing the run-length check on base chars,
# then re-add one VS16 per emoji in the collapsed run.

REPEATED_PUNCT_RE = re.compile(r"""
    ([!?¿¡])\1{2,}    # !! → !! (keep 2)
  | ([,;:&])\2{1,}      # ,, → , ;  ; → ;  (keep 1, comma/semicolon runs are always typos)
  | ([-_])\3{2,}         # --- → -- (keep 2)
  | ([.])\4{3,}           # .... → ... (keep 3, ellipsis)
  | ([—–])\5{2,}          # —— → — (keep 1, em/en dashes)
""", re.VERBOSE)

def _collapse_punct(m: re.Match) -> str:
    """Collapse repeated punctuation runs.

    - ! ? ¿ ¡ : 3+ → 2 (!! ?? ¿¿ ¡¡)
    - , ; : & : 2+ → 1 (comma/semicolon runs are always typos)
    - - _ : 3+ → 2
    - . : 4+ → 3 (ellipsis)
    - em/en dashes : 3+ → 1
    """
    if m.group(1) is not None:
        return m.group(1) * 2
    if m.group(2) is not None:
        return m.group(2)          # keep 1 for ,;:& runs
    if m.group(3) is not None:
        return m.group(3) * 2
    if m.group(4) is not None:
        return "." * 3
    if m.group(5) is not None:
        return m.group(5)
    return m.group(0)
MULTISPACE_RE = re.compile(r" {2,}")

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_HASHTAG_MIN_LEN = 7    # keep hashtags with ≥7 chars (e.g. #Euskara, #Proposamena)
                               # Basque agglutination: most compound hashtag-words ≥7 chars
                               # Shorthes like #Bilbo (5), #Eusk (4) are proper names/abbrevs → strip
DEFAULT_EMOJI_MAX_RATIO = 0.30  # discard line if >30% of characters are emojis


# ---------------------------------------------------------------------------
# Cleaner class (keeps config instead of globals)
# ---------------------------------------------------------------------------

class QuickCleaner:
    """Applies the four quick-win strategies to a line."""

    def __init__(self, min_hashtag_len: int = DEFAULT_HASHTAG_MIN_LEN,
                 max_emoji_ratio: float = DEFAULT_EMOJI_MAX_RATIO):
        self.min_hashtag_len = min_hashtag_len
        self.max_emoji_ratio = max_emoji_ratio

    # -- helpers --

    def _filter_hashtag(self, match: re.Match) -> str:
        word = match.group(1)
        if len(word) >= self.min_hashtag_len:
            return word
        return ""

    @staticmethod
    def _emoji_ratio(line: str) -> float:
        """Fraction of characters that are part of emoji sequences (U+1F000-1FFFF + modifiers)."""
        if not line:
            return 0.0
        # Count characters covered by emoji sequences
        covered = set()
        for m in EMOJI_SEQ_RE.finditer(line):
            for i in range(m.start(), m.end()):
                covered.add(i)
        return len(covered) / len(line) if covered else 0.0

    @staticmethod
    def _collapse_repeated_emojis(line: str) -> str:
        """Collapse runs of 3+ identical emoji into 2.

        Handles multi-codepoint emoji (base + VS16 + modifiers) by normalizing
        to base codepoint for the run-length comparison, then reconstructing.
        Example: \u25b6\ufe0f\u25b6\ufe0f\u25b6\ufe0f -> \u25b6\ufe0f\u25b6\ufe0f
        """
        seqs = [(m.start(), m.end(), m.group()) for m in EMOJI_SEQ_RE.finditer(line)]
        if len(seqs) < 3:
            return line

        result = []
        cursor = 0
        i = 0

        while i < len(seqs):
            run_start_pos = seqs[i][0]
            base = seqs[i][2][0]  # first codepoint identifies the emoji
            run = [seqs[i]]
            j = i + 1
            while j < len(seqs) and seqs[j][2][0] == base and seqs[j][0] == seqs[j-1][1]:
                run.append(seqs[j])
                j += 1

            # Copy text from cursor to start of this run
            result.append(line[cursor:run_start_pos])

            if len(run) >= 3:
                # Keep only the first 2 emoji of this run
                result.append(line[run_start_pos:run[1][1]])
                # Advance cursor past the ENTIRE run (not just the 2 we kept)
                cursor = run[-1][1]
            else:
                # Keep the run as-is
                result.append(line[run_start_pos:run[-1][1]])
                cursor = run[-1][1]

            i = j

        # Copy any remaining text after the last run
        if cursor < len(line):
            result.append(line[cursor:])

        return "".join(result)

    # -- main entry --

    def clean(self, line: str) -> str:
        """Clean a single line. Returns '' if the line should be discarded."""
        # Strategy 1: URL stripping
        line = URL_RE.sub("", line)

        # Strategy 3: Email removal
        line = EMAIL_RE.sub("", line)

        # Strategy 2: Mention & hashtag filtering
        line = MENTION_RE.sub("", line)
        line = HASHTAG_RE.sub(self._filter_hashtag, line)

        # Strategy 4: Repeated character normalization
        line = self._collapse_repeated_emojis(line)
        line = REPEATED_PUNCT_RE.sub(_collapse_punct, line)

        # Whitespace
        line = MULTISPACE_RE.sub(" ", line).strip()

        # Discard rules (after collapsing — so emoji ratio reflects cleaned text)
        if not line:
            return ""
        if len(line.split()) < 2:
            return ""
        if self._emoji_ratio(line) > self.max_emoji_ratio:
            return ""

        return line


# ---------------------------------------------------------------------------
# File / stream I/O
# ---------------------------------------------------------------------------

def process_file(cleaner: QuickCleaner, input_path: Path,
                 output_path: Path) -> Tuple[int, int]:
    lines_in, lines_out = 0, 0
    with open(input_path, encoding="utf-8", errors="replace") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            lines_in += 1
            cleaned = cleaner.clean(line)
            if cleaned:
                fout.write(cleaned + "\n")
                lines_out += 1
    return lines_in, lines_out


def process_directory(cleaner: QuickCleaner, input_dir: Path,
                      output_dir: Path) -> Tuple[int, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    txt_files = sorted(input_dir.glob("*.txt"))
    if not txt_files:
        print(f"No .txt files found in {input_dir}")
        return 0, 0

    total_in, total_out = 0, 0
    for txt_file in txt_files:
        out_file = output_dir / txt_file.name
        li, lo = process_file(cleaner, txt_file, out_file)
        total_in += li
        total_out += lo
        kept_pct = (lo / li * 100) if li else 0
        print(f"  {txt_file.name}: {li:,} -> {lo:,} lines ({kept_pct:.1f}% kept)")
    return total_in, total_out


def process_stdin(cleaner: QuickCleaner) -> Tuple[int, int]:
    lines_in, lines_out = 0, 0
    for line in sys.stdin:
        lines_in += 1
        cleaned = cleaner.clean(line)
        if cleaned:
            sys.stdout.write(cleaned + "\n")
            lines_out += 1
    return lines_in, lines_out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Quick-win cleaning for Morpheus training data (strategies 1-4)"
    )
    parser.add_argument("-i", "--input", help="Input file or directory")
    parser.add_argument("-o", "--output", help="Output file or directory")
    parser.add_argument("--stdin", action="store_true",
                        help="Read from stdin, write to stdout")
    parser.add_argument("--min-hashtag-len", type=int, default=DEFAULT_HASHTAG_MIN_LEN,
                        help=f"Min hashtag length to keep (default: {DEFAULT_HASHTAG_MIN_LEN})")
    parser.add_argument("--max-emoji-ratio", type=float, default=DEFAULT_EMOJI_MAX_RATIO,
                        help=f"Max emoji ratio before discarding "
                             f"(default: {DEFAULT_EMOJI_MAX_RATIO})")
    args = parser.parse_args()

    cleaner = QuickCleaner(min_hashtag_len=args.min_hashtag_len,
                           max_emoji_ratio=args.max_emoji_ratio)

    if args.stdin:
        li, lo = process_stdin(cleaner)
        print(f"\nstdin: {li:,} -> {lo:,} lines "
              f"({lo / max(li, 1) * 100:.1f}% kept)", file=sys.stderr)
        return

    if not args.input:
        parser.error("Either --input or --stdin is required")
    if not args.output:
        parser.error("--output is required (unless using --stdin)")

    input_path = Path(args.input)
    output_path = Path(args.output)

    if input_path.is_dir():
        print(f"Cleaning directory: {input_path} -> {output_path}")
        total_in, total_out = process_directory(cleaner, input_path, output_path)
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Cleaning file: {input_path} -> {output_path}")
        total_in, total_out = process_file(cleaner, input_path, output_path)

    kept_pct = (total_out / total_in * 100) if total_in else 0
    print(f"\nTotal: {total_in:,} -> {total_out:,} lines ({kept_pct:.1f}% kept)")
    print(f"Removed: {total_in - total_out:,} lines")


if __name__ == "__main__":
    main()
