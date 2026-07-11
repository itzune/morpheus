#!/usr/bin/env python3
"""
Phase 2 + 3 deep-cleaning for Morpheus training data.

Builds on clean_quick.py (strategies 1–4) and adds:

  Phase 2 (clean_phase2.py):
  5. HTML entity & escape cleanup
  6. Sentence splitting for social media / concatenated lines
  7. Long-line heuristics (flag / truncate clearly bad mega-lines)
  8. Repeated-line / boilerplate suppression (within-document)

  Phase 3 (this revision):
  9. Digit-heavy line removal (table rows, IDs, phone numbers)
 10. Decree/legal ID pattern removal
 11. Orphan date line removal

This is a conservative prototype.
The goal is to produce a *cleaner-v2 candidate* for manual review,
NOT to rewrite the full corpus pipeline overnight.

By design, these cleaners are additive and reversible:
- every strategy can be enabled/disabled with a flag
- the original source files are never overwritten
- dry-run mode prints a detailed diff-like before/after sample

Usage:
    python scripts/pipeline/clean_phase2.py \
        --input data/clean/HiTZ_BERnaT-Diverse_BSMauthor.txt \
        --output data/clean-v2/HiTZ_BERnaT-Diverse_BSMauthor.txt \
        --dry-run-lines 200

    python scripts/pipeline/clean_phase2.py \
        --input data/clean \
        --output data/clean-v2 \
        --dry-run-lines 200
"""

from __future__ import annotations

import argparse
import html as html_mod
import re
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Iterable, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------

# Already handled by clean_quick.py, re-declared so this script is
# self-contained for inspection.
URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)
EMAIL_RE = re.compile(r"\S+@\S+\.\S+")
MENTION_RE = re.compile(r"(?<!\w)@\w+")
HASHTAG_RE = re.compile(r"(?<!\w)#(\w+)")
EMOJI_SEQ_RE = re.compile(
    r"[\U00002500-\U000027BF"
    r"\U0001F000-\U0001FFFF"
    r"]"
    r"(?:[\uFE0F"
    r"\U0001F3FB-\U0001F3FF"
    r"\u200D\U0001F000-\U0001FFFF"
    r"])*"
)
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

# ============================================================
# Phase 3 patterns: Number / digit filtering
# ============================================================

# Line where >40% of space-delimited tokens contain a digit
# (catches table rows, sports scores, ID numbers)
DIGIT_HEAVY_LINE_RE = re.compile(r'\b\S*\d\S*\b')

# Long continuous digit sequences (>= 7 digits): phone numbers, DNI, postal codes
LONG_DIGIT_RUN_RE = re.compile(r'\d{7,}')

# Decree/legal ID patterns to remove
#   "7/1990 legea", "255/2012 autoak", "23 ter artikulua", 
#   "11/2015 Legearen", "872/10 2012/7/23ko epaia"
# These appear as noun phrases — we strip the ID fragment only, 
# not the surrounding words.
DECREE_ID_RE = re.compile(
    r'\b\d{1,4}/\d{1,4}'        # 7/1990, 255/2012, 11/2015
    r'(?:\s*/\s*\d{1,4})*'        # optionally more like 872/10 2012/7/23
    r'(?:\s*(?:legea|legean|legearen|legeak|legeetarako'
    r'|autoak|autoetan|dekretua|dekretuaren|dekretuak'
    r'|ebazpena|ebazpenaren|epaia|epaiaren'
    r'|agindua|aginduaren|erregelamendua|erregelamenduaren'
    r'|artikulua|artikuluaren|artikuluan|artikuluak'
    r'|atala|atalaren|idazpurua|kapitulua))?'
    r'|\b\d{1,4}\s+(?:ter|bis|quater)\s+artikulua\b'  # "23 ter artikulua"
)

# Orphan dates — lines that are nothing but a date/ID string
#   "02/16/2022 | 2595 Visits", "2016/04/07 Onartze-data"
ORPHAN_DATE_RE = re.compile(
    r'^\s*\d{1,4}[-/]\d{1,2}[-/]\d{1,4}\s*([|]\s*\d+\s*\w+)?\s*$'
)

# Lines that are pure numbers + separators (phone/ID/DNI lines)
# handled by _is_pure_number_line() using inline logic

# NEW Phase 2 patterns

HTML_TAG_RE = re.compile(r"<[^>]*>")
# Sentence boundary: punctuation PLUS whitespace PLUS uppercase letter
# (typically: end of one sentence, start of another in same line).
SENT_BOUNDARY_RE = re.compile(
    r"(?<=[a-záéíóúüñ])"       # preceded by lowercase letter
    r"([.?!])\s+(?=[A-ZÁÉÍÓÚÜÑ])"  # then punct + whitespace + uppercase
)


# ---------------------------------------------------------------------------
# Cleaner
# ---------------------------------------------------------------------------

class Phase2Cleaner:
    """Apply Phase 2 cleaning strategies to individual lines."""

    def __init__(
        self,
        enable_html: bool = True,
        enable_sentence_split: bool = True,
        enable_long_line: bool = True,
        enable_repeated_line: bool = True,
        enable_digit_filter: bool = True,
        long_line_warn: int = 512,
        long_line_discard: int = 2048,
        min_hashtag_len: int = 7,
        max_emoji_ratio: float = 0.30,
    ):
        self.enable_html = enable_html
        self.enable_sentence_split = enable_sentence_split
        self.enable_long_line = enable_long_line
        self.enable_repeated_line = enable_repeated_line
        self.enable_digit_filter = enable_digit_filter
        self.long_line_warn = long_line_warn
        self.long_line_discard = long_line_discard
        self.min_hashtag_len = min_hashtag_len
        self.max_emoji_ratio = max_emoji_ratio

        # Per-document seen-line cache for repeated-line suppression.
        self._seen: set = set()

    def reset_repeated_cache(self) -> None:
        self._seen = set()

    # -- Phase 1 strategies (repeated from clean_quick.py for completeness) --

    @staticmethod
    def _strip_urls(line: str) -> str:
        return URL_RE.sub("", line)

    @staticmethod
    def _strip_emails(line: str) -> str:
        return EMAIL_RE.sub("", line)

    @staticmethod
    def _clean_mentions(line: str) -> str:
        return MENTION_RE.sub("", line)

    def _clean_hashtags(self, line: str) -> str:
        def _repl(m: re.Match) -> str:
            word = m.group(1)
            return word if len(word) >= self.min_hashtag_len else ""
        return HASHTAG_RE.sub(_repl, line)

    @staticmethod
    def _collapse_repeated_punct(line: str) -> str:
        return REPEATED_PUNCT_RE.sub(_collapse_punct, line)

    @staticmethod
    def _emoji_ratio(line: str) -> float:
        if not line:
            return 0.0
        covered = set()
        for m in EMOJI_SEQ_RE.finditer(line):
            for i in range(m.start(), m.end()):
                covered.add(i)
        return len(covered) / len(line) if covered else 0.0

    @staticmethod
    def _collapse_repeated_emojis(line: str) -> str:
        seqs = [(m.start(), m.end(), m.group()) for m in EMOJI_SEQ_RE.finditer(line)]
        if len(seqs) < 3:
            return line
        result: list = []
        cursor = 0
        i = 0
        while i < len(seqs):
            run_start_pos = seqs[i][0]
            base = seqs[i][2][0]
            run = [seqs[i]]
            j = i + 1
            while j < len(seqs) and seqs[j][2][0] == base and seqs[j][0] == seqs[j - 1][1]:
                run.append(seqs[j])
                j += 1
            result.append(line[cursor:run_start_pos])
            if len(run) >= 3:
                result.append(line[run_start_pos:run[1][1]])
                cursor = run[-1][1]
            else:
                result.append(line[run_start_pos:run[-1][1]])
                cursor = run[-1][1]
            i = j
        if cursor < len(line):
            result.append(line[cursor:])
        return "".join(result)

    # -- Phase 2 strategies (NEW) --

    @staticmethod
    def _html_clean(line: str) -> str:
        """Decode HTML entities; strip leftover tags."""
        line = html_mod.unescape(line)
        line = HTML_TAG_RE.sub("", line)
        return line

    # -- Phase 3 strategies (digit filtering) --

    @staticmethod
    def _digit_token_ratio(line: str) -> float:
        """Fraction of space-delimited tokens that contain at least one digit."""
        tokens = line.split()
        if not tokens:
            return 0.0
        digit_tokens = sum(1 for t in tokens if any(c.isdigit() for c in t))
        return digit_tokens / len(tokens)

    @staticmethod
    def _digit_char_ratio(line: str) -> float:
        """Fraction of characters that are digits."""
        if not line:
            return 0.0
        return sum(1 for c in line if c.isdigit()) / len(line)

    @staticmethod
    def _has_long_digit_run(line: str) -> bool:
        """True if line contains a continuous digit run >= 7 chars."""
        return bool(LONG_DIGIT_RUN_RE.search(line))

    @staticmethod
    def _strip_decree_pattern(line: str) -> str:
        """Remove decree/legal reference numbers from a line.
        
        Strips patterns like "7/1990 legea", "255/2012 autoak", 
        "23 ter artikulua", preserving the rest of the sentence.
        
        Examples:
          "7/1990 legea, uztailaren 3koa" → ", uztailaren 3koa"
          "abenduaren 23ko 11/2015 Legearen testuan" → "abenduaren 23ko testuan"
          "255/2012 autoak." → "."  (empty result → discarded upstream)
        """
        # Match decree ID + optional legal word, anchored with word boundary
        line = DECREE_ID_RE.sub('', line)
        # Clean up artifacts:
        # - double spaces, leading/trailing spaces
        line = MULTISPACE_RE.sub(' ', line).strip()
        # - leading punctuation after removal: ", uztailaren" → "uztailaren"
        line = re.sub(r'^[,;:.\s]+', '', line)
        # - double punctuation: ", ." → "."
        line = re.sub(r'[,;:]\s*[.]', '.', line)
        # - trailing " ." or " ," → just period or nothing
        line = re.sub(r'\s+[.]\s*$', '.', line)
        line = re.sub(r'\s+[,;:]\s*$', '', line)
        return line.strip()

    @staticmethod
    def _is_orphan_date(line: str) -> bool:
        """True if line is nothing but a date/ID with no Basque content."""
        stripped = line.strip()
        if ORPHAN_DATE_RE.match(stripped):
            return True
        # Also catch variant: "02/16/2022" or "2016/04/07" alone
        if re.match(r'^\s*\d{1,4}[-/]\d{1,2}[-/]\d{1,4}\s*$', stripped):
            return True
        return False

    @staticmethod
    def _is_pure_number_line(line: str) -> bool:
        """True if line is essentially just numbers with minimal text.
        
        Catches: phone numbers, ID numbers, currency rows, sports scores.
        E.g., "943 22 46 43", "72257113Z Barruetabeña", "3:2 edo 15:10 (1.50:1)"
        
        Strategy: if >55% of characters are digits, and <5 meaningful
        alphabetic words exist, discard.
        """
        s = line.strip()
        if not s:
            return False
        
        # Count digit characters vs total
        digit_chars = sum(1 for c in s if c.isdigit())
        total_chars = len(s)
        
        # Count words that have at least 50% alpha characters
        words = s.split()
        alpha_words = [w for w in words 
                       if sum(1 for c in w if c.isalpha()) / max(len(w), 1) > 0.5]
        
        # Heavy digit line with almost no alphabetic content
        if digit_chars / max(total_chars, 1) > 0.55 and len(alpha_words) <= 1:
            return True
        
        # Line like "0 0 0 0 0 3 0 3 11.200" — table data
        if digit_chars / max(total_chars, 1) > 0.5 and len(alpha_words) == 0:
            return True
            
        return False

    def _phase3_filter(self, line: str) -> Tuple[str, Optional[str]]:
        """Apply Phase 3 digit filters.
        
        Returns (cleaned_line, discard_reason or None).
        """
        # 1. Pure number lines (phone, ID, table data) → discard
        if self._is_pure_number_line(line):
            return "", "digit_pure"
        
        # 2. Orphan dates → discard
        if self._is_orphan_date(line):
            return "", "digit_orphan_date"
        
        # 3. Lines with long continuous digit runs (>=7 digits) → discard
        if self._has_long_digit_run(line):
            return "", "digit_long_run"
        
        # 4. Lines where >40% of tokens contain digits → discard
        #    (table rows, scores, currency lists)
        if self._digit_token_ratio(line) > 0.4:
            return "", "digit_heavy_tokens"
        
        # 5. Lines where >50% of characters are digits → discard
        if self._digit_char_ratio(line) > 0.5:
            return "", "digit_heavy_chars"
        
        # 6. Strip decree/legal ID patterns from line
        line = self._strip_decree_pattern(line)
        
        # After stripping decree patterns, line might be empty
        if not line or len(line.split()) < 2:
            return "", "digit_decree_stripped"
        
        # 7. Orphan fragment filter: lowercase start, no sentence-ending punct, short
        if line and line[0].islower() and not line[-1] in ('.', '!', '?', '"', '»'):
            # Fragments that are clearly broken: short, no terminal punctuation
            # Colon-ending fragments are often valid list-item introductions
            if len(line) < 50 or line[-1] in (',', ';', '-'):
                return "", "orphan_fragment"
        
        return line, None

    @staticmethod
    def _split_sentences(line: str) -> List[str]:
        """Split a line into sub-sentences at strong Basque sentence boundaries."""
        parts = SENT_BOUNDARY_RE.split(line)
        sentences = []
        # After splitting, SENT_BOUNDARY_RE yields alternating:
        #   text_before_punct, punct_char, text_after
        # We reconstruct full sentences.
        if len(parts) == 1:
            return [parts[0]]
        i = 0
        while i < len(parts):
            if i + 2 < len(parts):
                sentences.append(parts[i] + parts[i + 1] + " " + parts[i + 2])
                i += 3
            else:
                sentences.append(parts[i])
                i += 1
        return [s.strip() for s in sentences if s.strip()]

    def _long_line_check(self, line: str) -> Tuple[str, str]:
        """Return (status, cleaned_line).

        Status is one of: "keep", "warn", "discard"
        """
        n = len(line)
        if n > self.long_line_discard:
            return "discard", ""
        if n > self.long_line_warn:
            return "warn", line
        return "keep", line

    def _wants_split(self, line: str) -> bool:
        """Return True if this line likely contains multiple sentences."""
        if not self.enable_sentence_split:
            return False
        count = len(SENT_BOUNDARY_RE.findall(line))
        return count >= 2

    def _is_repeated(self, line: str) -> bool:
        norm = line.strip().lower()
        if norm in self._seen:
            return True
        self._seen.add(norm)
        return False

    # -- main entry --

    def clean(self, line: str) -> Tuple[str, Optional[str]]:
        """Return (cleaned_line, action_tag or None).

        action_tag values:
          "discard"     — line should be dropped
          "warn"        — line kept but flagged (long, risky)
          "repeated"    — exact duplicate, suppressed
          "split"       — original line was split; caller yields parts
          None          — line kept, no flag
        """
        # Phase 1
        line = self._strip_urls(line)
        line = self._strip_emails(line)
        line = self._clean_mentions(line)
        line = self._clean_hashtags(line)
        line = self._collapse_repeated_emojis(line)
        line = self._collapse_repeated_punct(line)

        # Phase 2
        if self.enable_html:
            line = self._html_clean(line)

        # Whitespace
        line = MULTISPACE_RE.sub(" ", line).strip()

        # Phase 3: digit filtering (before long-line/repeated checks)
        if self.enable_digit_filter:
            line, digit_tag = self._phase3_filter(line)
            if digit_tag is not None:
                return "", digit_tag

        # Emoji-ratio discard (from Phase 1)
        if self._emoji_ratio(line) > self.max_emoji_ratio:
            return "", "discard"

        # Long-line check (Phase 2)
        if self.enable_long_line:
            status, line = self._long_line_check(line)
            if status == "discard":
                return "", "discard"

        # Repeated-line check (Phase 2)
        if self.enable_repeated_line and self._is_repeated(line):
            return "", "repeated"

        # Minimum word check (from Phase 1)
        if len(line.split()) < 2:
            return "", "discard"

        tag = "warn" if (self.enable_long_line and len(line) > self.long_line_warn) else None

        return line, tag


# ---------------------------------------------------------------------------
# File / stream I/O
# ---------------------------------------------------------------------------

def process_file(
    cleaner: Phase2Cleaner,
    input_path: Path,
    output_path: Path,
    dry_run_lines: int = 0,
) -> Tuple[int, int]:
    """Process a single file. Returns (lines_in, lines_out)."""
    lines_in = 0
    lines_out = 0

    with open(input_path, encoding="utf-8", errors="replace") as fin:
        write_fh = None
        if not dry_run_lines:
            write_fh = open(output_path, "w", encoding="utf-8")

        for lineno, raw_line in enumerate(fin, 1):
            lines_in += 1
            line = raw_line.rstrip("\n")

            if not line.strip():
                if write_fh:
                    write_fh.write("\n")
                lines_out += 1
                continue

            # Check if this is a header / source marker line
            cleaned, tag = cleaner.clean(line)
            if cleaned:
                if dry_run_lines and lineno <= dry_run_lines:
                    if tag:
                        print(f"[{tag}] {input_path.name}:{lineno}: {cleaned[:240]}")
                    else:
                        print(f"{input_path.name}:{lineno}: {cleaned[:240]}")
                if cleaner._wants_split(cleaned):
                    for sub in cleaner._split_sentences(cleaned):
                        sub, _ = cleaner.clean(sub)  # re-clean sub-sentences
                        if sub and len(sub.split()) >= 2:
                            if write_fh:
                                write_fh.write(sub + "\n")
                            lines_out += 1
                else:
                    if write_fh:
                        write_fh.write(cleaned + "\n")
                    lines_out += 1
            elif dry_run_lines and lineno <= dry_run_lines:
                print(f"[{tag}] {input_path.name}:{lineno}: <<dropped>> {raw_line[:160]}")
            # else: line dropped; not written

        if write_fh:
            write_fh.close()

    return lines_in, lines_out





def process_directory(
    cleaner: Phase2Cleaner,
    input_dir: Path,
    output_dir: Path,
    dry_run_lines: int = 0,
) -> Tuple[int, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    txt_files = sorted(input_dir.glob("*.txt"))
    if not txt_files:
        print(f"No .txt files in {input_dir}")
        return 0, 0

    total_in = 0
    total_out = 0
    for txt_file in txt_files:
        cleaner.reset_repeated_cache()
        out_file = output_dir / txt_file.name
        li, lo = process_file(cleaner, txt_file, out_file, dry_run_lines=dry_run_lines)
        total_in += li
        total_out += lo
        kept_pct = (lo / li * 100) if li else 0
        print(f"  {txt_file.name}: {li:,} → {lo:,} lines ({kept_pct:.1f}% kept)")
    return total_in, total_out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 2 deep-cleaning for Morpheus training data (strategies 5–8)"
    )
    parser.add_argument("-i", "--input", required=True, help="Input file or directory")
    parser.add_argument("-o", "--output", required=True, help="Output file or directory")
    parser.add_argument(
        "--dry-run-lines", type=int, default=0,
        help="First N lines: print before/after to stdout, do NOT write output files",
    )
    parser.add_argument("--no-html", action="store_true", help="Disable HTML entity cleaning")
    parser.add_argument("--no-split", action="store_true", help="Disable sentence splitting")
    parser.add_argument("--no-long", action="store_true", help="Disable long-line heuristics")
    parser.add_argument("--no-repeat", action="store_true", help="Disable repeated-line suppression")
    parser.add_argument("--no-digits", action="store_true", help="Disable digit/decree filtering")
    parser.add_argument("--min-hashtag-len", type=int, default=7)
    parser.add_argument("--max-emoji-ratio", type=float, default=0.30)
    parser.add_argument("--long-warn", type=int, default=512)
    parser.add_argument("--long-discard", type=int, default=2048)
    args = parser.parse_args()

    cleaner = Phase2Cleaner(
        enable_html=not args.no_html,
        enable_sentence_split=not args.no_split,
        enable_long_line=not args.no_long,
        enable_repeated_line=not args.no_repeat,
        enable_digit_filter=not args.no_digits,
        long_line_warn=args.long_warn,
        long_line_discard=args.long_discard,
        min_hashtag_len=args.min_hashtag_len,
        max_emoji_ratio=args.max_emoji_ratio,
    )

    input_path = Path(args.input)
    output_path = Path(args.output)

    if input_path.is_file():
        lines_in, lines_out = process_file(
            cleaner, input_path, output_path, dry_run_lines=args.dry_run_lines
        )
        print(f"{input_path.name}: {lines_in:,} → {lines_out:,} lines "
              f"({lines_out / lines_in * 100:.1f}% kept)")
    elif input_path.is_dir():
        total_in, total_out = process_directory(
            cleaner, input_path, output_path, dry_run_lines=args.dry_run_lines
        )
        print(f"\nTotal: {total_in:,} → {total_out:,} lines "
              f"({total_out / total_in * 100:.1f}% kept)")
    else:
        sys.exit(f"Input not found: {input_path}")


if __name__ == "__main__":
    main()
