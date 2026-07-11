#!/bin/bash
# Build PDF from the paper markdown using pandoc + pdflatex.
# Includes a Unicode pre-processor for math symbols, box-drawing chars,
# SentencePiece markers, and Basque accented characters.
set -e
cd "$(dirname "$0")/.."

INPUT="morpheus-on-device-basque-autocompletion.md"
TMP="/tmp/paper_fixed.md"
OUT="morpheus-on-device-basque-autocompletion.pdf"

# Unicode pre-processor
python3 << 'PY'
with open("morpheus-on-device-basque-autocompletion.md", "r") as f:
    text = f.read()

# Math symbols → plain text (avoids math-mode issues in tables)
text = text.replace("×", "x")
text = text.replace("÷", "/")
text = text.replace("→", "->")
text = text.replace("←", "<-")
text = text.replace("↔", "<->")
text = text.replace("≤", "<=")
text = text.replace("≥", ">=")
text = text.replace("≈", "~")
text = text.replace("≠", "!=")
text = text.replace("±", "+/-")
text = text.replace("Δ", "Delta")
text = text.replace("✓", "[OK]")
text = text.replace("✗", "[X]")

# Box-drawing chars
for c in "─│┌┐└┘├┤┬┴┼":
    text = text.replace(c, "+")

# SentencePiece marker
text = text.replace("▁", "|")

# Emoji and special chars
text = text.replace("⚠", "WARNING:")
text = text.replace("🚀", "")
text = text.replace("✅", "[OK]")
text = text.replace("❌", "[X]")

# Em dashes and special spaces
text = text.replace("—", "---")
text = text.replace("–", "--")
text = text.replace("\u00a0", "~")
text = text.replace("\u202f", " ")
text = text.replace("\u2009", " ")

# Smart quotes
text = text.replace("\u201c", '"')
text = text.replace("\u201d", '"')
text = text.replace("\u2018", "'")
text = text.replace("\u2019", "'")

# Bullet chars
text = text.replace("•", "-")
text = text.replace("·", "-")
text = text.replace("◦", "-")

# Greek letters
text = text.replace("β", "beta")
text = text.replace("α", "alpha")
text = text.replace("γ", "gamma")
text = text.replace("δ", "delta")
text = text.replace("ε", "epsilon")
text = text.replace("θ", "theta")
text = text.replace("λ", "lambda")
text = text.replace("μ", "mu")
text = text.replace("σ", "sigma")
text = text.replace("ω", "omega")
text = text.replace("ρ", "rho")

# Subscripts/superscripts
text = text.replace("₁", "_1")
text = text.replace("₂", "_2")
text = text.replace("²", "^2")
text = text.replace("⁵", "^5")

# Misc
text = text.replace("§", "Section ")
text = text.replace("−", "-")
text = text.replace("ł", "l")
text = text.replace("…", "...")
text = text.replace("©", "(c)")
text = text.replace("®", "(R)")

# Literal escape sequences that may appear in code descriptions
text = text.replace("\\ufffd", "U+FFFD")
text = text.replace("\\u2581", "U+2581")

with open("/tmp/paper_fixed.md", "w") as f:
    f.write(text)
print("Unicode fix complete")
PY

echo "Running pandoc + pdflatex..."
pandoc "$TMP" \
  -o "$OUT" \
  --pdf-engine=pdflatex \
  -V geometry:margin=1in \
  -V fontsize=11pt \
  -V linkcolor=blue \
  -V urlcolor=blue \
  --highlight-style=tango \
  --toc \
  --toc-depth=2 \
  -V colorlinks=true \
  -V header-includes="\usepackage[T1]{fontenc}\usepackage[utf8]{inputenc}\usepackage{textcomp}" \
  2>&1 | grep -i "error" || true

echo "PDF generated: $(ls -la "$OUT" | awk '{print $5}') bytes"
