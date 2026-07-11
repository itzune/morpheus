#!/usr/bin/env python3
"""Fast corpus-quality audit for Morpheus text corpora.

Purpose:
- give a quick go/no-go signal before tokenizer training
- surface visible autocomplete-risk artifacts
- estimate duplicate burden and rough language-mixture burden

This is intentionally heuristic and lightweight. It does NOT replace the full
source-aware audit described in docs/corpus-quality-research.md.

Examples:
    python scripts/pipeline/audit_corpus.py \
        --input data/corpus_sample.txt \
        --output-json reports/corpus_quality_fast_audit.sample.json \
        --output-md reports/corpus_quality_fast_audit.sample.md

    python scripts/pipeline/audit_corpus.py \
        --input data/clean \
        --output-json reports/corpus_quality_fast_audit.clean.json \
        --output-md reports/corpus_quality_fast_audit.clean.md
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional


URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
EMAIL_RE = re.compile(r"\b\S+@\S+\.\S+\b")
MENTION_RE = re.compile(r"(?<!\w)@\w+")
HASHTAG_RE = re.compile(r"(?<!\w)#\w+")
HTML_RE = re.compile(r"&(?:[a-zA-Z]+|#\d+|#x[0-9a-fA-F]+);|<[^>]+>")
REPLACEMENT_RE = re.compile("\uFFFD")
EMOJI_RE = re.compile(r"[\U00002500-\U000027BF\U0001F000-\U0001FFFF]")
REPEATED_PUNCT_RE = re.compile(
    r"([!?¿¡.,;:])\1{2,}|([!?¿¡.,;:])(?:\s*[!?¿¡.,;:]){2,}"
)
WORD_RE = re.compile(r"[\wÀ-ÿ'-]+")
SOURCE_HEADER_RE = re.compile(r".+: \d[\d,]* lines [→\-]+ sampling \d[\d,]* lines")

# Small heuristic stopword inventories only for fast triage.
# These are intentionally tiny and should be interpreted as weak signals.
EU_STOP = {
    "eta", "da", "ez", "izan", "izanen", "gara", "dut", "dugu", "gure",
    "zure", "zuen", "bat", "ere", "bai", "gaur", "hemen", "hor", "hau",
    "hori", "dago", "dira", "zen", "dute", "du", "edo", "baina", "asko",
    "mila", "esker", "ikusi", "euskaraz", "izan", "bezala", "arte", "gisa",
}
ES_STOP = {
    "de", "la", "el", "que", "y", "en", "los", "las", "del", "por",
    "con", "para", "una", "un", "se", "al", "lo", "como", "pero", "mas",
    "más", "sin", "sus", "le", "ya", "o", "este", "esta", "hoy", "hola",
    "gracias", "buenos", "dias", "días",
}


def pct(n: int, d: int) -> float:
    return (100.0 * n / d) if d else 0.0


@dataclass
class AuditResult:
    path: str
    lines_total: int = 0
    lines_blank: int = 0
    lines_source_headers: int = 0
    content_lines: int = 0
    lengths: List[int] = field(default_factory=list)
    metrics: Counter = field(default_factory=Counter)
    duplicates: Counter = field(default_factory=Counter)
    examples: Dict[str, List[str]] = field(default_factory=lambda: defaultdict(list))

    def record_example(self, key: str, line: str, limit: int) -> None:
        if len(self.examples[key]) < limit:
            self.examples[key].append(line)

    def add_line(self, line: str, example_limit: int = 3) -> None:
        self.lines_total += 1
        line = line.rstrip("\n")

        if not line.strip():
            self.lines_blank += 1
            return
        if SOURCE_HEADER_RE.fullmatch(line.strip()):
            self.lines_source_headers += 1
            self.record_example("source_headers", line, example_limit)
            return

        self.content_lines += 1
        self.lengths.append(len(line))
        self.duplicates[line] += 1

        low = line.lower()
        words = WORD_RE.findall(low)
        eu_hits = sum(w in EU_STOP for w in words)
        es_hits = sum(w in ES_STOP for w in words)
        if eu_hits >= 2 and es_hits == 0:
            self.metrics["lang_eu_clean_heuristic"] += 1
        elif eu_hits >= 1 and es_hits >= 1:
            self.metrics["lang_mixed_heuristic"] += 1
        elif es_hits >= 2 and eu_hits == 0:
            self.metrics["lang_non_eu_heuristic"] += 1
        else:
            self.metrics["lang_uncertain_heuristic"] += 1

        checks = {
            "url": URL_RE.search(line),
            "email": EMAIL_RE.search(line),
            "mention": MENTION_RE.search(line),
            "hashtag": HASHTAG_RE.search(line),
            "html": HTML_RE.search(line),
            "replacement_char": REPLACEMENT_RE.search(line),
            "emoji": EMOJI_RE.search(line),
            "punct_run": REPEATED_PUNCT_RE.search(line),
        }
        for key, match in checks.items():
            if match:
                self.metrics[key] += 1
                self.record_example(key, line, example_limit)

        if len(words) < 2:
            self.metrics["lt2_words"] += 1
            self.record_example("lt2_words", line, example_limit)
        if len(line) > 280:
            self.metrics["long_280"] += 1
            self.record_example("long_280", line, example_limit)
        if len(line) > 512:
            self.metrics["long_512"] += 1
            self.record_example("long_512", line, example_limit)

    def finalize(self, top_duplicates: int = 10) -> dict:
        dup_clusters = [(line, count) for line, count in self.duplicates.most_common() if count > 1]
        duplicate_lines = sum(count for _, count in dup_clusters)
        length_summary = {}
        if self.lengths:
            sorted_lengths = sorted(self.lengths)
            idx95 = min(len(sorted_lengths) - 1, math.floor(0.95 * (len(sorted_lengths) - 1)))
            length_summary = {
                "avg": round(sum(self.lengths) / len(self.lengths), 2),
                "median": statistics.median(self.lengths),
                "p95": sorted_lengths[idx95],
                "max": max(self.lengths),
            }

        return {
            "path": self.path,
            "lines_total": self.lines_total,
            "lines_blank": self.lines_blank,
            "lines_source_headers": self.lines_source_headers,
            "content_lines": self.content_lines,
            "length_summary": length_summary,
            "metrics": dict(self.metrics),
            "metric_percentages": {
                key: round(pct(value, self.content_lines), 3)
                for key, value in self.metrics.items()
            },
            "duplicate_summary": {
                "unique_duplicate_lines": len(dup_clusters),
                "lines_in_duplicate_clusters": duplicate_lines,
                "share_lines_in_duplicate_clusters_pct": round(pct(duplicate_lines, self.content_lines), 3),
                "top_duplicates": [
                    {"count": count, "line": line[:240]}
                    for line, count in dup_clusters[:top_duplicates]
                ],
            },
            "examples": dict(self.examples),
        }


def iter_input_files(path: Path) -> Iterable[Path]:
    if path.is_file():
        yield path
        return
    for child in sorted(path.glob("*.txt")):
        if child.is_file():
            yield child


def audit_file(path: Path, example_limit: int) -> dict:
    result = AuditResult(path=str(path))
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            result.add_line(line, example_limit=example_limit)
    return result.finalize()


def aggregate_reports(path: Path, reports: List[dict]) -> dict:
    total_content = sum(r["content_lines"] for r in reports)
    aggregate_metrics = Counter()
    total_lines = total_blank = total_headers = 0
    all_lengths = []
    top_files = []

    for r in reports:
        total_lines += r["lines_total"]
        total_blank += r["lines_blank"]
        total_headers += r["lines_source_headers"]
        aggregate_metrics.update(r["metrics"])
        file_avg = r.get("length_summary", {}).get("avg")
        if file_avg is not None:
            all_lengths.append(file_avg)
        top_files.append({
            "path": r["path"],
            "content_lines": r["content_lines"],
            "emoji_pct": r["metric_percentages"].get("emoji", 0.0),
            "punct_run_pct": r["metric_percentages"].get("punct_run", 0.0),
            "html_pct": r["metric_percentages"].get("html", 0.0),
            "duplicate_cluster_pct": r["duplicate_summary"].get("share_lines_in_duplicate_clusters_pct", 0.0),
        })

    return {
        "path": str(path),
        "lines_total": total_lines,
        "lines_blank": total_blank,
        "lines_source_headers": total_headers,
        "content_lines": total_content,
        "metrics": dict(aggregate_metrics),
        "metric_percentages": {
            key: round(pct(value, total_content), 3)
            for key, value in aggregate_metrics.items()
        },
        "files": reports,
        "highest_risk_files": sorted(
            top_files,
            key=lambda x: (x["html_pct"] + x["punct_run_pct"] + x["duplicate_cluster_pct"]),
            reverse=True,
        )[:10],
    }


def render_markdown(report: dict) -> str:
    is_dir = "files" in report
    lines = []
    lines.append(f"# Fast Corpus Quality Audit\n")
    lines.append(f"**Input:** `{report['path']}`  ")
    lines.append("**Method:** lightweight heuristic audit for go/no-go corpus triage before tokenizer training.  ")
    lines.append("**Caveat:** language-mixture buckets are heuristic, not authoritative LID.\n")

    lines.append("## Summary\n")
    lines.append(f"- Total lines: **{report['lines_total']:,}**")
    lines.append(f"- Blank lines: **{report['lines_blank']:,}**")
    lines.append(f"- Source-header lines: **{report['lines_source_headers']:,}**")
    lines.append(f"- Content lines analyzed: **{report['content_lines']:,}**\n")

    metrics = report.get("metrics", {})
    pcts = report.get("metric_percentages", {})
    key_order = [
        "lang_eu_clean_heuristic",
        "lang_mixed_heuristic",
        "lang_non_eu_heuristic",
        "lang_uncertain_heuristic",
        "url",
        "email",
        "mention",
        "hashtag",
        "html",
        "replacement_char",
        "emoji",
        "punct_run",
        "lt2_words",
        "long_280",
        "long_512",
    ]

    lines.append("## Key Metrics\n")
    lines.append("| Metric | Count | % of content lines |")
    lines.append("|---|---:|---:|")
    for key in key_order:
        if key in metrics:
            lines.append(f"| {key} | {metrics[key]:,} | {pcts.get(key, 0.0):.3f}% |")
    lines.append("")

    if not is_dir and report.get("length_summary"):
        ls = report["length_summary"]
        lines.append("## Length Summary\n")
        lines.append(f"- Avg chars/line: **{ls['avg']}**")
        lines.append(f"- Median chars/line: **{ls['median']}**")
        lines.append(f"- P95 chars/line: **{ls['p95']}**")
        lines.append(f"- Max chars/line: **{ls['max']}**\n")

    dup = report.get("duplicate_summary")
    if dup:
        lines.append("## Duplicate Burden\n")
        lines.append(f"- Unique duplicate lines: **{dup['unique_duplicate_lines']:,}**")
        lines.append(f"- Lines in duplicate clusters: **{dup['lines_in_duplicate_clusters']:,}**")
        lines.append(
            f"- Share of content lines in duplicate clusters: **{dup['share_lines_in_duplicate_clusters_pct']:.3f}%**\n"
        )
        if dup["top_duplicates"]:
            lines.append("### Top duplicate lines\n")
            for item in dup["top_duplicates"][:10]:
                lines.append(f"- **{item['count']}×** {item['line']}")
            lines.append("")

    examples = report.get("examples", {})
    if examples:
        lines.append("## Example Risk Lines\n")
        for key in ["html", "replacement_char", "url", "hashtag", "emoji", "punct_run", "long_280", "lt2_words"]:
            vals = examples.get(key)
            if vals:
                lines.append(f"### {key}")
                for val in vals[:3]:
                    lines.append(f"- {val}")
                lines.append("")

    if is_dir and report.get("highest_risk_files"):
        lines.append("## Highest-risk files (heuristic)\n")
        lines.append("| File | Content lines | html % | punct-run % | duplicate-cluster % |")
        lines.append("|---|---:|---:|---:|---:|")
        for item in report["highest_risk_files"]:
            lines.append(
                f"| `{item['path']}` | {item['content_lines']:,} | {item['html_pct']:.3f}% | {item['punct_run_pct']:.3f}% | {item['duplicate_cluster_pct']:.3f}% |"
            )
        lines.append("")

    lines.append("## Interpretation Guide\n")
    lines.append("- **Go** if the corpus shows low visible-artifact burden and mixed-language content appears plausibly authentic.")
    lines.append("- **Review before training** if duplicate/template burden, HTML residue, wrong-language content, or long noisy lines are materially present.")
    lines.append("- Treat `lang_*_heuristic` as a triage signal only; confirm on stratified manual samples before deleting data.\n")

    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Fast corpus-quality audit")
    parser.add_argument("--input", required=True, help="Input .txt file or directory of .txt files")
    parser.add_argument("--output-json", help="Write JSON report here")
    parser.add_argument("--output-md", help="Write Markdown report here")
    parser.add_argument("--example-limit", type=int, default=3, help="Examples per issue type")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"Input not found: {input_path}")

    files = list(iter_input_files(input_path))
    if not files:
        raise SystemExit(f"No .txt files found under: {input_path}")

    file_reports = [audit_file(path, example_limit=args.example_limit) for path in files]
    report = file_reports[0] if input_path.is_file() else aggregate_reports(input_path, file_reports)

    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.output_md:
        out = Path(args.output_md)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(render_markdown(report), encoding="utf-8")

    print(json.dumps({
        "path": report["path"],
        "content_lines": report["content_lines"],
        "metric_percentages": report.get("metric_percentages", {}),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
