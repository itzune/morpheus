#!/usr/bin/env python3
"""
LLM-based corpus quality audit.

Samples random lines from each source file, sends them to an LLM
for quality assessment, and produces a per-source report.

The LLM evaluates each sample on:
  1. Is this natural Basque text? (1-5)
  2. Is this suitable for an autocomplete training corpus? (1-5)
  3. What kind of text is this? (brief label)
  4. Any quality issues? (gazette numbers, boilerplate, code-switching, etc.)

Usage:
  python3 scripts/audit_corpus_llm.py \
    --data-dir data/clean-v2/ \
    --samples-per-source 20 \
    --output reports/llm_audit.json
"""

import argparse, json, random, sys, time
from pathlib import Path
from collections import defaultdict

# Try importing OpenAI client (works with any OpenAI-compatible API)
try:
    from openai import OpenAI
except ImportError:
    print("openai required: pip install openai", file=sys.stderr)
    sys.exit(1)


# ── LLM prompt ──────────────────────────────────────────────

SYSTEM_PROMPT = """You are a Basque text quality auditor. You evaluate lines of text from a Basque language model training corpus.

For each line, respond with a JSON object:
{
  "is_basque": 1-5,        // 1=not Basque, 3=unclear/mixed, 5=clearly natural Basque
  "autocomplete_fitness": 1-5,  // 1=useless for autocomplete training, 5=excellent
  "text_type": "label",     // one of: prose, dialogue, legal, gazette, news, web_comment, code, poetry, boilerplate, metadata, other
  "issues": ["issue1", ...] // empty if none. Possible issues:
                            //   gazette_numbers - numbers/dates that would train model to emit digits
                            //   boilerplate - legal/template text, not natural language
                            //   code_switching - mixture of Basque and other languages
                            //   non_basque - clearly not Basque
                            //   gibberish - nonsensical character sequences
                            //   repetition - repeated words/phrases
                            //   too_short - too short to be useful
                            //   fragment - incomplete sentence fragment
                            //   encoding_artifact - mojibake or encoding problems
                            //   punctuation_abuse - excessive punctuation
                            //   user_content - user-generated low-quality content
}

Only respond with the JSON object. No explanation.
"""

USER_PROMPT_TEMPLATE = """Evaluate these lines from a Basque training corpus. Source: {source}

Lines:
{lines}"""


# ── Sampling ────────────────────────────────────────────────

def sample_lines(filepath: Path, n: int, max_line_len: int = 500) -> list[tuple[int, str]]:
    """
    Reservoir sample N random lines from a potentially huge file.
    Returns list of (line_number, line_text) tuples.
    Only samples lines <= max_line_len to avoid table/messy content.
    """
    reservoir = []
    seen = 0

    with open(filepath, errors='replace') as f:
        for i, line in enumerate(f):
            line = line.rstrip('\n\r')
            if not line.strip():
                continue
            if len(line) > max_line_len:
                continue

            seen += 1
            if len(reservoir) < n:
                reservoir.append((i + 1, line))
            else:
                j = random.randint(0, seen)
                if j < n:
                    reservoir[j] = (i + 1, line)

    return sorted(reservoir, key=lambda x: x[0])


# ── LLM call ────────────────────────────────────────────────

def evaluate_batch(client: OpenAI, model: str, source: str,
                   lines: list[tuple[int, str]]) -> list[dict]:
    """Send a batch of lines to the LLM for evaluation."""
    formatted = "\n".join(f"[L{ln}] {txt}" for ln, txt in lines)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT_TEMPLATE.format(
                source=source, lines=formatted)},
        ],
        temperature=0.0,
        max_tokens=2000,
    )

    content = response.choices[0].message.content.strip()

    # Try to parse as a JSON array or object
    try:
        parsed = json.loads(content)
        if isinstance(parsed, dict):
            # Single object for all lines — duplicate it
            parsed = [parsed] * len(lines)
        return parsed
    except json.JSONDecodeError:
        # Try extracting JSON from markdown code blocks
        import re
        match = re.search(r'```(?:json)?\s*(\[.*?\]|\{.*?\})\s*```', content, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except:
                pass
        return [{"error": "parse_failed", "raw": content[:200]}] * len(lines)


# ── Main ────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LLM-based corpus quality audit")
    parser.add_argument("--data-dir", default="data/clean-v2/")
    parser.add_argument("--samples-per-source", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=5,
                        help="Lines per LLM call")
    parser.add_argument("--output", default="reports/llm_audit.json")
    parser.add_argument("--model", default=None,
                        help="OpenAI-compatible model name")
    parser.add_argument("--api-base", default=None,
                        help="API base URL")
    parser.add_argument("--api-key", default=None,
                        help="API key")
    parser.add_argument("--source-filter", default=None,
                        help="Only audit this source (filename substring)")
    parser.add_argument("--max-line-len", type=int, default=500)
    args = parser.parse_args()

    # ── Discover source files ──
    data_dir = Path(args.data_dir)
    source_files = sorted(data_dir.glob("*.txt"))
    if not source_files:
        print(f"No .txt files found in {data_dir}")
        sys.exit(1)

    if args.source_filter:
        source_files = [f for f in source_files if args.source_filter in f.name]
        if not source_files:
            print(f"No files matching '{args.source_filter}'")
            sys.exit(1)

    # ── LLM client ──
    api_key = args.api_key or __import__('os').environ.get('OPENAI_API_KEY')
    api_base = args.api_base or __import__('os').environ.get('OPENAI_API_BASE')

    if not api_key:
        print("ERROR: set OPENAI_API_KEY env var or pass --api-key", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(api_key=api_key, base_url=api_base)
    model = args.model or "gpt-4o-mini"  # cheap, fast

    # ── Audit each source ──
    all_results = {}

    for src_path in source_files:
        source_name = src_path.name
        size_mb = src_path.stat().st_size / (1024 * 1024)
        print(f"\n{'='*60}")
        print(f"Auditing: {source_name} ({size_mb:.0f} MB)")
        print(f"{'='*60}")

        # Sample lines
        samples = sample_lines(src_path, args.samples_per_source, args.max_line_len)
        print(f"  Sampled {len(samples)} lines from {src_path}")

        # Evaluate in batches
        evaluations = []
        n = args.samples_per_source
        batch_size = args.batch_size

        for batch_start in range(0, n, batch_size):
            batch = samples[batch_start:batch_start + batch_size]
            print(f"  Evaluating lines {batch[0][0]}-{batch[-1][0]}...", end=" ", flush=True)

            try:
                results = evaluate_batch(client, model, source_name, batch)
                for (line_num, line_text), result in zip(batch, results):
                    result["source"] = source_name
                    result["line_num"] = line_num
                    result["text"] = line_text
                    evaluations.append(result)
                print(f"OK ({len(results)} results)")
            except Exception as e:
                print(f"ERROR: {e}")
                for line_num, line_text in batch:
                    evaluations.append({
                        "source": source_name,
                        "line_num": line_num,
                        "text": line_text,
                        "error": str(e),
                    })

            # Rate limiting
            if batch_start + batch_size < n:
                time.sleep(0.5)

        # Aggregate per source
        scores_basque = [e.get("is_basque", 0) for e in evaluations if "is_basque" in e]
        scores_fitness = [e.get("autocomplete_fitness", 0) for e in evaluations if "autocomplete_fitness" in e]
        issues_count = defaultdict(int)
        type_count = defaultdict(int)
        for e in evaluations:
            for issue in e.get("issues", []):
                issues_count[issue] += 1
            tt = e.get("text_type", "unknown")
            type_count[tt] += 1

        source_result = {
            "source": source_name,
            "size_mb": round(size_mb, 1),
            "samples": len(samples),
            "evaluated": len(evaluations),
            "avg_is_basque": round(sum(scores_basque) / max(len(scores_basque), 1), 2),
            "avg_autocomplete_fitness": round(sum(scores_fitness) / max(len(scores_fitness), 1), 2),
            "text_types": dict(type_count),
            "issues": dict(issues_count),
            "verdict": _verdict(sum(scores_basque), len(scores_basque),
                                sum(scores_fitness), len(scores_fitness),
                                dict(issues_count)),
        }

        all_results[source_name] = source_result

        # Summary
        print(f"  Basque score: {source_result['avg_is_basque']}/5")
        print(f"  Fitness: {source_result['avg_autocomplete_fitness']}/5")
        print(f"  Verdict: {source_result['verdict']}")
        print(f"  Types: {source_result['text_types']}")
        print(f"  Issues: {source_result['issues']}")

    # ── Save ──
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to {output_path}")

    # ── Print summary table ──
    print(f"\n{'Source':40s}  {'Basque':>6s}  {'Fitness':>7s}  Verdict")
    print("-" * 70)
    for name, r in sorted(all_results.items()):
        print(f"{name:40s}  {r['avg_is_basque']:5.1f}/5  {r['avg_autocomplete_fitness']:6.1f}/5  {r['verdict']}")


def _verdict(score_sum, score_n, fitness_sum, fitness_n, issues):
    """Produce a red/yellow/green verdict."""
    avg_score = score_sum / max(score_n, 1)
    avg_fitness = fitness_sum / max(fitness_n, 1)

    # Red flags
    if avg_score < 2.5:
        return "🔴 REJECT — too much non-Basque content"
    if avg_fitness < 2.5:
        return "🔴 REJECT — not suitable for autocomplete training"
    if issues.get("gazette_numbers", 0) > score_n * 0.3:
        return "🔴 REJECT — heavy gazette/numbers contamination"
    if issues.get("code_switching", 0) > score_n * 0.3:
        return "🟡 WARN — significant code-switching"
    if issues.get("boilerplate", 0) > score_n * 0.2:
        return "🟡 WARN — boilerplate content detected"
    if avg_fitness < 3.5:
        return "🟡 WARN — marginal autocomplete fitness"
    return "🟢 PASS — good autocomplete training data"


if __name__ == "__main__":
    main()
