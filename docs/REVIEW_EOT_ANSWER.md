# Answer: detecting premature truncation in the FIM eval

You asked: *how will you adjust the qualitative keystrokes-saved evaluation
to detect if the new `<EOT>` loss weighting begins causing premature
truncation on longer, multi-word infills?*

## What I built

I added three new metrics to `scripts/fim_eval.py`, specifically designed to
catch the premature-truncation failure mode:

### 1. EOT emission rate
Tracks what % of examples the model stopped on `<EOT>` (vs hitting the 64-token
cap). This is the direct dial we're turning with the 5× weight.

**v2 baseline (no weighting): 76.9%** — the model *does* emit `<EOT>` most of
the time. The 23.1% that don't are what produce runaway text.

### 2. Prefix-truncation rate
For every non-exact-match generation, checks: *is the generated text a proper
prefix of the reference?* (i.e., `reference.startswith(generated)` and
`len(generated) < len(reference)`). This directly detects "model stopped too
early" — it produced the *beginning* of the correct answer but cut it off.

**v2 baseline: 0.0%** — zero premature truncation. The current problem is
purely over-continuation (model doesn't stop), not under-stopping. This is the
metric to watch: if it jumps above ~10% after EOT weighting, the weight is too
aggressive.

### 3. Length-bucketed truncation analysis
Stratifies the prefix-truncation rate by reference length:
- **short** (< 15 chars)
- **medium** (15–30 chars)
- **long** (30+ chars)

This is the key to your specific question about "longer, multi-word infills."
Premature truncation from EOT weighting will hit **long** infills
disproportionately — the model learns "stop early" as a general policy and
truncates the ones that need more tokens. If we see `long` bucket truncation
> 15% while `short` stays near 0%, that's the signature failure mode.

**v2 baseline: 0.0% across all buckets** (28 short, 31 medium, 88 long examples).

## How we'll use it

After v3 training (70/30 + 5× EOT), the eval will show a trajectory like one
of these:

| Scenario | EOT emission | Prefix truncation | Long-bucket truncation | Verdict |
|----------|-------------|-------------------|------------------------|---------|
| Ideal | 90%+ | < 5% | < 10% | Ship it |
| Over-corrected | 98%+ | 15%+ | 25%+ | Reduce EOT weight to 3× |
| Under-corrected | 80% | 0% | 0% | Increase EOT weight to 8× |

The combination of (a) overall prefix-truncation rate and (b) the long-bucket
breakdown gives us a two-dimensional view: *is* the model truncating, and *is
it hitting the long infills specifically?*

## Baseline reference (v2, pre-weighting)

| Metric | v2 (50/50, no EOT weight) |
|--------|---------------------------|
| Exact-match | 2.7% |
| Char accuracy | 31.2% |
| Keystrokes saved | -25.2% |
| Avg gen length | 56.8 chars (ref: 45.0) |
| EOT emission | 76.9% |
| Prefix truncation | 0.0% |
| Truncation (short) | 0.0% (28 ex) |
| Truncation (medium) | 0.0% (31 ex) |
| Truncation (long) | 0.0% (88 ex) |

Saved at `/root/fim_eval_v2_baseline.json` for direct comparison after v3.

## Implementation status

- ✅ `<EOT>` loss weighting (5×) implemented in `train.py` via `F.cross_entropy`
  `weight` parameter (per-class weighting, token id 4003)
- ✅ 70/30 FIM data rebuild launched (`--fim-rate 0.7`, ~50 min)
- ✅ v3 config created (`config/phase6_fim_v3.yaml`): resumes from v2 `best.pt`,
  new data path, `eot_loss_weight: 5.0`
- ✅ Improved `fim_eval.py` with all three new metrics
- ⏳ Training launches after rebuild completes (~50 min)
