# CSR Eval (real corpus) — step_0054000.Q4_K_M.gguf

- **Date:** 2026-07-10
- **Model:** `step_0054000.Q4_K_M.gguf`
- **Sentences:** 30 (Wikipedia + Berria)
- **Algorithm:** free-acceptance, 1 token/step (matches `evaluate_csr`)

## filtered (deployed)

- macro CSR: **0.2870** (28.70%)
- micro CSR: 0.2871 (28.71%)
- predictions: 854, accepted: 257

## raw (no filters)

- macro CSR: **0.2884** (28.84%)
- micro CSR: 0.2888 (28.88%)
- predictions: 852, accepted: 259

## Filter impact

| mode | macro CSR | micro CSR | accepted |
|------|-----------|-----------|----------|
| filtered (deployed) | 0.2870 | 0.2871 | 257 |
| raw (no filters) | 0.2884 | 0.2888 | 259 |

|gap| = 0.0014
