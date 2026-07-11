# CSR Eval (real corpus) — step_0032000.Q4_K_M.gguf

- **Date:** 2026-07-10
- **Model:** `step_0032000.Q4_K_M.gguf`
- **Sentences:** 30 (Wikipedia + Berria)
- **Algorithm:** free-acceptance, 1 token/step (matches `evaluate_csr`)

## filtered (deployed)

- macro CSR: **0.2792** (27.92%)
- micro CSR: 0.2796 (27.96%)
- predictions: 863, accepted: 262

## raw (no filters)

- macro CSR: **0.2780** (27.80%)
- micro CSR: 0.2780 (27.80%)
- predictions: 865, accepted: 260

## Filter impact

| mode | macro CSR | micro CSR | accepted |
|------|-----------|-----------|----------|
| filtered (deployed) | 0.2792 | 0.2796 | 262 |
| raw (no filters) | 0.2780 | 0.2780 | 260 |

|gap| = 0.0012
