# CSR Eval (real corpus) — morpheus-v2-mamba.Q5_K_M.gguf

- **Date:** 2026-07-12
- **Model:** `morpheus-v2-mamba.Q5_K_M.gguf`
- **Sentences:** 30 (Wikipedia + Berria)
- **Algorithm:** free-acceptance, 1 token/step (matches `evaluate_csr`)

## filtered (deployed)

- macro CSR: **0.2705** (27.05%)
- micro CSR: 0.2746 (27.46%)
- predictions: 869, accepted: 246

## raw (no filters)

- macro CSR: **0.2659** (26.59%)
- micro CSR: 0.2705 (27.05%)
- predictions: 874, accepted: 244

## Filter impact

| mode | macro CSR | micro CSR | accepted |
|------|-----------|-----------|----------|
| filtered (deployed) | 0.2705 | 0.2746 | 246 |
| raw (no filters) | 0.2659 | 0.2705 | 244 |

|gap| = 0.0046
