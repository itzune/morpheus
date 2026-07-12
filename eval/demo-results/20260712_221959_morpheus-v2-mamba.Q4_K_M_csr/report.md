# CSR Eval (real corpus) — morpheus-v2-mamba.Q4_K_M.gguf

- **Date:** 2026-07-12
- **Model:** `morpheus-v2-mamba.Q4_K_M.gguf`
- **Sentences:** 30 (Wikipedia + Berria)
- **Algorithm:** free-acceptance, 1 token/step (matches `evaluate_csr`)

## filtered (deployed)

- macro CSR: **0.2731** (27.31%)
- micro CSR: 0.2746 (27.46%)
- predictions: 869, accepted: 253

## raw (no filters)

- macro CSR: **0.2712** (27.12%)
- micro CSR: 0.2688 (26.88%)
- predictions: 876, accepted: 253

## Filter impact

| mode | macro CSR | micro CSR | accepted |
|------|-----------|-----------|----------|
| filtered (deployed) | 0.2731 | 0.2746 | 253 |
| raw (no filters) | 0.2712 | 0.2688 | 253 |

|gap| = 0.0019
