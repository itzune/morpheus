# CSR Eval (real corpus) — HiTZ.Latxa-Llama-3.1-8B.Q6_K.gguf

- **Date:** 2026-07-19
- **Model:** `HiTZ.Latxa-Llama-3.1-8B.Q6_K.gguf`
- **Sentences:** 30 (Wikipedia + Berria)
- **Algorithm:** free-acceptance, 1 token/step (matches `evaluate_csr`)

## filtered (deployed)

- macro CSR: **0.3318** (33.18%)
- micro CSR: 0.3297 (32.97%)
- predictions: 803, accepted: 323

## raw (no filters)

- macro CSR: **0.3318** (33.18%)
- micro CSR: 0.3297 (32.97%)
- predictions: 803, accepted: 321

## Filter impact

| mode | macro CSR | micro CSR | accepted |
|------|-----------|-----------|----------|
| filtered (deployed) | 0.3318 | 0.3297 | 323 |
| raw (no filters) | 0.3318 | 0.3297 | 321 |

|gap| = 0.0000
