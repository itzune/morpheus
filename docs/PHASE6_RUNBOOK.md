# Phase 6: FIM Continued Pretraining — Runbook

## Prerequisites

| Component | Path | Status |
|-----------|------|--------|
| Pretrained checkpoint (FIM vocab) | `checkpoints/step_0074000_fim.pt` | ✅ Ready (vocab 4016, 553MB) |
| FIM tokenizer | `tokenizer/basque_unigram_fim.model` | ✅ Ready (4004 pieces) |
| FIM training data | `data/train_fim.npy` | ⏳ Building (full corpus, ~4.6B tokens; train caps at 500M) |
| AR validation set | `data/valid_tokens_4k.npy` | ✅ Ready (FIM-for-free check) |
| FIM validation set | `data/valid_fim.npy` | ✅ Ready (2M tokens) |
| Phase 6 config | `config/phase6_fim.yaml` | ✅ Ready |
| train.py with `--pretrained` | `train.py` | ✅ Ready |
| Eval targets | `eval/targets.json` | ✅ Ready |
| GPU | NVIDIA L40 (47.7GB VRAM) | ✅ Available |

## Step 1: Verify FIM training data

```bash
cd /root/morpheus-mamba
python3 -c "
import numpy as np
d = np.load('data/train_fim.npy', mmap_mode='r')
print(f'Shape: {d.shape}, dtype: {d.dtype}')
print(f'Total tokens: {len(d):,}')
print(f'Total sequences (seq_len=1024): {len(d)//1024:,}')
# Check FIM token presence
fim_count = np.sum((d >= 4000) & (d <= 4003))
print(f'FIM tokens (4000-4003): {fim_count:,}')
# Check token range
print(f'Min ID: {d.min()}, Max ID: {d.max()}')
"
```

## Step 2: Launch Phase 6 training

```bash
cd /root/morpheus-mamba
screen -dmS phase6 bash -c "
python3 train.py \
    --config config/phase6_fim.yaml \
    --pretrained checkpoints/step_0074000_fim.pt \
    2>&1 | tee /root/phase6-train.log
"
```

**Key parameters (from config):**
- `total_tokens: 500,000,000` (~3,815 steps)
- `learning_rate: 1.0e-3` (lower than Phase 1's 2.0e-3 — continued pretrain)
- `batch_size: 64, gradient_accumulation: 2` (effective batch 128)
- `seq_len: 1024`
- `warmup_tokens: 5,000,000` (shorter than Phase 1, 1% of total)
- `eval_interval: 500, save_interval: 1000`

**Expected:**
- Training time: ~2.6 hours on L40 (54K tok/s throughput, measured from Phase 1)
- VRAM: ~20-25 GB
- AR valid loss: should NOT regress (FIM-for-free property)
- FIM valid loss: should decrease as model learns infilling
- Can extend with more tokens if eval looks good

**Monitor:**
```bash
tail -f /root/phase6-train.log
# Or check W&B: project "morpheus-v2-mamba"
```

## Step 3: FIM evaluation (P5)

```bash
cd /root/morpheus-mamba

# Evaluate FIM checkpoint (after Phase 6 training)
python3 scripts/fim_eval.py \
    --checkpoint checkpoints/phase6/best.pt \
    --sp-model tokenizer/basque_unigram_fim.model \
    --valid-file data/valid/wiki_valid.txt \
    --n-examples 200 \
    --output /root/fim_eval_results.json

# Baseline: evaluate AR-only checkpoint (before FIM training)
python3 scripts/fim_eval.py \
    --checkpoint checkpoints/step_0074000_fim.pt \
    --sp-model tokenizer/basque_unigram_fim.model \
    --valid-file data/valid/wiki_valid.txt \
    --n-examples 200 \
    --output /root/fim_eval_baseline.json
```

**Metrics:**
- Exact-match rate: does generated middle reconstruct original?
- Char accuracy: Levenshtein-based similarity
- Keystrokes saved: chars saved by accepting completion

## Step 4: Export FIM GGUF (P6)

```bash
cd /root/morpheus-mamba

# Export to HF format
python3 scripts/export/export_hf.py \
    --checkpoint checkpoints/phase6/best.pt \
    --output-dir exports/morpheus-fim-hf \
    --tokenizer tokenizer/basque_unigram_fim.model

# Convert to GGUF (f16)
python /root/llama.cpp-fresh/convert_hf_to_gguf.py \
    exports/morpheus-fim-hf \
    --outfile exports/morpheus-fim.f16.gguf --outtype f16

# Quantize
/root/llama.cpp-fresh/build/bin/llama-quantize \
    exports/morpheus-fim.f16.gguf exports/morpheus-fim.Q5_K_M.gguf Q5_K_M
```

## Step 5: Deploy + dogfood (P6)

```bash
# 1. Start llama-server with FIM model
screen -dmS llama bash -c "
sleep 100000 | /root/llama.cpp-fresh/build/bin/llama-server \
    --model exports/morpheus-fim.Q5_K_M.gguf \
    --port 8080 \
    --ctx-size 2048 \
    --temp 0.2 \
    --top-k 5 \
    2>&1 | tee /root/llama-fim.log
"

# 2. Start the proxy (with FIM template in /v1/complete)
screen -dmS demo bash -c "
cd /root/morpheus-mamba/demo && \
exec /tmp/demo-venv/bin/python -u -m uvicorn server:app \
    --host 0.0.0.0 --port 9090
"

# 3. Run OpenAI conformance smoke test (including FIM route)
/tmp/demo-venv/bin/python /root/morpheus-mamba/demo/test_openai_compat.py

# 4. Point Continue.dev at the proxy
# Copy demo/continue_config.json to ~/.continue/config.json
# Open VS Code, start typing, observe ghost-text completions
```

## Success Criteria

1. **FIM-for-free:** AR valid perplexity does not regress (Δ < 5%)
2. **FIM capability:** Exact-match rate > 5%, char accuracy > 50%
3. **Keystrokes saved:** Positive net keystrokes saved on FIM eval
4. **Server conformance:** All 9+2 smoke test checks pass
5. **Dogfood:** Ghost-text appears in VS Code with both prefix-only and FIM completions
