# SETUP.md — Environment Setup and Dependencies

## 1. Overview

Morpheus v2 Mamba requires a **GPU-capable training environment** with CUDA 12.4, PyTorch 2.4+, and the `mamba-ssm` package. This is a significant departure from v1's CPU-only development setup.

**Training hardware:** 1× NVIDIA L40 (48 GB VRAM)
**Inference hardware:** Consumer x86-64 CPU (development can use any machine)

---

## 2. Docker Training Environment (Recommended)

### 2.1 Build the Container

```bash
docker build -t morpheus-mamba-train -f Dockerfile.train .
```

### 2.2 Launch Training

```bash
docker compose -f docker-compose.train.yml run --rm morpheus-train bash
```

Inside the container:

```bash
# Verify GPU access
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.get_device_name(0)}')
print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
"

# Smoke test mamba-ssm
python3 -c "
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from mamba_ssm.models.config_mamba import MambaConfig
import torch

config = MambaConfig(d_model=64, n_layer=2, vocab_size=100)
model = MambaLMHeadModel(config, device='cuda', dtype=torch.bfloat16)
x = torch.randint(0, 100, (1, 16), device='cuda')
out = model(x)
print(f'Smoke test OK — output shape: {out.logits.shape}')
"
```

---

## 3. Native (Non-Docker) Setup

### 3.1 System Requirements

- Ubuntu 22.04+ (or compatible Linux)
- NVIDIA driver ≥ 525 with CUDA 12.4
- Python 3.11+
- `build-essential`, `cmake`, `ninja-build`

### 3.2 Install Python Dependencies

```bash
# Create virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# PyTorch with CUDA 12.4
pip install torch==2.4.* torchvision --index-url https://download.pytorch.org/whl/cu124

# Mamba dependencies (order matters!)
pip install causal-conv1d>=1.4.0 --no-build-isolation
pip install mamba-ssm --no-build-isolation

# Training utilities
pip install sentencepiece==0.2.* wandb safetensors tqdm numpy datasets

# llama.cpp (for export/inference, not training)
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j$(nproc)
cd ..
```

---

## 4. Verify Installation

```bash
python -c "
import torch
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from mamba_ssm.models.config_mamba import MambaConfig

print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')

# Quick smoke test
config = MambaConfig(d_model=64, n_layer=2, vocab_size=100)
model = MambaLMHeadModel(config, device='cuda', dtype=torch.bfloat16)
x = torch.randint(0, 100, (1, 16), device='cuda')
out = model(x)
print(f'Smoke test OK — output shape: {out.logits.shape}')
print(f'Mamba-ssm installed correctly!')
"
```

---

## 5. Link v1 Assets

Morpheus v2 reuses the tokenizer and data splits from v1. Set up symlinks:

```bash
# From the morpheus-mamba directory:

# Tokenizer
ln -s ../morpheus/tokenizer/basque_unigram_32k.model tokenizer/
ln -s ../morpheus/tokenizer/basque_unigram_32k.vocab tokenizer/

# Data splits (if v1 preprocessing is complete)
ln -s ../morpheus/data/splits data/splits
```

If symlinks don't work in your setup, copy the tokenizer files directly:

```bash
cp ../morpheus/tokenizer/basque_unigram_32k.model tokenizer/
cp ../morpheus/tokenizer/basque_unigram_32k.vocab tokenizer/
```

---

## 6. Configuration Files

Model and training configurations are in `config/`:

| File | Description |
|------|-------------|
| `config/small.yaml` | Morpheus-Small (~130M params) — fast iteration, baseline |
| `config/base.yaml` | Morpheus-Base (~200M params) — primary production target |
| `config/large.yaml` | Morpheus-Large (~370M params) — maximum quality fallback |

---

## 7. Troubleshooting

### `mamba-ssm` build fails

- Ensure CUDA 12.4 is properly installed and `nvcc` is on PATH
- Ensure `causal-conv1d` is installed **before** `mamba-ssm`
- Try `MAX_JOBS=4 pip install mamba-ssm --no-build-isolation` to limit compilation parallelism

### VRAM OOM during training

Reduce batch size or sequence length:
```yaml
# config/base.yaml
batch_size: 64       # Default: 128
seq_len: 256         # Default: 512
```

### `torch.compile` errors with mamba-ssm

Try `mode="reduce-overhead"` instead of default, or disable `torch.compile`:
```yaml
compile: false
```

---

## 8. Key Dependencies Reference

| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | ≥ 2.4 | Deep learning framework |
| `mamba-ssm` | ≥ 2.2 | Mamba-2 architecture + CUDA kernels |
| `causal-conv1d` | ≥ 1.4 | Optimized 1D convolution for Mamba |
| `sentencepiece` | ≥ 0.2 | Tokenizer (reuse v1 model) |
| `safetensors` | ≥ 0.4 | HuggingFace-safe model serialization |
| `llama.cpp` | Latest | GGUF conversion + CPU inference |
| `wandb` | Latest | Experiment tracking |
