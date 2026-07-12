# Morpheus: On-Device Predictive Autocompletion for Basque Using State Space Models

> **Preprint — July 2026**
>
> **Authors:** Xabier Ezpeleta
>
> **Code & Models:** `github.com/itzune/morpheus-mamba`
>
> **Live Demo:** Docker-based local deployment supporting both Smart Compose–style ghost text and smartphone keyboard–style word chips
>
> **Keywords:** predictive autocompletion, Basque, Euskara, Mamba, State Space Models, agglutinative languages, on-device inference, keystroke savings, evaluation methodology, next-word prediction, subword tokenization

---

## Abstract

We present **Morpheus**, an on-device predictive autocompletion system for Basque (Euskara), a low-resource agglutinative language. The research question is whether a Gmail Smart Compose–equivalent multi-token continuation system — which Google serves from Cloud TPUs using an ~80M-parameter LSTM trained on ~8 billion emails — can run **entirely on a consumer laptop**, with zero network calls, for use in text editors. We select Mamba-2, a Selective State Space Model offering constant-memory O(1) per-step inference with no KV cache — the same property (no KV cache) that led Google to choose LSTM over Transformer for Smart Compose, but achieved here with a modern architecture that runs on-device.

Our model (91M parameters, 55 MB Q4_K_M) is trained on ~10B tokens of curated Basque text using a 4K-vocabulary SentencePiece Unigram tokenizer. **A controlled vocabulary-size ablation** across 4K, 8K, 16K, and 32K reveals that MorphAcc consistency drops from 66.7% at 4K to 28.6% at 32K, mirroring the QuechuaTok finding. We argue that **fertility correlates negatively with morphological accuracy**, making it a misleading quality metric for agglutinative tokenizer evaluation: lower fertility is achieved exactly by fusing morphemes into opaque units. We note this finding is based on MorphAcc across 4 vocabulary sizes; while it replicates QuechuaTok's result on a different language family, we did not train full models at 8K/16K/32K to verify downstream PPL (§4.4.5).

After full training (76K steps, 14.9 hours), the best checkpoint (step 74K) achieves **held-out PPL of 7.13**, **25.3% CSR** (95% CI [24.0%, 26.5%]), and **76% MorphAcc**. A key methodological finding emerges: **PPL is the only metric that consistently produces coherent, reliable signal** for checkpoint ranking (7.56 → 7.17 → 7.13, all 14 evaluation files agree). The autocomplete-specific metrics proved fragile: CSR requires token-ID prompts to avoid BOS/tokenizer divergence bugs (string prompts gave ~4% vs ~28%), produces overlapping confidence intervals at n=300. A sentence-level typing simulation reveals a **CSR paradox**: the model's native Basque achieves the lowest simulated CSR (19.6%), below English (25.5%) and Spanish (29.3%) which represent <1% of training data — a structural artifact of agglutinative word length, not a model deficiency. An **unresolved anomaly**: in the PyTorch backend, *all* autocomplete metrics (CSR, Top-1, Top-3, Top-5 accuracy) move opposite to PPL improvement across the training trajectory — the model becomes more confident but not more accurate at exact-match next-word prediction. Only confidence tracks PPL. The CSR-specific effect is backend-dependent (does not replicate in GGUF) and may be partly a computation artifact.

We document five **inference engineering strategies** — retokenization fallback, sticky merge, top-k exceeding display-k, next-word candidate extraction, and completion logging with replay — that address failure modes unique to deploying subword-tokenized models as interactive keyboards in agglutinative languages. A cross-model comparison using **Bits Per Character (BPC)** shows Morpheus (91M, BPC 0.970) matching GPT-2 eus-euscrawl (124M, BPC 0.981) and approaching Latxa-Qwen3.5-2B (1.88B, BPC 0.822) at 1/20th the parameters. A data scaling analysis finds the model converged at ~8.8B tokens (~1.9 epochs), likely due to **data quality constraints** rather than quantity, suggesting that for low-resource languages, aggressive quality filtering of a 2–5B token corpus would be more efficient than maximizing raw token count.

---

## 1. Introduction

Predictive autocompletion—suggesting the next words as a user types—is a mature technology for high-resource languages. Google's Gmail Smart Compose serves billions of suggestions daily from an ~80M-parameter LSTM on Cloud TPUs. Gboard provides on-device next-word prediction on smartphones with a 1.4M-parameter, 1.4 MB model. GitHub Copilot applies the same paradigm to code, entirely server-side. However, these systems target languages with simple morphology (English, Spanish, Chinese) where next-word prediction is primarily a collocation problem — and none runs a Smart Compose–equivalent multi-token continuation system entirely on-device.

**The research question.** Can a Smart Compose–equivalent system run **entirely locally on a consumer laptop**, without any network calls, for use in text editors? This is the primary question Morpheus investigates. A secondary question — whether such a system can also power **mobile next-word prediction** (the Gboard paradigm) — requires a much smaller model; our 91M model serves as a starting point for that trajectory rather than a finished mobile solution. See §2.1 for a detailed comparison of production systems.

**Basque (Euskara) is fundamentally different.** As a language isolate with agglutinative morphology, a single Basque verb can encode subject, direct object, indirect object, tense, mood, and aspect through suffix chains (e.g., *ikusiko zenizkidakeen* — "you would have been able to see them to me"). A Basque noun takes 12+ case suffixes, plus number and definiteness marking (e.g., *etxeetaraino* — "up to the houses"). This means that predicting the next word is not just about collocations—it requires **morphological productivity**: the ability to generate grammatically correct suffix sequences that the model has never seen as a unit during training.

Recent work on agglutinative language modeling confirms this challenge. QuechuaTok (Contreras, 2026) showed that standard BPE tokenizers achieve only 6.67% morpheme boundary accuracy on Quechua, while morphology-aware tokenization reaches 83.33%. Lane et al. (2022) demonstrated that morph-based word completion for Plains Cree requires explicit morphological segmentation to be usable.

**Morpheus is designed for this challenge.** We build an on-device predictive autocompletion system for Basque that supports two complementary prediction paradigms — **multi-token continuation** (Smart Compose–style inline ghost text for desktop use, accepted with Tab) and **next-word prediction** (smartphone keyboard–style discrete word chips, accepted with a tap) — and that:

1. Runs **entirely locally** on consumer hardware (no cloud dependency, privacy-first)
2. Provides suggestions within **≤ 50ms P90 latency** on a standard x86-64 CPU
3. Achieves **meaningful keystroke savings** for practical utility
4. Handles **Basque morphology** — not just memorized collocations but productive case suffix prediction
5. Supports **Euskañol** (Basque-Spanish code-switching), common in informal communication
6. Fits within **≤ 300 MB** on disk (feasible for browser extensions and desktop applications)
7. Supports **both prediction paradigms** with inference engineering tailored to each

This paper is primarily a **systems paper** — its core question is whether Smart Compose–equivalent autocompletion can run on-device for Basque. The evaluation methodology findings (PPL reliability, CSR paradox) emerged from the challenge of evaluating that system, and the inference engineering strategies (§5.5) from deploying it. We present these methodological contributions as lessons learned, using the system as a case study.

---

## 2. Related Work

### 2.1 Predictive Autocompletion

Two prediction paradigms dominate real-world autocomplete systems. **Multi-token continuation** (Smart Compose, Copilot) generates inline ghost text accepted with Tab. **Next-word prediction** (Gboard) offers discrete word chips accepted with a tap. They differ not only in UX but also in deployment constraints, model scale, and engineering challenges. Table 1 situates Morpheus among production systems.

**Table 1: Production autocomplete systems**

| System | Params | Size | Training Data | Deployment | Architecture | Paradigm |
|--------|--------|------|---------------|-----------|--------------|----------|
| Gboard (on-device NWP) | 1.4M | 1.4 MB | Federated (per-user) | On-device (mobile) | LSTM | Next-word |
| Gmail Smart Compose | ~80M | Server-side | ~8B emails (~320B+ tokens est.) | Cloud TPU (data center) | LSTM | Multi-token |
| GitHub Copilot | Multi-billion | Server-side | Proprietary code corpus | Cloud GPU (Azure) | Transformer (FIM) | Multi-token |
| GPT-2 eus-euscrawl | 124M | ~50 MB (Q4) | ~423M tokens | On-device (desktop) | Transformer | Multi-token |
| **Morpheus v2** | **91M** | **55 MB** (Q4_K_M) | **Latxa Corpus v2 (curated, ~10B tok seen†)** | **On-device (laptop)** | **Mamba-2** | **Both** |
| Latxa-Qwen3.5-2B | 1,882M | ~1.2 GB (Q4) | Latxa Corpus v2 (public, ~4.2B tokens) | High-end only | Qwen3.5 (instruct) | Multi-token |

†~10B tokens seen over ~2.16 epochs (4.62B unique tokens in the curated corpus); see §4.2.

**The KV-cache insight.** Despite Transformers achieving better perplexity, Google chose an LSTM for Smart Compose because Transformer self-attention requires maintaining keys and values from all previous decoding steps, making per-step latency grow with context length (Chen et al., 2019). GitHub Copilot, also Transformer-based, requires an elaborate global proxy infrastructure (HTTP/2, request cancellation, streaming, geographic routing) to achieve <200ms latency (Cheney, 2025). Google solved the latency problem with data-center TPUs; Morpheus solves it with Mamba-2's O(1) per-step inference at the architecture level — enabling the Smart Compose paradigm to run **on-device** on a consumer laptop (91M, 55 MB, zero network calls), comparable in scale to Smart Compose's 80M but without the data-center dependency.

**Data asymmetry.** Smart Compose's training data has two advantages unavailable to Basque: **volume** (~8B emails, ~30–60× our 10B token corpus) and **domain match** (trained on emails, deployed for writing emails — the distribution the model learns is exactly what users produce). Morpheus trains on general Basque prose (Wikipedia, news, literature) but deploys as a general-purpose text editor, producing corpus-induced artifacts (§6.10) where the model over-predicts encyclopedic patterns. For Basque, no large email or conversational corpus exists — data quality is the binding constraint, not quantity (§6.7).

**Mobile vs. desktop.** The 91M model is sized for the desktop multi-token continuation use case (comparable to Smart Compose). The mobile next-word prediction paradigm (Gboard: 1.4M/1.4MB) requires a distilled variant (~5–10M) that has not yet been trained; the inference engineering strategies in §5.5 are designed to carry over.

Trnka & McCoy (2008) defined **Keystroke Savings Rate (KSR)** as the gold standard for word prediction evaluation: the percentage of keystrokes saved by accepting predictions. We adapt this as **Character Savings Rate (CSR)** — a simulation-based metric following the free-acceptance model where accepting a suggestion costs one keystroke (Tab).

### 2.2 Agglutinative Language Modeling

**QuechuaTok** (Contreras, 2026) introduced MorphAcc — morpheme boundary accuracy — showing that standard tokenizers radically underperform on agglutinative languages. BPE achieves fertility 1.636 but only 6.67% MorphAcc on Quechua. We adopt MorphAcc for Basque evaluation and replicate the vocabulary-size finding: our 4K Unigram tokenizer achieves 66.7% MorphAcc consistency (vs 28.6% at 32K), mirroring QuechuaTok's 4K result of 66.67%.

**Plains Cree word completion** (Lane et al., 2022) used a finite-state morphological analyzer (FST) to segment Cree words into morphemes before prediction. They found that morphological segmentation is both the input representation and the evaluation target for agglutinative languages. Their KSR improvements (15-30% over non-morphological baselines) motivate future work on morphology-aware tokenization for Morpheus.

**Euskarazko LLM-ak (Basque LLMs)**: The HiTZ center has released Latxa (Llama-2/3-based, 7B-70B) and Orai NLP has released Kimu (Gemma-2-based, 2B-9B). EvalEU benchmark (itzune.eus/evaleu, 2026) shows Kimu 9B outperforming Latxa 8B on text-relevant tasks (XNLI 74.2% vs 56.7%, EusProficiency 51.2% vs 46.3%). While these models demonstrate strong Basque language modeling, their sizes (8B-70B params) are incompatible with on-device deployment. Latxa and Kimu also inherit their parent models' tokenizers (32K and 256K respectively) without published Basque-specific tokenizer ablations — a gap our vocabulary-size experiment addresses. For baseline comparison (§6.6), we evaluate two HiTZ models at scales comparable to or larger than Morpheus: **HiTZ/gpt2-eus-euscrawl** (GPT-2 small, 124M, trained on the EusCrawl corpus) and **HiTZ/Latxa-Qwen3.5-2B** (1.88B, a Qwen3.5-based instruct model adapted for Basque using the publicly available Latxa Corpus v2, ~4.2B tokens; Sainz et al., 2025). The former represents the small-model regime; the latter represents what a 20× larger instruct-tuned model achieves on the same evaluation corpus.

### 2.3 State Space Models

Mamba (Gu & Dao, 2023) introduced Selective State Space Models as an alternative to Transformers, offering linear-time inference with constant memory — no KV cache. Mamba-2 (Dao & Gu, 2024) reformulated the architecture as Structured State Space Duality, achieving 6-8× Transformer training speed with multi-head SSM support.

**LFM-2.5-230M** (Liquid AI, 2026) demonstrated SSM viability on edge devices: a 230M hybrid Mamba-MLP model running at 42 tok/s on a Raspberry Pi 5 and 213 tok/s on a Galaxy S25 Ultra, within 400 MB memory. This validates the on-device deployment path we pursue.

### 2.4 Evaluation Methodology for Autocomplete

**ChaI-TeA** (2024) is the closest analog to our evaluation: a large-scale benchmark for Chinese input methods with 26K+ test prefixes and a `saved@k` metric. They attempted LLM-as-judge for semantic matching of alternative completions but found it "very challenging" and deferred it — confirming the multiple-valid-completions problem we encounter.

**Kosyak & Tyers (2022)** evaluated predictive text for agglutinative languages using FST-based models and KSR with user studies, confirming that the multiple-valid-forms problem is central to agglutinative autocomplete evaluation.

**WSTypist** (2026) used simulation-based mobile typing evaluation, confirming that simulation metrics are accepted practice when user studies are infeasible, provided they are at scale with a realistic acceptance model.

---

## 3. Architecture Selection

### 3.1 Problem Constraints

Our constraints create a tight optimization problem:

| Constraint | Target | Why |
|-----------|--------|-----|
| **Inference latency** | P90 ≤ 50ms per token | Users perceive > 100ms as lag |
| **Memory footprint** | ≤ 300 MB on disk, ≤ 500 MB RAM | Must fit consumer devices + browser extensions |
| **No network calls** | Zero external dependencies | Privacy requirement; all data stays local |
| **Basque morphology** | Correct case suffix prediction | Core user need, not just collocations |
| **Training budget** | 1× NVIDIA L40 (48 GB), 5-10 days | Realistic for a small research team |
| **Code-switching** | Basque-Spanish (Euskañol) | ~30% of informal Basque text contains Spanish |

### 3.2 Candidate Architectures

We evaluated three candidate architectures against these constraints:

#### Path A: xLSTM (Modern Recurrent)
- 50-100M parameters
- O(1) per-step inference, constant memory, no KV cache
- Training from scratch on curated Basque text (§4.2)
- ONNX Runtime INT8 deployment
- **Pros:** Safest latency, reuses v1 infrastructure, predictable
- **Cons:** No pre-trained Basque knowledge, lower expected quality ceiling

#### Path B: Distilled Transformer (from Kimu 9B)
- 200-500M parameters (pruned + distilled from Orai Kimu 9B)
- O(n) inference with KV cache
- Best expected quality due to pre-trained Basque knowledge
- llama.cpp GGUF deployment
- **Pros:** Highest potential quality (EvalEU-validated teacher), mature ecosystem
- **Cons:** KV cache adds latency unpredictability, tokenizer mismatch (Gemma 256K → must prune), most engineering complexity

#### Path C: Mamba-2 SSM
- 100-300M parameters
- O(1) per-step inference, constant memory, no KV cache
- Parallelizable training (scan operation)
- llama.cpp GGUF deployment
- **Pros:** Best quality-latency balance, future-proof (SSMs are the on-device trajectory), LFM-2.5 validates edge feasibility
- **Cons:** Newer, less battle-tested ecosystem; ONNX support immature (use GGUF instead)

### 3.3 Decision: Mamba-2 (Path C)

We selected Mamba-2 for the following weighted scoring:

| Criterion | Weight | xLSTM | Distilled Transformer | **Mamba-2** |
|-----------|--------|-------|----------------------|-------------|
| Expected prediction quality | 30% | 3 | **5** | **4** |
| Inference latency safety | 25% | **5** | 3 | **5** |
| Engineering effort / risk | 20% | **4** | 2 | 3 |
| Ecosystem maturity | 10% | 3 | **5** | 3 |
| Future-proofing | 10% | 3 | 4 | **5** |
| Training cost | 5% | **5** | 3 | **4** |
| **Weighted Score** | | 3.80 | 3.70 | **4.00** |

Mamba-2 combines the LSTM-like inference properties essential for on-device autocomplete (constant memory, predictable latency) with dramatically better language modeling capacity than recurrent architectures at the same scale. This directly mirrors Google's Smart Compose decision (§2.1): both chose recurrent over Transformer architectures to avoid KV-cache latency, but Mamba-2 solves it at the architecture level rather than with data-center TPUs.

The decision to not pursue Path B (distillation) was driven by practical concerns: the Gemma tokenizer's 256K vocabulary would require aggressive pruning for a 200M model (embedding table alone = 524 MB), and KV cache management on consumer CPU introduces latency variance that violates our 50ms P90 constraint.

---

## 4. Model Architecture & Training

### 4.1 Model Configuration

| Parameter | Morpheus-Small |
|-----------|---------------|
| Architecture | Mamba-2 (pure SSM) |
| d_model | 768 |
| n_layer | 24 |
| d_state | 64 |
| expand | 2 |
| headdim | 64 |
| Total params | **~91M** |
| Vocabulary size | 4,000 |
| Seq length (training) | 1024 |
| Batch size | 64 (×2 accumulation) |
| Effective tokens/step | 131,072 |
| Learning rate | 2e-3 (cosine decay to 1e-5) |
| Warmup tokens | 50M |
| Total training tokens | 10B |
| **Total steps** | **76,294** |

At 4K vocabulary, only 3.4% of parameters are in the embedding table (3.07M of ~91M), compared to 27% at 32K. This parameter efficiency is a key advantage of the small-vocabulary approach for compact models.

### 4.2 Data Curation

**Corpus:** 4.62 billion subword tokens (9.24 GB) from a curated subset of the publicly available **Latxa Corpus v2** (`HiTZ/latxa-corpus-v2`; Etxaniz et al., 2024). To maximize data quality, we omitted 3 of the 14 sub-corpora — `hplt-v1` (83.8% duplicates, replaced by `hplt-v2`), `BOG` (sentence-splitting destroyed legal text into fragments), and `Aldizkariak` (35% boilerplate) — retaining 11 sources with additional deep-cleaning (normalization, deduplication, quality filtering). An LLM-based audit rated the retained sources 4.6/5 on average. Training saw ~10B tokens total (~2.16 epochs over the 4.62B-token corpus).

**Cleaning:** Four-phase data cleaning pipeline applied before tokenization: (1) document re-parsing (encoding normalization, corrupted document detection), (2) form regularity (punctuation boundaries, whitespace normalization), (3) content filtering (validation/test leakage removal, line length filters, outlier removal), (4) deduplication (exact + near-deduplication via MinHash LSH).

**Validation leakage prevention:** 68,755 lines from the held-out validation set (`wiki_valid.txt`) are excluded from training pretokenization via `--exclude-lines-file`, ensuring zero overlap between training and evaluation data.

**Tokenizer:** SentencePiece Unigram with **4,000-token vocabulary**, trained on 9 sources (all 11 except BOPV and BOTHA). See §4.4 for the vocabulary-size ablation that motivated this choice.

**4K tokenizer quality** (verified 2026-07-04): fertility 2.52 tokens/word, 100% roundtrip fidelity, 99.95% character coverage, 107-character alphabet. All core agglutinative patterns split cleanly: `etxe+a` → `▁etxe a` (house + the), `etxe+tik` → `▁etxe tik` (house + from). Multi-layer morphology also resolves: `mendikoak` → `▁mendi ko ak` (mountain + of + the-plural), three distinct morphemes. Compare this to the 32K tokenizer which fused entire inflectional clusters into opaque tokens like `▁etxetik`, `▁etxera` — see §4.4 for the full comparison.

### 4.3 Training

Training was performed on a single NVIDIA L40 GPU (48 GB GDDR6, Ada Lovelace) with the following setup:

- **Framework:** PyTorch 2.4+ with `mamba-ssm` and `causal-conv1d` CUDA kernels
- **Precision:** bfloat16 (BF16) for training stability
- **Optimizer:** AdamW (β₁=0.9, β₂=0.95, weight_decay=0.1)
- **Schedule:** Linear warmup (50M tokens) → cosine decay to 1e-5
- **GPU config:** Power limit 260W, persistence mode enabled
- **Throughput:** ~55k tok/s (settled)
- **W&B run:** `knid3x95` ("rich-forest-14"), project `morpheus-v2-mamba`

**Training progress** (4K model, W&B run `knid3x95`):

| Milestone | Step | Valid Loss | Valid PPL | Tokens Seen | % Complete |
|-----------|------|------------|-----------|-------------|------------|
| Resume point | 16,000 | — | — | 2.10B | 21.0% |
| Checkpoint | 32,000 | 2.0229 | 7.56 | 4.19B | 41.9% |
| Mid-training | 45,000 | 1.9864 | 7.28 | 5.90B | 59.0% |
| Evaluation | 54,000 | 1.9698 | 7.17 | 7.08B | 70.8% |
| **Best (converged)** | **74,000** | **1.9641** | **7.13** | **9.70B** | **97.0%** |
| Final | 76,294 | 1.9641 | 7.13 | 10.0B | 100% |

Validation loss decreased monotonically throughout training (2.05 → 1.96), with PPL improving from 7.8 (step 25K) to 7.13 (step 74K). The model is fully converged — validation loss is flat from step 67K onward (Δ < 0.001). The best checkpoint (step 74K, valid_loss=1.9641, PPL=7.13) is used for all final evaluations and deployed models. Step 76K (final step) achieved identical PPL (7.13) and is not distinguishable in quality. The training budget of ~10B tokens yields a tokens-to-parameters ratio of approximately 110:1 — below the Mosaic inference-optimal (190:1) and MiniCPM small-model-optimal (192:1) recommendations, indicating the data budget is reasonable by modern small-model standards; see §6.7 for a detailed scaling-law analysis.

**Checkpoint integrity:** Checkpoints are saved atomically (write to `.tmp` then `os.replace()`) every 2,000 steps. A file-based stop monitor detects checkpoint completion via size-stability polling (3 consecutive unchanged polls + size ≥ 540MB) and sends SIGINT for clean W&B flush. The pre-training validation protocol (corpus audit, proxy overfit test, autocomplete smoke test) is documented in Appendix F.

### 4.4 Tokenizer Strategy: Deep Research and Implications

The choice of tokenizer is the single most consequential design decision for agglutinative language modeling. Our tokenizer research spanned seven recent papers (2024–2026) covering 70+ languages and multiple agglutinative language families.

#### 4.4.1 Six Papers That Reshaped Our Understanding

Our tokenizer research spanned six recent papers (2024–2026) covering 70+ languages and multiple agglutinative language families. Three findings converge:

1. **Fertility is misleading for agglutinative tokenizers.** QuechuaTok (Contreras, 2026) — the most directly relevant work — evaluated BPE, Unigram, WordPiece, and morphology-aware PRPE on Southern Quechua (a suffixing agglutinative language structurally analogous to Basque). BPE achieves the lowest fertility (1.636) by memorizing entire polymorphemic words as single tokens, yet achieves only **6.67% MorphAcc**. Unigram at 4K achieves **66.67% MorphAcc**, dropping to 26.67% at 8K. We replicate this finding for Basque (§4.4.4): the negative correlation between fertility and MorphAcc across 4 vocabulary sizes is consistent, though we note that N=4 data points establish correlation, not causation, and we did not train full models at each vocabulary size to verify downstream PPL (§4.4.5).
2. **Unigram > BPE for agglutinative languages.** Xu & Kim (2026) found Unigram consistently outperforms BPE on POS tagging across six Uralic languages. Stephen & Libovický (2026) confirmed Unigram > BPE > WordPiece for morphological alignment, with smaller vocabularies yielding better alignment.
3. **Morphological pre-segmentation improves downstream performance.** García et al. (2025) showed that training BPE on morphologically pre-segmented text improved masked LM performance for Spanish. Hu (2025) found word-level tokenization outperformed BPE for morphologically rich languages under low-resource conditions.

A notable counterpoint: Arnett et al. (2025) found that morphological alignment explains only a small fraction of variance (R² = 0.005–0.024) in downstream performance across 70 languages — for Basque specifically, MorphScore precision was 0.11–0.12, among the lowest of all languages evaluated. This suggests that while morphological alignment is necessary, it is not sufficient for downstream quality at scale.

#### 4.4.2 What Basque LLMs Chose

**Latxa (HiTZ, 2024):** The dominant Basque LLM family (7B–70B) uses Llama 2's BPE tokenizer (32K vocabulary) without modification. The Latxa paper does not discuss tokenizer choices.

**Kimu (Orai NLP, 2025):** The Gemma-2-based Basque model family inherits Gemma's SentencePiece tokenizer (256K vocabulary).

**The open question:** Neither Latxa nor Kimu has published an ablation study on tokenizer impact for Basque. At 7B-70B parameters, models overcome tokenizer deficiencies through scale — but our 91M model cannot afford a suboptimal tokenizer.

#### 4.4.3 Retrospective on Our Decision

Our v1 tokenizer decision chose SentencePiece Unigram at 32K based on fertility 1.71. The 2026 research reveals:

1. **Unigram was correct over BPE.** Xu & Kim (2026) and Stephen & Libovický (2026) support Unigram for agglutinative settings.
2. **32K vocabulary places us in the surface-form memorization regime.** At 32K, the tokenizer memorizes frequent wordforms as atomic units, fragmenting morphemes arbitrarily.
3. **No morphological pre-segmentation was the critical omission.** The literature converges: high morpheme-boundary accuracy requires injecting morphology into tokenizer training.

#### 4.4.3.1 The Morfessor Attempt and Pivot

An initial Morfessor 2.0 pre-segmentation attempt failed due to poor segmentation quality on mixed-language text (the MDL objective cannot distinguish Basque morphology from arbitrary character sequences in foreign words). We pivoted to testing vocabulary size as the primary variable — a simpler experiment with a stronger literature basis. Details in Appendix A.

#### 4.4.4 The Vocabulary Ablation Experiment

We formulated a testable hypothesis:

> The morphological boundary accuracy of a SentencePiece Unigram tokenizer degrades monotonically with vocabulary size because larger vocabularies accumulate frequent surface forms as atomic tokens, fragmenting morpheme boundaries arbitrarily.

**Metric: MorphAcc consistency.** For each test word with a known root–suffix boundary (e.g., `etxe|tik`), we check whether the tokenizer places a token boundary at the morpheme boundary. A word scores `boundary_correct = true` if and only if the root and suffix are in **separate tokens** — not merely substrings of a single fused token. For example, `etxetik` → `▁etxe` `tik` scores ✓ (boundary preserved); `etxetik` → `▁etxetik` scores ✗ (fused into one token). The aggregate metric is the percentage of test words with `boundary_correct = true`.

**Experimental design.** We trained SentencePiece Unigram tokenizers at 4K, 8K, 16K, and 32K on a proportional sample from the full corpus (336.8 MB, 3.5M lines, ~84M tokens, drawn proportionally from all 15 source files). All tokenizers used identical SentencePiece parameters: `model_type=unigram`, `character_coverage=0.9995`, `byte_fallback=True`. Training took ~60s per tokenizer on CPU. The 32K baseline was our existing production tokenizer trained on the same corpus with the same parameters.

**Test words.** We evaluated 21 Basque words covering five roots (`etxe`, `lagun`, `mendi`, `gizon`, `kale`) and eight case suffixes (absolutive `-a`, allative `-ra`, ablative `-tik`, genitive locative `-ko`, inessive `-an`, comitative `-arekin`, benefactive `-arentzat`, causal `-arengatik`). Representative examples:

| Root | Suffix | Test word | Expected boundary |
|------|--------|-----------|-------------------|
| etxe | -a | `etxea` | `etxe\|a` |
| etxe | -tik | `etxetik` | `etxe\|tik` |
| etxe | -arekin | `etxearekin` | `etxe\|arekin` |
| mendi | -ra | `mendira` | `mendi\|ra` |

*(The full 21-word list with per-tokenizer results is available in the supplementary materials.)*

**Results:**

| Tokenizer | Vocab Size | Fertility | MorphAcc Consistency | Change |
|-----------|-----------|-----------|---------------------|--------|
| `baseline-32k` | 32,000 | 1.85 | **28.6%** (6/21) | baseline |
| `raw-16k` | 16,000 | 2.06 | **52.4%** (11/21) | +23.8pp |
| `raw-8k` | 8,000 | 2.28 | **61.9%** (13/21) | +33.3pp |
| `raw-4k` | 4,000 | 2.58 | **66.7%** (14/21) | **+38.1pp** |

The hypothesis was confirmed with striking precision:

1. **The QuechuaTok finding replicates for Basque.** The 4K MorphAcc (66.7%) is nearly identical to QuechuaTok's 4K Unigram result (66.67%).
2. **The degradation is monotonic and accelerating.** Each doubling of vocabulary costs progressively more: 4K→8K loses 4.8pp, 8K→16K loses 9.5pp, 16K→32K loses 23.8pp.
3. **Fertility is the mechanism, not the metric.** The 4K tokenizer has the worst fertility (2.58 vs 1.85 for 32K) because it is forced to decompose — and that forced decomposition is precisely what preserves morpheme boundaries.
4. **4K is the sweet spot for Basque at 91M parameters.** At 4K, every root is a separate token from every case suffix, and verbal agreement morphemes are independently accessible.

**Limitations of this experiment.** The MorphAcc test set is small — 21 words across 5 roots — and the QuechuaTok downstream PPL result is from a different language family (Quechua, Quechuan) applied to Basque (isolate). We did not train full models at 8K/16K/32K to verify downstream PPL for Basque directly. The 4K decision rests on (1) the MorphAcc consistency pattern replicating QuechuaTok, (2) the parameter-efficiency argument (§4.4.5), and (3) the QuechuaTok downstream PPL finding — not on a Basque-specific PPL ablation. Additionally, the higher fertility at 4K (2.58 tokens/word vs 1.85 at 32K) means the 1024-token context window covers ~28% fewer words (~397 vs ~553). For the autocomplete use case, where context is typically the current sentence or paragraph, this trade-off is acceptable; for long-context tasks it would be more consequential.

#### 4.4.4.1 Beyond Nominal Morphology: The Verbmorph Gap

The MorphAcc consistency metric in §4.4.4 tests **nominal morphology** — root + case suffix. The decisive difference between 4K and 32K emerges even more starkly in **verbal morphology**, where Basque's polysynthetic verb structure encodes subject, object, indirect object, tense, and mood in a single word:

| Verb form | Gloss | 4K tokenization | 32K tokenization |
|-----------|-------|----------------|-----------------|
| `dizkizut` | *I have them to you* | `▁di` `zki` `zu` `t` | `▁dizkizut` |
| `dakizkioke` | *he can know them to him* | `▁da` `ki` `zki` `o` `ke` | `▁dakizki` `ok` `e` |
| `zitzaizkidan` | *they were to me* | `▁zitzai` `zki` `dan` | `▁zitzaizkidan` |

**At 4K, the pluralizer `zki` is an independent reusable token** present in all three verbs. A Mamba-2 model can productively recombine `di`, `zki`, `zu`, `t` to form unseen verb inflections. **At 32K, `zki` is buried inside opaque atomic tokens** that share no subword structure.

This is the **fertility paradox**: lower fertility (fewer tokens per word) is achieved exactly by fusing morphemes into opaque units. The fertility metric favors the very behavior that destroys morphological generalization. **Low fertility is the *mechanism* of the surface-form memorization regime, not a desirable property.**

#### 4.4.4.2 Where 4K Still Fails

The 4K tokenizer is not perfect: it struggles with (1) **multi-layer suffixes** where a case suffix itself decomposes into multiple morphemes (e.g., `etxearentzat` → `▁etxe a rentzat` instead of `etxe|arentzat`), and (2) **epenthetic vowels** where Basque inserts `e-` before consonant-initial suffixes (e.g., `lagunetik` → `▁lagun etik` instead of `lagun|tik`). These remaining failures motivate the Apertium pre-segmentation future work (§7.2). Full failure-mode table in Appendix B.

#### 4.4.5 Downstream Perplexity Confirmation

The vocabulary ablation was further validated by downstream perplexity evidence from the QuechuaTok study: 4K Unigram achieves the lowest downstream perplexity among vocabulary sizes tested, confirming that the morphological alignment advantage translates to better language modeling, not just better MorphAcc. However, this is a cross-language-family extrapolation — we did not verify downstream PPL for Basque at 8K/16K/32K directly (see limitations in §4.4.4).

At our model scale, the parameter-efficiency argument is also decisive: at 4K vocab, only 3.4% of the 91M parameters are embeddings (vs 27% at 32K), freeing capacity for the SSM layers that drive sequence modeling quality.

---

## 5. Deployment Pipeline

### 5.1 Export: PyTorch → HuggingFace → GGUF

```
1. PyTorch checkpoint → HuggingFace format (custom export script)
2. HuggingFace → GGUF FP16 (llama.cpp convert_hf_to_gguf.py)
3. GGUF FP16 → Q4_K_M quantization (llama-quantize)
```

The export writes `add_bos_token: false` because the model was trained without BOS. A critical debugging finding: llama-server auto-prepends BOS for string prompts and its built-in SentencePiece tokenizer diverges from the reference library on this 4K vocabulary, producing CSR of ~4% vs ~28% with correct token-ID prompts. The demo server therefore sends token IDs, not strings (§5.4).

### 5.2 Quantization

| Format | Size | BPW | Notes |
|--------|------|-----|-------|
| FP32 (PyTorch checkpoint) | ~547 MB | 32 | Training only |
| FP16 (GGUF) | **181 MB** | 16 | Reference quality |
| Q8_0 (GGUF) | 97 MB | 8 | Near-lossless |
| Q5_K_M (GGUF) | 64 MB | 5.5 | High quality |
| Q4_K_M (GGUF) | **55 MB** | 4.92 | **Deployment default** |

### 5.3 Inference

On the NVIDIA L40 (GPU), the f16 model generates at ~550 tok/s in batch. On consumer x86-64 CPU (Intel Xeon, 8 threads), the Q4_K_M quantized model achieves ~250 tok/s in batch inference via llama-server. Interactive single-token generation is estimated at ~40-80 tok/s.

### 5.4 Demo Server

A Docker-based demo server wraps `llama-server` and serves both prediction paradigms over WebSocket/HTTP, using token-ID prompts to bypass the BOS/tokenizer divergence documented in §5.1. It supports greedy (temperature=0) and sampling modes, ghost-suffix deduplication for inline display, and model hot-reload for checkpoint comparison. The server's next-word candidate logic is the subject of §5.5; its multi-token logic is straightforward greedy continuation.

### 5.4.1 Two Prediction Paradigms

The demo implements two distinct prediction paradigms, which we distinguish throughout the evaluation because they have different user experiences, failure modes, and applicable metrics:

| | **Multi-token continuation** | **Next-word prediction** |
|---|---|---|
| **Metaphor** | Smart Compose (Gmail) | Predictive keyboard (smartphone) |
| **Morpheus model** | 91M (55 MB), on-device | 91M — needs distillation for mobile |
| **Output** | N tokens of gray inline text | 3 discrete word chips |
| **Acceptance** | Tab accepts entire continuation | Tap selects one word |
| **Failure mode** | Repetition loops, ghost-text jitter | Tokenization trap, prediction vanishing |
| **Evaluated by** | CSR (§6) | Completion logging + replay (§5.5.6) |

Multi-token continuation is the simpler inference case: the model greedily extends the context. Next-word prediction is harder because it must produce *whole words* as discrete options, exposing the tokenization trap (§5.5.1). This distinction maps to evaluation: CSR measures multi-token quality, while completion logging measures next-word quality. Conflating them would obscure why PPL improvements are not confirmed by CSR while the keyboard experience benefits from candidate carry-forward.

### 5.5 Inference Engineering for Agglutinative Keyboards

This section addresses the **next-word prediction** paradigm (§5.4.1): deploying a language model as a real-time predictive keyboard that offers whole-word suggestion chips. This is the harder of the two paradigms because it must produce discrete, complete words — which exposes the tokenization trap, a structural failure mode of subword tokenization that does not appear in multi-token ghost-text continuation or in batch evaluation. Each strategy below is motivated by a concrete failure observed during development. The strategies are architecture-agnostic and would carry over directly to a future distilled mobile model (§2.1).

#### 5.5.1 The Tokenization Trap

In an agglutinative language with a 4K subword vocabulary, the same word can be reachable through multiple token paths — and the path the user's partial input lands on may not reach the correct completion.

**Example:** The Basque word *Kaixo* ("hello") tokenizes as `[▁Ka, i, xo]`. But when the user types *Kaix*, the tokenizer segments it as `[▁Ka, ix]` — a different path that cannot reach the `xo` token. The model may know the word perfectly, but the greedy continuation from `[▁Ka, ix]` produces *Kaixan*, *Kaix-*, *Kaixko* — never *Kaixo*.

This is not a model deficiency; it is a structural artifact of subword tokenization. The problem is especially acute in agglutinative languages because long, morphologically complex words have many possible segmentation paths, and short prefixes often land on the wrong one.

#### 5.5.2 Retokenization Fallback

**Strategy:** When generating word-completion candidates, query the model from progressively shorter prefixes in parallel, then filter results by the user's actual typed prefix.

For input *Kaix*, the system queries three paths simultaneously:

| Path | Prefix | Token IDs | Can reach *Kaixo*? |
|------|--------|-----------|---------------------|
| 0 | `Kaix` | `[▁Ka, ix]` | ✗ (wrong path) |
| 1 | `Kai` | `[▁Ka, i]` | ✓ (`xo` is reachable) |
| 2 | `Ka` | `[▁Ka]` | ✓ (but noisier) |

Path 1 reaches `[▁Ka, i]`, where the model predicts `xo` at 54.2% probability. The result *Kaixo* passes the `startswith("Kaix")` filter and surfaces as a candidate. All paths fire in parallel via `asyncio.gather`, keeping latency at ~1× a single call rather than 3×.

A **from-scratch path** (empty prefix, predicting the next word from preceding context only) rescues single-token whole words. For example, *bezala* is a single token (`▁bezala`); when the user types *b*, the token `▁b` cannot reach `▁bezala`. Querying from the preceding context (*"Ni ondo, beti "*) surfaces *bezala* as a next-word prediction, which passes the `startswith("b")` filter.

#### 5.5.3 Sticky Merge: Candidate Carry-Forward

**Problem:** When the model predicts a next word (e.g., *izan* after *idatzia*) and the user types the first letter (*i*), the system switches from next-word prediction to word-completion mode. The token path changes (`▁izan` is one token, but `▁i` + continuation is a different path), and the previously good prediction vanishes from the candidate list — even though the user is typing exactly the word that was predicted.

**Strategy:** Maintain a *sticky pool* of the previous render's candidates. When new candidates arrive, filter the sticky pool by the current typed prefix. Survivors are merged with fresh candidates, receiving a small probability boost (+0.1) to compensate for the fact that cross-path probabilities (next-word vs. word-completion) are not directly comparable.

```
State 1: "...idatzia" → [dago, dezakezu, daiteke, behar, izan]
  (izan at rank 5, not visible in top-3 chips, but stored in sticky pool)

State 2: "...idatzia i" → fresh: [iristeko, itxa, ikur, ...]
  Sticky survivor: izan (prob=0.087, boosted to 0.187)
  Merged result: [iristeko, izan ✓, itxa]
```

The sticky pool resets on chip acceptance and message send, preventing stale candidates from persisting across word boundaries.

#### 5.5.4 Top-k Exceeds Display-k

The keyboard displays 3 suggestion chips but fetches 5 candidates from the server. The extra candidates populate the sticky pool, enabling carry-forward of lower-ranked but relevant predictions. Without this, *izan* (rank 5, prob=0.087) would never enter the sticky pool and could not be rescued in State 2.

#### 5.5.5 Next-Word Candidate Extraction

When the model's greedy continuation at a word-completion level begins with a space (▁-prefixed token), it signals that the model considers the current word complete and is predicting the *next* word. Rather than discarding these tokens (as noise), we extract them as next-word candidates with an `is_next_word` flag. The frontend handles these differently: inserting a leading space before the word, matching the user's expectation that the current word is finished.

This also handles the edge case where the user has typed a complete word without a trailing space (e.g., *"Kaixo, zer"*): the model predicts *moduz* as a next word, and the candidate appears despite no explicit word boundary.

#### 5.5.6 Completion Logging and Replay

Every chip acceptance is logged to a JSONL file with: timestamp, model checkpoint, context, smart context, accepted word and its probability, and all candidates offered. This transforms real user sessions into an evaluation dataset that can be replayed against any checkpoint:

```bash
python scripts/replay_completions.py --models step_0032000.Q4_K_M step_0054000.Q4_K_M
```

The replay script hot-reloads each checkpoint, queries the same contexts, and checks whether the user-accepted word appears in the top-k.

The keyboard candidate algorithm (retokenization fallback, sticky merge, top-k fetch, acceptance semantics) is also ported to PyTorch in `src/eval_utils.py` as `evaluate_next_word_csr`, enabling training-time validation that faithfully reflects the deployed demo. This runs natively on the GPU model (no llama.cpp dependency) during periodic validation, reporting decomposed metrics (Top-1/Top-3/Top-5 accuracy, acceptance rate, average prefix length, average confidence) alongside a simulated CSR. It is used as a **secondary metric** — PPL remains primary for checkpoint ranking — and the decomposed metrics avoid the CSR paradox (§6.12) because they do not conflate model quality with morphological word length.

The replay system also enabled a critical debugging finding: an apparent model regression (step 54K "forgetting" *Kaixo*) was traced to a stale Docker cache running an older `llama.cpp` with a bug in the SSM scan computation for Mamba-2 (`n_groups > 1`, fixed in commit `dc2187d48`), not a model deficiency. **When deploying Mamba-2 models with `llama.cpp`, pin to a build that includes this commit (2025-07-04 or later).** Details in Appendix C.

---

## 6. Evaluation

### 6.1 Evaluation Methodology

We employ a multi-metric evaluation suite designed to capture different aspects of autocomplete quality for agglutinative languages. Each metric has known limitations, and we use them in concert to triangulate true model quality.

#### Metrics

1. **Perplexity (PPL)** — The full next-token distribution quality. Computed on 1.83M held-out tokens (the validation set excluded from training via line-level leakage prevention). This is the smoothest, lowest-variance metric with no exact-match artifact. Computed with training-matching semantics: 1024-token windows, `</s>` separators included in loss, `ignore_index=0` for `<unk>`, no BOS, bfloat16.

2. **Character Savings Rate (CSR)** — Simulates keystroke-by-keystroke typing. For each character in the target completion, the model receives `prompt + typed_so_far` and we check if its top-1 greedy prediction aligns with the remaining target. Acceptance costs 1 keystroke (Tab), following Trnka & McCoy (2008). We report **bootstrap 95% confidence intervals** (1000 resamples) on 300 held-out sentences. *This metric measures the **multi-token continuation** paradigm (§5.4.1): the model produces a greedy continuation and we check character-level alignment.*

3. **Morpheme Boundary Accuracy (MorphAcc)** — For test cases with known morphological segmentation (e.g., `etxe|tik`), we compute whether the model's top-5 predictions include a token that respects the morpheme boundary. 50 tests across 5 roots × 4 case suffixes, with and without context.

4. **Case Paradigm Completion** — For each of 6 Basque nouns, test all 14 grammatical cases (84 total). The model receives the bare root and we check if the correct case suffix ranks in top-K.

5. **Completion Logging + Replay** — Real user chip acceptances are logged with full candidate context and replayed across checkpoints (§5.5.6). *This measures the **next-word prediction** paradigm.*
6. **Keyboard Simulation (next-word)** — Two variants: (a) a frontend-faithful typing simulation (sticky merge, top-3 chips, acceptance semantics) that types 15 translated sentences (5 Basque, 5 English, 5 Spanish) char-by-char, and (b) a PyTorch-native port of the demo keyboard algorithm (`evaluate_next_word_csr` in `src/eval_utils.py`) that runs during training validation on the same 30 CSR test sentences. Both report decomposed metrics: Top-1 accuracy (was the correct word ever the #1 candidate?), Top-3 accuracy (= acceptance rate, was it in the displayed chips?), Top-5 accuracy (was it in the raw fetched pool?), average prefix before acceptance, and average confidence — alongside a simulated CSR. *This is a **secondary metric**; PPL remains primary for checkpoint ranking. The decomposed metrics avoid the CSR paradox (§6.12) because they do not conflate model quality with morphological word length.*
7. **Bits Per Character (BPC)** — Total NLL in bits divided by character count. **Tokenizer-independent**, enabling fair comparison between models with 4K, 50K, and 248K vocabularies. Used for cross-model comparison (§6.6).
8. **Simplified Next-Word CSR (cross-model)** — Greedy decode until a word boundary, extract the first word, compare to target. No inference engineering. Used for fair raw-model comparison across architectures (§6.6).

#### Why multiple metrics

No single metric is sufficient for agglutinative autocomplete. In practice, **PPL is the only metric that consistently produces coherent, reliable signal** for checkpoint ranking. The autocomplete-specific metrics proved fragile to implement and underpowered at available sample sizes — we detail why in §6.8. The metrics serve as sanity checks rather than ranking tools at this scale.

### 6.2 Perplexity (PPL)

PPL is computed on two text sets: (1) the held-out validation set (1.83M tokens, genuinely excluded from training via line-level leakage prevention), and (2) a real corpus of Wikipedia + Berria articles (140K tokens, fetched live July 2026, after the training corpus freeze date).

| Metric | Step 32K | Step 54K | Step 74K | Trend |
|--------|----------|----------|----------|-------|
| **Held-out valid PPL** (clean) | 7.56 (loss 2.0229) | 7.17 (loss 1.9698) | **7.13** (loss 1.9638) | ↓ monotonic |
| **Real corpus PPL** (contaminated) | 10.53 (loss 2.3540) | 9.90 (loss 2.2923) | **9.83** (loss 2.2853) | ↓ monotonic |

**Per-file consistency:** Every single one of the 14 real-corpus files shows monotonic improvement across all three checkpoints, with 32K→54K per-file deltas ranging from −0.54 to −0.89 PPL points and 54K→74K deltas ranging from −0.03 to −0.10. The 54K→74K improvement is small — the model has essentially converged — but it is consistent across all files with no reversals. This is statistically unambiguous.

> **Note:** PPL is not comparable across different vocabulary sizes. The old 32K-vocab model reached PPL=30.78 at step 30K, but this number is not directly comparable to the 4K-vocab model's PPL=7.56 because the vocabulary size affects PPL (smaller vocab = fewer choices per token = lower PPL). All comparisons here use the *same* 4K vocabulary, so they are valid.

### 6.3 Character Savings Rate (CSR)

**Definition:**

$$CSR = 1 - \frac{\text{keystrokes\_needed} + \text{acceptance\_cost}}{\text{len(target\_completion)}}$$

where acceptance_cost = 1 (Tab key to accept the suggestion), following the free-acceptance model of Trnka & McCoy (2008).

**Results** (300 held-out sentences from `wiki_valid.txt`, seed=20260710, f16 GPU inference):

| Metric | Step 32K | Step 54K | Step 74K |
|--------|----------|----------|----------|
| **Macro CSR** | 24.90% | 25.23% | **25.26%** |
| **Micro CSR** | 24.57% | 25.03% | 25.06% |
| **95% CI (macro)** | [23.64%, 26.21%] | [23.98%, 26.48%] | [24.00%, 26.52%] |
| n_tests | 300 | 300 | 300 |

**All confidence intervals overlap → the differences are NOT statistically significant.** 74K is directionally the best (agreeing with PPL), but CSR cannot distinguish the three checkpoints at this quality level. Notably, the 54K→74K improvement in CSR (+0.03pp) is negligible despite continued PPL improvement (7.17→7.13), confirming that CSR saturates once models reach competence.

By target length, short completions (4–6 words) are saturated at all checkpoints (~27%); the 32K→54K improvement concentrates in long targets (10–12 words, +1.39pp) but does not continue to 74K. CSR has saturated.

### 6.4 Morpheme Boundary Accuracy (MorphAcc)

**Results** (50 tests, 5 roots × 4 suffixes × 2 conditions [bare + contextual], f16 GPU):

| Metric | Step 32K | Step 54K | Step 74K |
|--------|----------|----------|----------|
| **MorphAcc** | 70% (35/50) | **76%** (38/50) | **76%** (38/50) |
| Avg boundary prob mass | 17.8% | 19.4% | 19.5% |

MorphAcc improved from 32K to 54K (+6pp) but plateaus at 54K — step 74K shows no further improvement (76%, same 38/50 hits). The average boundary probability mass continues to creep up slightly (19.4% → 19.5%), suggesting marginally more confident morphological predictions, but the hit count has saturated. This is consistent with the tokenizer-bound nature of MorphAcc: once the model learns the morpheme boundaries that the 4K tokenizer makes available, further training cannot improve MorphAcc without a better tokenizer (§7.2).

**Context matters:** Bare nouns (`etxe`, `mendi`) produce probability distributions dominated by punctuation and fragments — the model needs syntactic context to predict case suffixes. With full sentence context (`Bihar...nire etxe → ra`), the model correctly places significant probability mass on the correct suffix. This confirms the Mamba-2 architecture *can* learn morphology — it just needs sufficient context and training.

**Comparison to old 32K-vocab model:** The old 32K-vocab model at step 30K achieved only 20% MorphAcc (1/5 tests). The 4K-vocab model's 70-76% MorphAcc represents a dramatic improvement, validating the vocabulary-size ablation (§4.4.4).

### 6.5 Case Paradigm Completion

**Results** (84 tests: 6 roots × 14 cases, f16 GPU):

| Metric | Step 32K | Step 54K | Step 74K |
|--------|----------|----------|----------|
| **Paradigm Hit@1** | 13.1% (11/84) | 13.1% (11/84) | 10.7% (9/84) |
| **Paradigm Hit@3** | 20.2% (17/84) | 21.4% (18/84) | 22.6% (19/84) |
| **Paradigm Hit@5** | 28.6% (24/84) | 27.4% (23/84) | 27.4% (23/84) |

The paradigm metrics are noisy across checkpoints — Hit@1 actually *drops* from 13.1% to 10.7% between 54K and 74K, while Hit@3 *improves* from 21.4% to 22.6%. Hit@5 is flat. This noise further confirms that the autocomplete-specific metrics do not reliably track model quality: the paradigm test is high-variance (84 tests, bare-root prompts) and small improvements in PPL do not produce monotonic improvements in case suffix prediction. The absolutive case (-a, the citation form) is well-learned (83% Hit@5 at all checkpoints). The ergative (-ak) and inessive (-an) show partial learning. Most other cases remain below threshold — morphology emerges late and requires more context than bare-root prompts provide.

### 6.6 Cross-Model Baseline Comparison

To contextualize Morpheus's performance, we evaluated two external Basque language models under the same evaluation protocol: **HiTZ/gpt2-eus-euscrawl** (GPT-2 small, 124M parameters, trained on the EusCrawl corpus, ~423M tokens) and **HiTZ/Latxa-Qwen3.5-2B** (Latxa, 1.88B parameters, a Qwen3.5-based instruct model adapted for Basque, trained on the publicly available Latxa Corpus v2, ~4.2B tokens; Sainz et al., 2025). All three models were evaluated on the same corpus (`eval/real_corpus/`, 475,750 characters across 14 Wikipedia and Berria news files) and the same 30-sentence CSR test set.

#### BPC: The Correct Cross-Model Metric

Per-token perplexity (PPL) is tokenizer-dependent and cannot be compared across models with different vocabulary sizes. A model with a 4K vocabulary produces more, shorter tokens than one with 50K, inflating per-token PPL without reflecting actual prediction quality. **Bits Per Character (BPC)** normalizes this: it computes the total negative log-likelihood in bits and divides by the number of characters, making it independent of tokenization choices.

| Model | Params | Vocab | BPC | PPL (token) | Tok/Char |
|-------|--------|-------|-----|-------------|----------|
| GPT-2 eus-euscrawl | 124M | 50K | 0.981 | 29.21 | 0.202 |
| **Morpheus v2 (Mamba-2)** | **91M** | **4K** | **0.970** | **9.83** | **0.294** |
| Latxa-Qwen3.5-2B | 1,882M | 248K | 0.822 | 4.89 | 0.359 |

> **Note on shared corpus:** Morpheus and Latxa-Qwen3.5-2B both derive their training data from the Latxa Corpus v2 (§4.2). Morpheus uses a curated subset (11 of 14 sub-corpora, with additional deep-cleaning) trained for ~2.16 epochs (~10B tokens seen); Latxa-Qwen3.5-2B uses the same corpus for continued pretraining of a Qwen3.5 base (~4.2B tokens). The BPC difference between them is therefore attributable to model size (91M vs 1,882M), architecture (Mamba-2 vs Transformer), and training regime (from-scratch vs instruct-tuned continued pretraining) — not to data source differences. GPT-2 eus-euscrawl, by contrast, was trained on EusCrawl only (~423M tokens), making the Morpheus–GPT-2 comparison a natural experiment in data volume (24× difference) at similar parameter scale.

**Key findings:**

1. **Morpheus achieves marginally better BPC than GPT-2** (0.970 vs 0.981), despite having 27% fewer parameters (91M vs 124M). The difference is small and primarily attributable to training data volume: Morpheus was trained on ~10B tokens vs GPT-2's ~423M tokens — a 24× difference. The near-tie in BPC despite the 24× data difference reveals that character-level metrics saturate well before the model has learned the morphological patterns needed for good autocomplete — see §6.7 for a detailed data scaling analysis.

2. **Latxa achieves the lowest BPC** (0.822), as expected for a 1.88B-parameter model — 20× larger than Morpheus. However, the BPC gap (0.148 bits/char) is modest relative to the parameter difference, reflecting diminishing returns from scale.

3. **PPL is misleading across vocabularies.** GPT-2's per-token PPL of 29.21 appears catastrophic compared to Morpheus's 9.83, but BPC reveals the models are nearly equivalent (0.981 vs 0.970). The apparent PPL gap is almost entirely an artifact of GPT-2's 50K vocabulary producing longer tokens that are individually harder to predict.

#### Simplified Next-Word CSR (No Inference Engineering)

To provide a fair raw-model comparison without our inference engineering advantages, we evaluated all three models with a simplified next-word CSR protocol: greedy decode until a word boundary, extract the first word, and compare to the target word.

| Model | CSR (macro) | 95% CI | Word Accuracy |
|-------|-------------|--------|---------------|
| GPT-2 eus-euscrawl | 0.110 | [0.060, 0.167] | 37.6% (56/149) |
| **Morpheus v2** | **0.094** | [0.058, 0.131] | **60.4% (90/149)** |
| Latxa-Qwen3.5-2B | 0.237 | [0.201, 0.275] | 68.5% (102/149) |

**Key findings:**

1. **Latxa has the highest CSR** (0.237) and word accuracy (68.5%), consistent with its superior BPC. Its CIs do not overlap with Morpheus or GPT-2.

2. **GPT-2 and Morpheus have overlapping CIs** — the CSR difference (0.110 vs 0.094) is not statistically significant. However, **Morpheus has 1.6× higher word accuracy** (60.4% vs 37.6%), meaning it predicts the correct word far more often. This is another instance of the CSR paradox (§6.12): Morpheus predicts correct words more frequently but saves fewer keystrokes per correct prediction, because agglutinative Basque words are longer and require more characters before the model converges.

3. **Latxa produces noisy completions.** As an instruct model, Latxa generates web navigation artifacts, markdown headers, and repeated text when used for raw text completion (e.g., the prompt "Euskal Herriko" produces "bidaia-gida/Ibilbideak/Arriurdin mendia"). This highlights a key architectural advantage of Morpheus: as a **base language model** (not instruction-tuned), it is naturally suited for the autocomplete task, where the model must continue text seamlessly rather than follow instructions.

4. **Inference engineering adds 3.9× CSR.** Morpheus's simplified CSR of 0.094 increases to 0.362 with the full inference pipeline (retokenization fallback, sticky merge, top-k alternatives — see §5.5). This demonstrates that the engineering strategies documented in this paper are not marginal optimizations but a major contribution, nearly quadrupling the raw model's autocomplete utility.

### 6.7 Data Scaling Analysis: Training Data Requirements for Low-Resource Language Models

Was the 10B-token training budget appropriately sized? We analyze this through scaling-law context and empirical convergence.

**Tokens-to-parameters ratio.** Our 91M model trained on ~10B tokens yields a ratio of ~**110:1** (tokens per parameter). This sits below the Mosaic inference-optimal (190:1; Sardana et al., 2024) and MiniCPM small-model-optimal (192:1; Hu et al., 2024) recommendations — both more applicable than Chinchilla's 20:1 (compute-optimal, Hoffmann et al., 2022) because our model is deployed as a per-keystroke autocomplete system (the inference-heavy scenario Mosaic addresses). By modern small-model standards, the ratio is not excessive.

| Framework | Ratio | Source |
|-----------|-------|--------|
| Chinchilla (compute-optimal) | 20:1 | Hoffmann et al. (2022) |
| Mosaic (inference-optimal) | 190:1 | Sardana et al. (2024) |
| MiniCPM (small-model optimal) | 192:1 | Hu et al. (2024) |
| **Morpheus (actual)** | **110:1** | — |

**Empirical convergence.** The training trajectory reveals where the model actually stopped learning:

| Training segment | Tokens seen | Δ PPL | Tokens per 1.0 PPL improvement |
|-----------------|------------|-------|-------------------------------|
| Step 32K → 54K | 4.2B → 7.1B | −0.39 | 7.4B |
| Step 54K → 74K | 7.1B → 9.7B | −0.04 | 137.6B |

The improvement rate dropped by **18.6×**. The model effectively converged around **step 67K (~8.8B tokens)** — the final ~1.2B tokens produced no measurable PPL improvement.

**Data quality as the binding constraint.** The convergence at 8.8B tokens is consistent with data quality as the limiting factor. Our corpus quality audit (§4.2, §6.10) identified ~20–30% visible artifacts (social media residue, duplicates, mixed-language content, date/number patterns). If ~20–30% of the corpus is low-value noise, the effective high-quality data is ~7–8B tokens — roughly matching the 8.8B convergence point. DeepSeek's finding that "data quality significantly influences the optimal model/data scaling" (Bi et al., 2024) supports this interpretation.

**Caveat: multi-epoch confound.** An alternative explanation cannot be ruled out: 8.8B tokens over a 4.62B-token corpus is ~1.9 epochs. Diminishing PPL returns after ~2 epochs is a well-known phenomenon in small-model training — the model has simply seen the data twice. The convergence point may reflect multi-epoch saturation rather than (or in addition to) data quality limits. These explanations are not mutually exclusive: if the effective clean data is ~7–8B tokens, the model would exhaust novel high-quality information at ~1.5–1.7 epochs, and the remaining epoch would produce minimal gains. Disentangling these factors would require training on quality-filtered subsets of varying sizes (§7.3).

**Implication.** A 2–5B token corpus of aggressively filtered, high-quality Basque text would likely match or exceed the current 10B token mixed-quality corpus. The practical sweet spot for low-resource agglutinative language modeling at this scale is **2–5B tokens of quality-filtered data** — a finding with implications beyond Basque, since for any language where high-quality text is finite, data quality is the binding constraint, not quantity.

### 6.8 Evaluation Reliability: PPL vs. Autocomplete Metrics

| Signal | Paradigm | Favors 74K? | Significant? | What it measures |
|--------|----------|-------------|-------------|------------------|
| **PPL** (1.83M held-out tokens) | — | ✅ Yes (7.56→7.17→7.13) | ✅ Yes (all 14 files agree) | Language model quality (full distribution) |
| **CSR** (300 held-out sentences) | Multi-token | Directionally (24.90→25.23→25.26) | ❌ CIs overlap | Keystroke economy (lower bound) |
| **Completion replay** (real usage) | Next-word | ✅ Yes (Top-1 60→80%) | — | Chip acceptance hit rate |
| **Typing simulation** (15 sentences) | Next-word | — | — | Word accuracy, CSR paradox |

**Core finding:** PPL is the only metric that produces coherent, trustworthy signal. It unambiguously confirms step 74K is the best model (7.56 → 7.17 → 7.13, all 14 files agree monotonically). The autocomplete metrics proved unreliable for checkpoint ranking:

- **CSR** is fragile to implement (string prompts gave ~4% vs ~28% with token-ID prompts due to BOS and tokenizer divergence bugs) and, even when correct, produces overlapping CIs at n=300. It uses exact-match gold and cannot credit valid alternative Basque continuations — it detects *regression* but not *progress* between competent models. This was confirmed empirically: a 30-sentence eval showed 54K as *worse* than 32K (noise); at n=300, the direction flipped to agree with PPL.
- **MorphAcc** saturates at 76% from step 54K (tokenizer-bound).
- **Completion replay** favors 54K (Top-1 60→80%) but conflates model quality with inference engineering.

**CSR is a lower bound — and structurally biased.** The typing simulation (§6.12) reveals a deeper problem: CSR is not merely resolution-limited, it is **structurally biased against the target language**. Agglutinative morphology requires longer typed prefixes before prediction, consuming the keystroke savings CSR measures. An unresolved anomaly (§6.13) — where *all* autocomplete metrics (CSR, Top-1, Top-3, Top-5) move opposite to PPL improvement in PyTorch — provides further evidence that exact-match metrics can mislead checkpoint selection, though the CSR-specific effect appears backend-dependent.

### 6.9 Overall Assessment at Step 74K (Training Complete)

| Metric | Result | Interpretation |
|--------|--------|----------------|
| Held-out PPL | 7.13 | Strong LM quality; converged (flat from step 67K) |
| CSR (macro, n=300) | 25.26% [24.00, 26.52] | Meaningful keystroke savings; saturated |
| MorphAcc | 76% | Strong morphological competence; saturated at 54K (tokenizer-bound) |
| Paradigm Hit@5 | 27.4% | Partial case system; noisy across checkpoints |
| Q4_K_M size | 55 MB | Well within 300 MB budget |

The model is fully converged. The 54K→74K improvement is visible only in PPL (7.17 → 7.13): CSR barely moves, MorphAcc is flat, Paradigm is noisy. Once a model reaches competence, only PPL has the resolution to measure further improvement.

### 6.10 Corpus-Induced Prediction Artifacts

A persistent quality issue observed during qualitative testing is the model's tendency to autocomplete with **dates, numbers, and temporal expressions** in contexts where a human would predict more general continuations. For example:

| Prompt | Model suggestion | Expected style |
|--------|-----------------|----------------|
| `Aipatu bezala,` | `2015eko ekainean,` | General continuation |

The model predicts `2015eko ekainean` (*"in June 2015"*) rather than a more broadly useful continuation. This is not a random error but a **systematic corpus bias**: the training corpus is dominated by encyclopedic (Wikipedia) and journalistic (Berria) prose, where sentences following phrases like *"as mentioned,"* *"as stated,"* or *"according to"* overwhelmingly reference specific dates, years, and quantities. The model has faithfully learned this distribution.

This artifact persists despite the exclusion of official gazette sources (BOG, BOPV, BOTHA), which were removed precisely because they contained bare numbers in sentence position (decree IDs, budget amounts). The residual date/number bias comes from **legitimate prose** — Wikipedia articles and news articles are inherently rich in temporal and numeric references. This represents a fundamental tension: the same encyclopedic and journalistic sources that provide high-quality, well-formed Basque prose also teach the model that dates and numbers are highly predictable continuations. It is also a manifestation of **domain mismatch**: the model is trained on encyclopedic and journalistic prose but deployed as a general-purpose text editor autocomplete. Google's Smart Compose avoids this problem entirely because its training data (emails) matches its deployment context (writing emails) — the model learns exactly the distribution users produce (§6.7). For Basque, no large conversational or email corpus exists, so we train on the best available general prose and accept the resulting domain artifacts as a known limitation.

**Implications for evaluation:** This artifact is not captured by PPL (predicting frequent date patterns *lowers* PPL) and is only partially captured by CSR (the held-out sentences may or may not contain dates). It is most visible in open-ended autocomplete testing with real user prompts, where the mismatch between the model's learned distribution and the user's intent becomes apparent. This reinforces the finding that **no single metric is sufficient** and that qualitative testing with domain experts remains essential — a conclusion independently reached by the GitHub Copilot team, who report that "language-specific evaluations lead to better outcomes along quality and style preferences" beyond what execution-based testing, LLM-based evaluations, and A/B testing provide (Fu & Mogensen, 2026).

**Mitigation directions** are discussed in §7.1 (data-level) and §7.3 (domain fine-tuning). See also Appendix E for qualitative completion examples across domains.

### 6.11 Cross-Lingual Transfer

Although the model is trained exclusively on Basque-focused corpora, web-crawled sources and parliamentary transcripts inevitably contain small amounts of non-Basque text (~0.6% English by weighted volume; Spanish similarly present). We tested next-word prediction on common collocations in all three languages using the keyboard-mode demo endpoint. The model is **strongly Basque-specialized**: Basque prompts achieve 60% top-1 and 80% top-3 accuracy with 51% average confidence — roughly 3× the top-1 accuracy and 1.7× the confidence of English or Spanish. The model resolves ergative alignment contrasts correctly (`dugu` after a transitive frame, `gara` after an intransitive frame) and handles suffix attachment (`arreta` → `arretagatik`, `Aldez` → `aurretik`). Despite this Basque dominance, the model exhibits **weak incidental cross-lingual transfer** — correctly predicting high-frequency English/Spanish collocations (*Thank you very* → *much*, *Los Estados* → *Unidos*) but with low confidence and failing on less formulaic phrases. This is an **artifact of corpus composition, not a feature**: even aggressively monolingual Basque corpora contain enough multilingual contamination to produce measurable cross-lingual effects. The typing simulation (§6.12) confirms this transfer is functional, not merely collocational: the model sustains full 10–12 word English/Spanish sentences with 72–74% acceptance rate (the model predicts the correct word via progressive prefix completion). Qualitative cross-lingual completion examples are in Appendix E.

| Language | N prompts | Top-1 | Top-3 | Avg. max confidence |
|----------|-----------|-------|-------|---------------------|
| **Basque** | 10 | **60.0%** | **80.0%** | **0.511** |
| English | 30 | 23.3% | 23.3% | 0.196 |
| Spanish | 30 | 16.7% | 30.0% | 0.299 |

---

### 6.12 The CSR Paradox: Agglutinative Morphology Penalizes Native-Language Keystroke Savings

To evaluate the keyboard-mode autocomplete under realistic usage, we developed a typing simulation that **faithfully replicates the frontend algorithm** (§5.5): char-by-char typing, sticky merge, top-3 chip display from a top-5 fetch, and full acceptance semantics. Fifteen sentences — five each in Basque, English, and Spanish — are translations of the same semantic content, controlling for topic and structure. The model accepts a suggestion the moment a matching candidate appears in the top-3 chips.

**Results:**

| Language | Top-1 | Top-3 (accept) | Top-5 | Simulated CSR | Avg. confidence | Avg. prefix |
|----------|:----:|:--------------:|:-----:|:-------------:|:---------------:|:-----------:|
| **Basque** (native) | 35.9% | **79.5%** | 79.5% | **19.6%** | 0.228 | **4.4 chars** |
| English (<1% corpus) | 40.0% | 72.0% | 82.0% | 25.5% | 0.416 | 2.7 chars |
| Spanish (<1% corpus) | **52.2%** | 73.9% | 78.3% | **29.3%** | **0.554** | 3.1 chars |
| **Overall** | 43.0% | 74.8% | 80.0% | 24.8% | 0.405 | 3.3 chars |

**The model's native language achieves the lowest simulated CSR** — below two languages that represent less than 1% of the training corpus. Spanish, the least-represented language, achieves the highest CSR. This is counterintuitive: one would expect the model to perform best on the language it was trained on.

This is **not a model deficiency**. It is a structural property of agglutinative morphology, and it reveals a fundamental flaw in using CSR as a primary metric for agglutinative autocomplete.

**Root cause: morphological length.** Basque words are longer and morphologically complex. The model needs an average of 4.7 typed characters before the correct Basque suggestion appears, versus 3.1 for English and Spanish. A word like *paseatzera* ("to go for a walk") cannot be predicted from *pa* — the model must see *paseatzer* before it confidently offers the full word. In contrast, English *walk* is predictable from *w* once the collocational context (*go for a...*) is established. The keystroke savings are consumed by the longer prefix the user must type before a suggestion becomes available, even though the model ultimately predicts the correct word.

**The Top-K spectrum reveals the mechanism.** Basque achieves the highest Top-3 accuracy (79.5%) — the model identifies the correct word in its top-3 chips more often than in English (72.0%) or Spanish (73.9%). Yet Basque has the lowest Top-1 accuracy (35.9% vs 52.2% for Spanish). The 43.6pp gap between Top-1 and Top-3 for Basque (vs 21.7pp for Spanish) is the signature of agglutinative prediction: multiple valid morphological continuations (*paseatzera*, *paseatzeko*, *paseatzen*) distribute probability mass, so the correct word appears in the top-3 but rarely as the single highest-ranked candidate. Notably, Basque Top-3 equals Top-5 (79.5%) — the inference engineering (sticky merge, retokenization fallback) already surfaces every correct word from the raw top-5 pool into the displayed top-3 chips, leaving no room for improvement from wider candidate fetches.

**Confidence inversely correlates with CSR.** Basque has the lowest average confidence on accepted suggestions (0.228) yet the highest acceptance rate (79.5%). The model is "cautiously correct" on Basque — it identifies the right word but with distributed probability mass across morphological variants. English and Spanish, with shorter words and stronger collocational patterns, produce high-confidence predictions but lower acceptance rates — many short function words (*the, is, a / el, y, un*) are too ambiguous at 1–2 characters to predict, dragging down the acceptance rate without reflecting on model quality.

Only **one high-confidence (≥0.8) acceptance occurred in Basque** (0.922), compared to 11 in English and 11 in Spanish. Yet Basque achieved the highest acceptance rate (79.5%) — the model identified the correct word most often, just with lower per-word confidence. This is the signature of agglutinative prediction: multiple valid morphological continuations distribute probability mass, producing lower per-word confidence without indicating poorer predictions.

**Implications for evaluation methodology.** CSR is not merely a lower bound (§6.8) — it is a **structurally biased metric** that penalizes the very language the system is designed to serve. If CSR were used as a primary optimization target, it would systematically favor shorter-word languages. This aligns with GitHub's Copilot experience, where acceptance-rate optimization "could lead to incorrectly favoring a high volume of simple and short suggestions" (Fu & Mogensen, 2026) — the same structural bias at production scale in a different domain.

**Recommendation:** CSR for agglutinative autocomplete should be accompanied by (1) Top-K accuracy (Top-1, Top-3, Top-5), (2) average prefix length before acceptance, and (3) average confidence. A naive reader comparing our 24.8% simulated CSR to the ~80% achievable in English autocomplete would conclude the model is poor, when in fact it achieves 79.5% Top-3 accuracy on its target language. The gap is structural, not qualitative.

### 6.13 The Inversion: All Autocomplete Metrics Move Opposite to PPL

The CSR paradox (§6.12) shows that CSR penalizes agglutinative languages structurally. A more troubling question is whether autocomplete metrics track model quality *within* a single language across training. To test this, we ran the next-word CSR simulation (the PyTorch port of the keyboard algorithm, §5.5.6) on **seven checkpoints** spanning the full training trajectory, from step 10K (early training) through step 76K (converged). The same 30 Basque CSR test sentences were used at every checkpoint, and the simulation faithfully replicates the deployed keyboard algorithm (retokenization fallback, sticky merge, top-3 display, acceptance semantics).

**Results across the full training trajectory:**

| Step | Held-out PPL | NW-CSR | 95% CI | Top-1 | Top-3 | Top-5 | Avg Prefix | Confidence |
|-----:|:-----------:|:------:|:------:|:-----:|:-----:|:-----:|:----------:|:----------:|
| 10K | — | 0.375 | [0.313, 0.438] | 0.537 | 0.859 | 0.866 | 3.5 | 0.408 |
| 20K | — | **0.402** | [0.344, 0.469] | 0.503 | 0.852 | 0.859 | 3.2 | 0.415 |
| 30K | — | 0.382 | [0.330, 0.443] | 0.537 | 0.839 | 0.852 | 3.4 | 0.422 |
| 32K | 7.56 | 0.396 | [0.352, 0.449] | 0.523 | 0.866 | 0.873 | 3.6 | 0.401 |
| 54K | 7.17 | 0.385 | [0.355, 0.448] | 0.503 | 0.859 | 0.866 | 3.6 | 0.416 |
| 74K | **7.13** | **0.361** | [0.311, 0.424] | 0.503 | 0.832 | 0.852 | 3.4 | **0.433** |
| 76K | 7.13 | 0.373 | [0.320, 0.450] | 0.517 | 0.832 | 0.852 | 3.3 | 0.432 |

**What is clear:** the model improved (PPL 7.56 → 7.13). **What is unexpected:** in PyTorch, *none* of the autocomplete metrics track this improvement. NW-CSR decreases (0.375 → 0.361), Top-1 accuracy decreases (0.537 → 0.503), Top-3 decreases (0.859 → 0.832), and Top-5 decreases (0.866 → 0.852). The inversion is **not specific to CSR** — it affects all Top-K metrics equally. The only metric that improves monotonically is **confidence** (0.408 → 0.433): the model becomes more certain about its predictions, but not more accurate at matching the gold-standard word.

This finding broadens the anomaly considerably. It is not a quirk of CSR's formulation (keystroke accounting) or of acceptance semantics (sticky merge, retokenization fallback). The model's exact-match next-word prediction accuracy on this fixed 30-sentence test set does not improve with training, even though its probability distribution sharpens.

**Hypotheses (preliminary, unresolved).** The Top-K data strongly supports hypothesis (1):
1. **Multiple valid continuations** *(most supported)* — a better language model converges on *a* valid continuation, not *the* gold-standard one. In agglutinative Basque, many morphological variants are syntactically valid after any given prefix (*paseatzera*, *paseatzeko*, *paseatzen*). As training improves, the model better estimates the full distribution — which may shift probability mass *away* from the specific gold answer toward other valid forms. PPL captures this distributional improvement; exact-match Top-K cannot.
2. **Agglutinative probability distribution** — the better model distributes probability across morphological variants more evenly, depressing top-1 on any single one. Consistent with the 43.6pp Top-1-to-Top-3 gap observed in §6.12.
3. **Computation artifacts** — bf16 forward pass may introduce systematic biases in logprob rankings; see GGUF cross-validation below.
4. **Evaluation set sensitivity** — 30 sentences from a single Wikipedia topic; stylistic drift as training progresses.
5. **Small sample size** — all CIs overlap (n=30); the trend is directional, not proven.

**Practical implication:** **No exact-match autocomplete metric should be used as a primary checkpoint selection criterion** (§6.8). PPL is the only metric that reliably tracks model quality. The anomaly shows that Top-K accuracy and CSR can all move opposite to PPL, making them unreliable for model selection.

**GGUF cross-validation: the CSR inversion is backend-dependent.** We ran the same 30 sentences through the deployed GGUF model (Q5_K_M, llama.cpp). The CSR effect **does not replicate in GGUF**: PyTorch CSR decreases monotonically (0.397 → 0.361) while GGUF CSR modestly *increases* from 32K to 54K (0.362 → 0.390), roughly tracking the PPL improvement, then plateaus (0.389 at 74K). Step 32K is *lowest* in GGUF despite being *highest* in PyTorch — supporting hypothesis (3) that bf16 forward pass vs. Q5_K_M dequantization produce different logprob distributions. Full cross-backend comparison table in Appendix D. Note: Top-K data is only available for the PyTorch backend; the GGUF cross-validation measured CSR only.

**The CSR inversion is backend-specific; the broader Top-K inversion is PyTorch-only (untested in GGUF).** The practical conclusion holds regardless: autocomplete metrics are too noisy across backends and sample sizes to serve as reliable checkpoint selection criteria. All PyTorch CIs overlap (n=30). Quantization also reduces confidence across all checkpoints (e.g., step 74K: 0.433 → 0.381), as expected from 5-bit weight encoding.

---

## 7. Future Work

### 7.1 Immediate

1. **Investigate the metric inversion** (§6.13) with a larger, more diverse evaluation set (100+ sentences). All autocomplete metrics (CSR, Top-1, Top-3, Top-5) decrease as PPL improves in PyTorch; only confidence tracks PPL. Training is complete (76K steps, ~10B tokens, 14.9 hours, PPL flat from step 67K).
2. **Investigate repetition loops** — both checkpoints exhibit greedy-decoding repetition on some prompts. Repetition penalty or nucleus sampling may improve practical autocomplete quality more than further training.
3. **Mitigate date/number prediction artifacts** (§6.10). The model over-predicts temporal and numeric expressions because the corpus is dominated by encyclopedic and journalistic prose. Candidate mitigations include: (a) **downweighting lines with high digit density** during training, (b) **post-hoc filtering** of numeric-heavy suggestions in the demo server, or (c) **domain reweighting** to reduce the relative proportion of Wikipedia/news text in favor of more conversational or instructional prose. Each approach has tradeoffs: downweighting may harm legitimate date prediction, post-hoc filtering adds latency, and domain reweighting requires additional clean data sources.
4. **Scale CSR evaluation** — 300 sentences with CIs is a significant improvement over 30, but 1000+ would further tighten confidence intervals.

### 7.2 Near-term (requires morphological analyzer)

1. **Integrate Apertium Basque as the morphological analyzer.** Apertium is the only practical Basque tool that emits explicit morpheme boundaries (`+` symbols) rather than only lemmas or UD features. Throughput is ~9.4×10⁵ words/second, sufficient for a 22 GB corpus pass. The output must be **surface-preserving**: use segmented forms like `etxe#tik` rather than underlying morphology (`etxe+a+tik`) that doesn't match the visible text.

2. **Pre-segment corpus with surface-preserving morpheme boundaries** and retrain a morphology-aware tokenizer. Based on QuechuaTok results, this should increase MorphAcc from ~67% toward ~80%+.

3. **Distill a mobile-variant model** for the next-word prediction paradigm (§5.4.1). The 91M model (55 MB) is sized for desktop (Smart Compose scale); mobile requires Gboard-scale (~5-10M, ~5-10 MB). A distilled Morpheus-Mobile using the 91M model as teacher would bring the inference engineering strategies (§5.5) into a smartphone-sized model. Mamba-2's O(1) memory (no KV cache) is a key advantage for this target.

### 7.3 Medium-term

1. **Train Morpheus-Base (207M)** on the 4K tokenizer and compare against Morpheus-Small. A larger model may produce autocomplete improvements that are statistically detectable at current sample sizes — the 91M model's gains between 32K and 54K were directionally positive but too small for the multi-token metrics to confirm.
2. **MWE token injection**: Extract the 1,000 most frequent Basque multi-word expressions and inject them as single tokens. This directly reduces autoregressive decoding steps by 40-60% for covered phrases.
3. **Extend cross-model comparison** (§6.6): The current BPC comparison includes GPT-2 (124M), Latxa-Qwen3.5-2B (1.88B), and Morpheus (91M). A larger Latxa variant (7B+) and a base (non-instruct) Latxa model would provide a cleaner baseline, as the current Latxa's instruct-tuning produces noisy completions for raw text completion. Additionally, applying the full inference engineering pipeline (§5.5) to the baseline models would isolate the contribution of inference engineering from model quality.
4. **Domain-specific fine-tuning.** The base model is trained on general Basque prose (Wikipedia, news, literature), which produces a general-purpose autocomplete but also corpus-induced artifacts such as the date/number prediction bias documented in §6.10. A lightweight fine-tuning stage on domain-specific corpora — e.g., legal Basque (terminology, statute language), medical, educational, or conversational/informal text — could improve suggestion relevance for specialized use cases while also mitigating the general-corpus artifacts. The current training infrastructure supports pretraining from scratch and checkpoint resume (continuing the same run), but **does not yet support fine-tuning-specific features**: separate fine-tuning datasets, layer freezing, differential learning rates, or parameter-efficient methods such as LoRA. The 91M parameter scale is small enough that full fine-tuning on a single GPU is feasible; the on-device deployment constraint means domain-adapted variants could be distributed as separate GGUF files and hot-swapped at runtime via the demo server's model reload endpoint (§5.4). This is also a candidate mitigation for the §6.10 date/number artifacts: a domain adapter trained on conversational or instructional prose would shift the model's distribution away from the encyclopedic patterns that produce numeric predictions.
5. **Data scaling experiment for Basque.** The analysis in §6.7 suggests that 2–5B tokens of aggressively filtered, high-quality Basque text would likely match or exceed the current 10B token mixed-quality corpus. To empirically determine the optimal data size, train four models with identical architecture on 1B, 2B, 5B, and 10B token subsets of a quality-filtered corpus, and compare held-out PPL at convergence. This would yield the first data scaling law curve for Basque — a contribution to low-resource language modeling methodology. The experiment would also test whether the convergence at ~8.8B tokens (§6.7) is a property of data quality (fixable with better filtering) or of model capacity (requiring a larger model to benefit from more data).

### 7.4 Long-term

1. **Hybrid Mamba-Attention architecture**: If pure Mamba-2 struggles with long-range morphological agreement, add 2-4 attention layers ("Jamba-style" hybrid).
2. **Distillation from Kimu 9B**: If a distilled 200M Transformer from the 9B Basque teacher becomes feasible, compare against the pure Mamba baseline.
3. **User study with CSR measurement**: Deploy to real Basque speakers and measure actual keystroke savings with A/B testing.

---

## 8. Code Availability

All code, model checkpoints, and evaluation artifacts are publicly available at `github.com/itzune/morpheus-mamba`. The repository contains:

- **Training pipeline**: Mamba-2 training with atomic checkpointing, gradient accumulation, and W&B integration
- **Data pipeline**: Corpus cleaning, SentencePiece Unigram tokenizer training, and pretokenization with validation leakage prevention
- **Export pipeline**: PyTorch → HuggingFace → GGUF conversion with BOS-token configuration
- **Evaluation suite**: PPL, CSR with bootstrap confidence intervals, MorphAcc, and case paradigm completion
- **Demo server**: Docker-based FastAPI inference server supporting both multi-token ghost-text continuation (Smart Compose–style) and next-word prediction (smartphone keyboard–style chips), with completion logging and checkpoint replay
- **Tokenizer ablation data**: Full per-word tokenization results across 4K/8K/16K/32K vocabularies

The primary model checkpoint (step 74K, fully trained) and all quantized GGUF variants (Q4_K_M, Q5_K_M) are included. The model is also published on HuggingFace: `itzune/morpheus` (safetensors format) and `itzune/morpheus-gguf` (GGUF format).

---

## 9. Conclusion

Morpheus demonstrates that **State Space Models (Mamba-2) are a viable architecture for on-device predictive autocompletion in agglutinative languages**. The central research question — whether a Smart Compose–equivalent system can run **entirely on a consumer laptop** for text editors — is answered affirmatively: our 91M Mamba-2 model (55 MB, zero network calls) provides multi-token ghost-text autocompletion on-device, comparable in scale to Smart Compose's ~80M but without the data-center dependency (Table 1, §2.1). After full training (76K steps, ~10B tokens, 14.9 hours), the best checkpoint (step 74K) achieves:

- **Held-out PPL of 7.13** (converged, flat from step 67K)
- **25.3% CSR** (95% CI [24.0%, 26.5%]) on 300 held-out sentences
- **76% MorphAcc** — 3.8× improvement over the 32K tokenizer's 20%
- **55 MB on-disk** (Q4_K_M) — deployable on consumer laptops

### Primary Contributions

1. **Vocabulary-size ablation for Basque.** Unigram MorphAcc drops from 66.7% at 4K to 28.6% at 32K, mirroring QuechuaTok. At 4K, verbal agreement morphemes (`zki`, `zu`, `ke`) are independently accessible; at 32K, they are buried in opaque atomic units. **Fertility correlates negatively with morphological accuracy** for agglutinative tokenizer evaluation — lower fertility is achieved by fusing morphemes into opaque units. We note this is based on 4 data points (MorphAcc, not downstream PPL) and establish correlation, not causation.

2. **Evaluation methodology: PPL is the only reliable metric; autocomplete metrics are fragile.** PPL improved monotonically (7.56 → 7.17 → 7.13, all 14 files agree) and is the only metric that unambiguously ranked checkpoints. CSR is fragile to implement (string prompts gave ~4% vs ~28% with token IDs due to BOS/tokenizer bugs) and saturates at small sample sizes (CIs overlap at n=300). The **CSR paradox** (§6.12) shows the model's native Basque achieves the *lowest* simulated CSR (19.6%) — below English/Spanish at <1% of training data — because agglutinative word length requires longer typed prefixes. An **unresolved anomaly** (§6.13): in the PyTorch backend, *all* autocomplete metrics (CSR, Top-1, Top-3, Top-5 accuracy) move opposite to PPL improvement across the training trajectory — the model becomes more confident but not more accurate at exact-match next-word prediction. The CSR-specific inversion is backend-dependent (does not replicate in GGUF) and may be partly a computation artifact rather than a metric property. **CSR penalizes the very language such systems are designed to serve and should not be used as a primary optimization target.** This is corroborated by GitHub's Copilot team, who found acceptance-rate optimization structurally misleading (Fu & Mogensen, 2026).

3. **Inference engineering for agglutinative keyboards.** Five strategies — retokenization fallback, sticky merge, top-k exceeding display-k, next-word candidate extraction, and completion logging with replay — that address failure modes unique to subword-tokenized keyboards. Inference engineering adds 3.9× CSR on top of the raw model, demonstrating it is a major contribution, not a marginal optimization. We also report a critical dependency: Mamba-2 models require `llama.cpp` ≥ commit `dc2187d48` to avoid silently incorrect greedy outputs.

4. **Cross-model baseline comparison using BPC.** BPC is the correct tokenizer-independent metric. Morpheus (91M, BPC 0.970) matches GPT-2 eus-euscrawl (124M, BPC 0.981) — driven by 24× more training data, not architecture — and approaches Latxa-Qwen3.5-2B (1.88B, BPC 0.822) at 1/20th the parameters. Per-token PPL is misleading across vocabularies.

5. **On-device Smart Compose feasibility.** A Smart Compose–equivalent system can run entirely on-device (91M Mamba-2, 55 MB, ~10B tokens) for a morphologically complex language. Both Google and we chose recurrent architectures over Transformers to avoid KV-cache latency (§2.1). The data scale asymmetry (Google's 30–60× larger corpus) reflects the high-resource vs. low-resource divide (§6.7).

6. **Data scaling analysis for low-resource LM.** Our 110:1 tokens-to-parameters ratio is reasonable by modern small-model standards (below Mosaic 190:1, MiniCPM 192:1). However, the model converged at ~8.8B tokens — not because the budget was excessive, but because **data quality is the binding constraint**. ~20–30% corpus noise means effective high-quality data is ~7–8B tokens, matching the convergence point. For low-resource languages, aggressive quality filtering of a 2–5B token corpus would likely outperform maximizing raw token count.

### Limitations

- CSR of 25% is below the ~80% achievable in English autocomplete, reflecting both agglutinative difficulty and the **structural CSR paradox** (§6.12)
- **Date/number prediction artifacts** (§6.10): domain mismatch between training corpus (encyclopedic/journalistic) and deployment context (general text editor)
- **Evaluation metric limitations** (§6.8): PPL is the only reliable checkpoint-ranking metric; CSR and MorphAcc saturate at available sample sizes
- No morphological pre-segmentation (Apertium) yet; MorphAcc could improve from 76% toward 83%+
- No user study; all evaluation is simulation-based or expert-judged
- The 91M model is too large for mobile (Gboard: 1.4M/1.4MB defines the mobile target); a distilled variant (~5–10M) has not yet been trained
- Data quality is the binding constraint (§6.7); a 2–5B token high-quality subset would likely match current quality but has not been empirically tested

### The Path Forward

The model has converged (PPL flat from step 67K). The next steps are: (1) integrate **Apertium Basque** for surface-preserving morpheme pre-segmentation, pushing MorphAcc toward 83%; (2) apply aggressive quality filtering to a 2–5B token subset; (3) distill a **Morpheus-Mobile** (~5–10M) for the Gboard paradigm; (4) investigate the metric inversion with larger evaluation sets.

---

## Appendix A: The Morfessor Attempt and Pivot

Our first approach to improving morphological alignment was **morphological pre-segmentation** — training a Morfessor 2.0 model on the corpus, using it to segment words into morphemes before tokenizer training, then training a Unigram tokenizer on the pre-segmented text. This followed v1's ADR-002 ("Morfessor Pre-Segmentation as Mandatory Pre-Processing Step"), which had been accepted but never executed.

We trained Morfessor 2.0 on 4M words from a corpus sample. The segmentation quality was **poor** due to the mixed-language nature of the corpus (Basque + Spanish + English):

| Input | Morfessor output | Problem |
|-------|-----------------|--------|
| `cookie` | `coo\|kie` | English word segmented incorrectly |
| `dutenez` | `duten\|ez` | Should be `dute\|nez` or unsplit |

Morfessor's MDL objective fails on mixed-language text without language filtering: it cannot distinguish Basque morphology from arbitrary character sequences in foreign words. Language-filtering the corpus before Morfessor was deemed too costly for the expected benefit.

**Pivot:** Rather than fix Morfessor, we noticed that the QuechuaTok paper showed vocabulary size alone drives significant MorphAcc gains (Unigram 4K = 66.67% **without** any pre-segmentation). We pivoted to testing vocabulary size as the primary variable — a simpler experiment with a stronger literature basis. Morfessor pre-segmentation remains unexplored and is deferred to future work with Apertium (§7.2), which provides linguistically grounded morpheme boundaries rather than statistical guesses.

---

## Appendix B: Where 4K Still Fails — Full Failure-Mode Table

The 4K tokenizer handles single-layer case suffixes well but struggles with two categories:

| Word | 4K tokenization | Expected | Problem |
|------|----------------|----------|--------|
| `etxean` | `▁etxean` | `etxe\|an` | Whole word not split (inessive) |
| `etxearentzat` | `▁etxe` `a` `rentzat` | `etxe\|arentzat` | Multi-layer suffix split incorrectly |
| `etxearengatik` | `▁etxe` `aren` `gatik` | `etxe\|arengatik` | Multi-layer suffix split incorrectly |
| `lagunera` | `▁lagun` `era` | `lagun\|ra` | Epenthetic `e-` fused with suffix |
| `lagunetik` | `▁lagun` `etik` | `lagun\|tik` | Epenthetic `e-` fused with suffix |

Two failure modes:
1. **Multi-layer suffixes** (`a+ren+tzat`): when a case suffix itself decomposes into multiple morphemes, 4K sometimes splits at the wrong internal boundary.
2. **Epenthetic vowels**: Basque inserts `e-` before consonant-initial suffixes after certain roots (`lagun` + `tik` → `lagunetik`). The 4K tokenizer fuses this epenthetic vowel with the suffix (`lagun` + `etik`) rather than with the root.

These remaining failures motivate the Apertium pre-segmentation future work (§7.2): a morphological analyzer that provides explicit, linguistically grounded boundaries would resolve both multi-layer suffixes and epenthetic vowel placement — problems that vocabulary size alone cannot fully solve.

---

## Appendix C: Inference Engine Sensitivity — The SSM_SCAN Bug

During development, step 54K appeared to have "forgotten" the word *Kaixo* (predicting it at 0% probability) while step 32K predicted it correctly at 47.5%. Investigation via the replay system revealed that the model was not at fault: a stale Docker cache had built an older version of `llama.cpp` that contained a bug in the SSM (State Space Model) scan computation for Mamba-2 architectures (`n_groups > 1`, fixed in commit `dc2187d48`). The bug produced silently incorrect greedy outputs for certain weight configurations — step 54K's weights triggered it, step 32K's did not.

**Recommendation:** When deploying Mamba-2 models with `llama.cpp`, pin to a build that includes commit `dc2187d48` (2025-07-04 or later). If a checkpoint appears to have regressed, rebuild the inference engine with `--no-cache` before investigating the model. The completion logging and replay system (§5.5.6) is invaluable for detecting such issues, as it enables direct A/B comparison of checkpoints on identical inputs.

---

## Appendix D: Metric Inversion — Full GGUF Cross-Validation Table

| Step | Backend | NW-CSR | Top-1 | Top-3 | Top-5 | Confidence |
|-----:|---------|:------:|:-----:|:-----:|:-----:|:----------:|
| 32K | PyTorch (bf16) | 0.396 | 0.523 | 0.866 | 0.873 | 0.400 |
| 32K | GGUF (Q5_K_M) | 0.362 | — | — | — | 0.372 |
| 54K | PyTorch (bf16) | 0.385 | 0.503 | 0.859 | 0.866 | 0.415 |
| 54K | GGUF (Q5_K_M) | 0.390 | — | — | — | 0.413 |
| 74K | PyTorch (bf16) | 0.361 | 0.503 | 0.832 | 0.852 | 0.433 |
| 74K | GGUF (Q5_K_M) | 0.389 | — | — | — | 0.381 |

Top-K accuracy was only measured in the PyTorch backend. The GGUF cross-validation measured CSR only (via the demo server's keyboard API). In PyTorch, all Top-K metrics decrease from 32K to 74K alongside CSR; in GGUF, CSR modestly increases. Whether GGUF Top-K also tracks PPL remains untested.

The CSR inversion does not replicate in GGUF: PyTorch CSR decreases monotonically (0.396 → 0.361) while GGUF CSR modestly increases from 32K to 54K (0.362 → 0.390), then plateaus (0.389 at 74K). This supports hypothesis (3): bf16 forward pass vs. Q5_K_M dequantization produce different logprob distributions. However, the broader Top-K inversion (all metrics decreasing in PyTorch) is untested in GGUF — the demo server API does not expose per-rank candidate data needed for Top-K computation.

---

## Appendix E: Domain Completion Examples

Greedy completions from the step 74K checkpoint across five text domains. Prompts are prose sentences extracted from the body of each document (skipping navigation and boilerplate). The model receives no system prompt or instruction — it simply continues the text.

**Wikipedia**

Example 1:

- Prompt: *Euskaltzaindia euskara zaindu, aztertu, zabaldu, batu eta hobetzea helburu duen*
- Prediction: *erakundea da.*

Example 2:

- Prompt: *Euskal literatura euskaraz egindakoa da, Euskal Herriko idazleek gaztelaniaz, frantsesez*
- Prediction: *eta ingelesez idatzitakoa.*

Example 3:

- Prompt: *Biodibertsitatea, bioaniztasuna edo dibertsitate biologikoa ekosistema batean, bioma batean edo*
- Prediction: *gehiagotan biltzea da.*

Example 4:

- Prompt: *Nafarroa Garaia Iberiar penintsularen iparraldean dago. Mugakide hauek ditu: Lapurdi,*
- Prediction: *Nafarroa Beherea, Zuberoa, Nafarroa Beherea eta Zuberoa.*

**News**

Example 1:

- Prompt: *Eusko Jaurlaritzako Segurtasun Sailak jakinarazi duenez, gaur goizean gizon bat*
- Prediction: *atxilotu dute Gasteizen, bikotekide ohia jotzea egotzita.*

Example 2:

- Prompt: *AEBek eta Iranek berriro egin diote eraso elkarri, eta Ormuzko*
- Prediction: *itsasartea zeharkatu dute.*

Example 3:

- Prompt: *Volkswagen taldeak auto modelo gutxiago egingo ditu*
- Prediction: *eta horrek eragina izango du automobilgintzaren sektorean.*

Example 4:

- Prompt: *Iranek Zipreko itsasontzi bat gerarazi du eta marinel bat desagertuta*
- Prediction: *dago.*

**Legal**

Example 1:

- Prompt: *Hemen ikusgai dauden gainerako formatuen edukia PDF dokumentu elektroniko ofizial*
- Prediction: *etan kontsultatu daiteke.*

**Education**

Example 1:

- Prompt: *“Harrera eta Erreserbak” proiektuak turismo-ostatuen sektorean harrera-funtzioak eta erreserben kudeaketa*
- Prediction: *integrala egiteko aukera ematen du.*

Example 2:

- Prompt: *Baserriberri. Egurrean eraikitzeko eta eraikinak zaharberritzeko zurezko lan-teknikei buruzko formakuntza*
- Prediction: *jasoko dute ikasleek.*

Example 3:

- Prompt: *Sareak Administratzea eta Planifikatzea moduluko 64 bideotutorial daude. / Bideoen*
- Prediction: *estekak:*

**Literature**

Example 1:

- Prompt: *Baigorri eta Baztan arteko korrespondentzia fazeria erlazioez (1804)*
- Prediction: *eta 1805eko korrespondentziaz (1806) ari gara.*

**Cross-Lingual Completions (English & Spanish)**

The model was trained on Basque-focused corpora, but web-crawled sources contain incidental English and Spanish (<1% each). These completions illustrate the weak cross-lingual transfer documented in §6.11.

**English**

Example 1:

- Prompt: *Thank you very*
- Prediction: *much.*

Example 2:

- Prompt: *In recent years, the development of technology has*
- Prediction: *been found in the Basque Country.*

Example 3:

- Prompt: *The United States of America is a country in*
- Prediction: *the 19th century.*

Example 4:

- Prompt: *I like to listen to music while I am*
- Prediction: *starting to start to start to start to start*

**Spanish**

Example 1:

- Prompt: *El tiempo está bueno hoy y voy a salir a*
- Prediction: *la calle.*

Example 2:

- Prompt: *En los últimos años, el desarrollo de la tecnología ha*
- Prediction: *ido creciendo, ya que la tecnología ha sido uno de los*

Example 3:

- Prompt: *Me gusta escuchar música mientras estoy*
- Prediction: *en directo.*

Example 4:

- Prompt: *Los Estados*
- Prediction: *Unidos, 1990.*

Example 5:

- Prompt: *La República Francesa es un país de*
- Prediction: *gran calidad, ya que la gran mayoría de la población vas*

**Observations.** Wikipedia completions are frequently perfect — the model reproduces encyclopedic prose with high fidelity, which is both a strength (accurate continuations) and a corpus-induced artifact (§6.10). News completions are often grammatically correct and contextually relevant, including multi-token continuations that add new information. Legal text produces structurally plausible completions. Education prompts yield mixed results: real prose sentences receive coherent continuations, but title/header lines produce empty or degenerate outputs (the model predicts EOS). Literature is the weakest domain — the archaic Basque orthography (17th–19th century) is out-of-distribution for the modern Batua-trained model, and completions are either empty or produce plausible-sounding but hallucinated content.

Cross-lingual completions reveal the model's incidental transfer. High-frequency collocations are predicted correctly (*Thank you very* → *much*, *Los Estados* → *Unidos*, *El tiempo está bueno hoy y voy a salir a* → *la calle*). Longer continuations expose the Basque bias: *“the development of technology has”* → *“been found in the Basque Country”* (the model redirects to its dominant domain), and *“La República Francesa es un país de”* → *“gran calidad, ya que la gran mayoría de la población vas...”* (trails off into Basque mid-sentence). Repetition failures (*starting to start to start*) are common when the model lacks confident continuations in the non-dominant language.

---

## Appendix F: Three-Gate Pre-Training Validation Protocol

Perplexity is the standard language modeling metric, but it is insufficient as a sole quality gate for autocomplete: a model can achieve low PPL while producing degenerate suggestions if the training corpus contains systematic artifacts (e.g., bare numbers in sentence position from official gazette texts). To ensure data and pipeline integrity before committing GPU resources, we designed a three-gate pre-training validation protocol. **No training run may proceed to the L40 without passing all three gates.**

**Gate 1: Corpus Content Audit (CPU, ~30 min).** An LLM-based quality audit (40 random lines per source, 1–5 Basque quality scale) that identified the low-quality sub-corpora subsequently omitted from training (§4.2). The retained 11 sources scored an average of 4.6/5.

**Gate 2: Proxy Overfit Test (GPU, ~20s).** A canary test: a 0.7M-parameter Mamba-2 model (128× smaller than target) attempts to memorize 5 hand-crafted Basque sentences not in the training corpus. If it can memorize novel Basque from the tokenized `.npy` format in 300 steps, the pipeline (tokenizer, serialization, architecture, training loop) is proven sound. Any downstream failure must then come from data quality or training scale, not infrastructure.

| Metric | Value |
|---|---|
| Initial loss (step 0) | 8.30 (random initialization) |
| Final loss (step 300) | 0.049 (170× reduction) |
| Canary token accuracy | 57.7% (≥50% threshold) |
| Runtime | 18.5 seconds |
| NaN events | 0 |

**Gate 3: Autocomplete Smoke Test (GPU, ~5 min).** Evaluates whether the full 91M model produces useful autocomplete suggestions on real Basque text. **Passed:** at step 32K, the model achieves CSR=24.9% on 300 held-out sentences and MorphAcc=70%, confirming the model produces coherent Basque completions.

| Gate | Runtime | What it validates | clean-v3 result |
|---|---|---|---|
| 1: LLM audit | ~30 min CPU | Source quality, fragments, boilerplate | 11/14 sub-corpora retained |
| 2: Proxy overfit | ~20s GPU | Tokenizer integrity, data format | ✅ 57.7% canary accuracy |
| 3: Autocomplete smoke | ~5 min GPU | Autocomplete quality on real text | ✅ CSR=24.9%, MorphAcc=70% |

Each gate targets a distinct failure class that PPL alone cannot detect: data contamination and source quality (Gate 1), pipeline and serialization integrity (Gate 2), and inference-time autocomplete behavior (Gate 3). Together they ensure that low PPL reflects genuine language modeling quality rather than artifacts of data or infrastructure.

---

## References

1. Contreras, M. (2026). *QuechuaTok: Morphological Boundary Accuracy as a Necessary Metric for Tokenizer Evaluation in Agglutinative Low-Resource Languages*. arXiv:2606.23943.
2. Arnett, C., Hudspeth, M., & O'Connor, B. (2025). *Evaluating Morphological Alignment of Tokenizers in 70 Languages*. arXiv.
3. Stephen, A., & Libovický, J. (2026). *Evaluating Morphological Plausibility of Subword Tokenization via Statistical Alignment with Morpho-Syntactic Features*. arXiv.
4. Xu, N., & Kim, A. (2026). *Tokenization and Morphological Fidelity in Uralic NLP: A Cross-Lingual Evaluation*. arXiv.
5. Táboas García, T., Przybyła, P., & Wanner, L. (2025). *Exploring morphology-aware tokenization: A case study on Spanish language modeling*. EMNLP 2025.
6. Hu, J. F. (2025). *Tokenization Strategies for Low-Resource Agglutinative Languages in Word2Vec: Case Study on Turkish and Finnish*. ACDSA 2025.
7. Etxaniz, J., Sainz, O., Perez, N., Aldabe, I., Rigau, G., Agirre, E., Ormazabal, A., Artetxe, M., & Soroa, A. (2024). *Latxa: An Open Language Model and Evaluation Suite for Basque*. arXiv:2403.20266.
8. Chen, M. X., et al. (2019). *Gmail Smart Compose: Real-Time Assisted Writing*. arXiv:1906.00080.
9. Hard, A., et al. (2018). *Federated Learning for Mobile Keyboard Prediction*. arXiv:1811.03604.
10. Fu, S., & Mogensen, J. (2026). *The Road to Better Completions: Building a Faster, Smarter GitHub Copilot with a New Custom Model*. GitHub Blog.
11. Cheney, D. (2025). *How GitHub Copilot Serves 400 Million Completion Requests a Day*. QCon San Francisco 2024 / InfoQ.
12. Gu, A., & Dao, T. (2023). *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*. arXiv:2312.00752.
13. Dao, T., & Gu, A. (2024). *Transformers are SSMs: Generalized Models and Efficient Algorithms through Structured State Space Duality*. arXiv:2405.21060.
14. Beck, M., et al. (2024). *xLSTM: Extended Long Short-Term Memory*. NeurIPS 2024.
15. Lane, W., Harrigan, A., & Arppe, A. (2022). *Interactive Word Completion for Plains Cree*. ACL 2022.
16. Trnka, K., & McCoy, K. (2008). *Evaluating Word Prediction: Framing Keystroke Savings*. ACL 2008.
17. Liquid AI (2026). *LFM-2.5-230M: On-Device State Space Models*. liquid.ai.
18. Orai NLP (2026). *Kimu: Basque-Adapted Language Models*. orai.eus.
19. ChaI-TeA (2024). *ChaI-TeA: A Benchmark for Chinese Input Method*. arXiv:2412.18377.
20. Kosyak, A., & Tyers, F. (2022). *Predictive text for agglutinative languages*.
21. WSTypist (2026). *Simulation-based mobile typing evaluation*. arXiv:2602.06489.
22. Rust, P., et al. (2021). *How Good is Your Tokenizer? On the Monolingual Performance of Multilingual Language Models*. ACL 2021.
23. Hoffmann, J., et al. (2022). *Training Compute-Optimal Large Language Models* (Chinchilla). arXiv:2203.15556.
24. Sardana, N., et al. (2024). *Beyond Chinchilla-Optimal: Accounting for Inference in Language Model Scaling Laws*. arXiv:2401.00448.
25. Hu, S., et al. (2024). *MiniCPM: Unveiling the Potential of Small Language Models with Scalable Training Strategies*. arXiv:2404.06395.
26. Bi, X., et al. (2024). *DeepSeek LLM: Scaling Open-Source Language Models with Longtermism*. arXiv:2401.02954.
27. Muennighoff, N., et al. (2023). *Scaling Data-Constrained Language Models*. arXiv:2305.16264.
28. Sainz, O., et al. (2025). *Instructing Large Language Models for Low-Resource Languages: A Systematic Study for Basque*. EMNLP 2025.

---

*This document is a living research record. Current as of July 11, 2026 — training complete (76,294 steps, best checkpoint step 74,000, PPL 7.13).*
