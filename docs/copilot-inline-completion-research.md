# GitHub Copilot Inline Suggestions (Ghost Text): Architecture Research

**Date:** July 11, 2026
**Purpose:** Understand how VSCode Copilot's inline text completion works — model type, deployment architecture, and how it compares to Morpheus's on-device approach.

---

## TL;DR

GitHub Copilot's inline suggestions (ghost text) are a **server-side, cloud-hosted LLM inference** system — the exact same paradigm as Gmail Smart Compose, but for code instead of natural language. It is **not** on-device inference. The model is a **custom decoder-only Transformer** (GPT family), specifically fine-tuned for **Fill-in-the-Middle (FIM)** code completion. The original 2021 model was OpenAI Codex (12B parameters). The current model is a custom GitHub/Microsoft model trained through mid-training → SFT → RL. The system serves 400M+ requests/day with <200ms mean latency using an elaborate global proxy infrastructure (HTTP/2 multiplexing, streaming, request cancellation).

**This is the strongest possible contrast to Morpheus**: Copilot is the production-scale, server-side, cloud-dependent version of exactly what Morpheus does on-device. Copilot requires a global fleet of proxy servers, Azure-hosted GPU clusters, and sophisticated network engineering to achieve <200ms latency. Morpheus achieves on-device inference with zero network calls, eliminating all of that infrastructure.

---

## 1. Model Type and Lineage

### 1.1 Original Model (2021): OpenAI Codex

- **Architecture:** Decoder-only Transformer (GPT-3 family)
- **Parameters:** ~12 billion (Codex-12B)
- **Training:** Pretrained on a large corpus of public source code from GitHub, then fine-tuned for code generation
- **Deployment:** Entirely server-side. The VSCode extension sends requests to OpenAI's API (later via GitHub's proxy)
- **Key paper:** Chen et al. (2021), "Evaluating Large Language Models Trained on Code" (arXiv:2107.03374)

### 1.2 Current Model (2024-2026): Custom GitHub/Microsoft Model

GitHub has since moved to **custom models** trained specifically for the completions experience. From the GitHub Blog post "The road to better completions" (Shengyu Fu & John Mogensen, GitHub/Microsoft CoreAI):

**Training pipeline (4 stages):**

1. **Base pretraining** — a general-purpose base model is pretrained on a very large, diverse corpus (standard LLM pretraining)
2. **Mid-training** — the base model is further trained on a curated, de-duplicated corpus of modern, idiomatic public and internal code from **~10 million repositories** across **600+ programming languages**. This ensures the model knows modern APIs, new language syntax, and recent library versions. ("Mid-training" = the stage after base pretraining but before final fine-tuning)
3. **Supervised fine-tuning (SFT)** — specifically trained for **Fill-in-the-Middle (FIM)** code completion. The blog explicitly contrasts this with chat models: "Newer general-purpose chat models perform well in natural language to generate code, but underperform on fill-in-the-middle (FIM) code completion. In practice, chat models experience cursor-misaligned inserts, duplication of code before the cursor (prefix), and overwrites of code after the cursor (suffix)." SFT improves prefix/suffix awareness and formatting fidelity
4. **Reinforcement learning (RL)** — a custom RL algorithm teaches the model what makes suggestions useful along three axes: Quality (syntax-valid, compilable, style-consistent), Relevance (on-task, context-aware), Helpfulness (reduces manual effort, prefers modern APIs). They noted that early RL "over-optimized for longer completions, adding too many comments in the form of 'reward hacking'" — mitigated with comment guardrails

**Key architectural detail:** The model is **NOT a base LLM** in the sense of being a raw pretrained model used directly. It is heavily fine-tuned specifically for the FIM completion task. However, it is also **NOT an instruct/chat model** — the blog explicitly says chat models underperform on FIM. It is a **specialized completion model**: a base model that has been mid-trained on code and then fine-tuned specifically for fill-in-the-middle completion behavior.

### 1.3 Fill-in-the-Middle (FIM)

The key technique that makes inline ghost text work for code (as opposed to end-of-text completion). FIM gives the model both:

- **Prefix** — the code before the cursor
- **Suffix** — the code after the cursor

The model is prompted with special tokens like `<|fim_prefix|>...<|fim_suffix|>...<|fim_middle|>` and generates the code that belongs in the middle. This is critical because in code editing, the cursor is often in the *middle* of a file, and the completion must respect the code that comes after the cursor (e.g., closing braces, function signatures, etc.).

This is more sophisticated than Morpheus's pure prefix-only continuation. Morpheus does end-of-text continuation (prefix only), which is appropriate for prose/text editors where there is rarely structured "suffix" to respect.

### 1.4 Results

The new custom model delivers:
- **20% more accepted and retained characters** (not just accepted, but kept in final code)
- **12% higher acceptance rate**
- **3× higher token-per-second throughput**
- **35% reduction in latency**

GitHub pivoted from optimizing for raw acceptance rate to optimizing for "accepted and retained characters" — because pure acceptance rate optimization "could lead to incorrectly favoring a high volume of simple and short suggestions."

---

## 2. Deployment Architecture: Entirely Server-Side

### 2.1 No On-Device Inference

**GitHub Copilot does NOT run any model locally on the user's machine.** From the InfoQ presentation by David Cheney (tech lead, copilot-proxy):

> "I'm currently the tech lead on copilot-proxy, which is the service that connects your IDEs to LLMs hosted in Azure."

The architecture is:
```
VSCode Extension → GitHub OAuth → copilot-proxy → Azure-hosted LLM
```

The IDE extension:
1. Authenticates to GitHub via OAuth
2. Exchanges the OAuth credential for a **short-lived code completion token** (lifetime ~10-30 minutes)
3. Sends completion requests to the **copilot-proxy** (a Go service) with this token
4. The proxy validates the token, swaps it for the actual API service key, forwards the request to the **LLM hosted in Azure**
5. Results are **streamed** back to the IDE

There is **no local model inference**. Every ghost text suggestion is a round-trip to Azure.

### 2.2 Scale and Latency

- **400M+ completion requests/day** (and growing)
- **8,000 requests/second** at peak (European afternoon → US workday overlap)
- **<200ms mean response time** end-to-end
- **~45% of requests are "typed through"** (cancelled because the user kept typing)

### 2.3 The Infrastructure Required to Achieve <200ms

This is where the contrast with Morpheus becomes starkest. To achieve sub-200ms latency for a cloud-hosted LLM, GitHub built an enormous amount of infrastructure:

**HTTP/2 multiplexing:**
- A single TCP+TLS connection carries multiple request streams (like SSH tunneling)
- Long-lived connections kept open for the process lifetime (minutes to days)
- Critical for avoiding 5-6 round trips of TCP+TLS handshake setup per request
- Required because "cancellation occurs every other request on average" — without HTTP/2, constant connection teardown would make latency explode

**Request cancellation (novel HTTP usage):**
- When the user keeps typing, the previous in-flight request must be cancelled
- They use HTTP/2 stream resets (not TCP connection resets) to cancel individual requests without closing the connection
- The cancellation propagates from IDE → proxy → Azure LLM via Go's `context` object
- This is essential because "the cost of the request, that initial inference before you generate the first token, is the majority of the cost" — so cancelled requests that run to completion waste enormous compute

**Global deployment:**
- Proxy instances colocated with LLM instances in Azure regions worldwide (Europe, Asia, Americas)
- Users routed to their closest region via **octoDNS** (GitHub's DNS configuration tool)
- Health-checked: unhealthy proxy instances "vote themselves out of DNS" and traffic reroutes
- **Rejected the CDN "point of presence" (PoP) model** because "traffic tromboning" (user → Singapore PoP → West Coast model → back) was worse than direct-to-region routing

**Streaming:**
- Results are streamed as soon as generation starts — "It doesn't really matter how long the request is, we immediately start streaming it as soon as it starts"

**GLB (GitHub Load Balancer):**
- Based on HAProxy, one of the few load balancers with good HTTP/2 end-to-end support
- Holds client connections open even during proxy redeployments

**Connection pooling:**
- The proxy fans in thousands of client connections onto a small pool of connections to the LLM

### 2.4 The Proxy as a Middleware Swiss Army Knife

The copilot-proxy also provides:
- **Authentication** (token validation without calling an external auth service per request)
- **Request mutation** — e.g., when a model started over-emitting a specific token, they added negative affinity weighting on the fly without a client rollout
- **Traffic splitting** across multiple model instances/capacity units
- **Traffic mirroring** for shadow testing new models
- **A/B testing** without client involvement
- **Client version management** — fix-ups for old client versions that can't be updated fast enough
- **Observability** — their own latency histograms that the upstream provider doesn't give them

---

## 3. Comparison to Morpheus

| Aspect | GitHub Copilot (Inline Suggestions) | Morpheus v2 |
|--------|-------------------------------------|-------------|
| **Model architecture** | Decoder-only Transformer (GPT family) | Mamba-2 (Selective State Space Model) |
| **Model size** | Not disclosed (12B originally; current custom model likely billions) | 91M parameters |
| **Deployment** | **Server-side** (Azure cloud) | **On-device** (consumer laptop CPU) |
| **Network calls** | Every suggestion is a round-trip to Azure | Zero network calls |
| **Latency target** | <200ms mean (with massive infra) | ≤50ms P90 (local CPU) |
| **Scale** | 400M+ requests/day, 8000 req/s peak | Single user |
| **Infrastructure** | Global proxy fleet, HTTP/2 multiplexing, streaming, DNS routing, connection pools, health checks | `llama-server` process on localhost |
| **Completion type** | Fill-in-the-Middle (FIM) — prefix + suffix aware | Prefix-only continuation |
| **Model specialization** | Fine-tuned for FIM (SFT + RL), not a base model, not a chat model | Base LM (no fine-tuning for completion) |
| **Domain** | Code (600+ programming languages) | Basque natural language text |
| **Training data** | ~10M repositories of code | 4.62B tokens of Basque text |
| **Cancellation handling** | HTTP/2 stream reset, context propagation | N/A (local, instant) |
| **Typed-through rate** | ~45% of requests cancelled | N/A (no requests to cancel) |
| **Cost model** | Cloud GPU compute at scale | Free (local CPU) |
| **Privacy** | Code sent to cloud | Everything stays local |

### 3.1 The Fundamental Architectural Divide

Copilot and Morpheus solve the **same problem** (multi-token ghost-text continuation) but from opposite ends:

**Copilot** uses a large, powerful model in the cloud and invests enormous engineering effort to make the network round-trip fast enough (<200ms). The model is a multi-billion-parameter Transformer that would never fit on a consumer laptop. The infrastructure (HTTP/2, streaming, global proxy fleet, request cancellation) exists *because* the model is remote — all of that engineering is fighting network latency.

**Morpheus** uses a small model (91M) that runs locally on the CPU. There is no network latency to fight. The <50ms P90 is achieved by the model being small enough and the architecture (Mamba-2, no KV cache) being efficient enough for local inference. No proxy, no HTTP/2, no streaming, no global deployment — just a local process.

**Key insight:** Copilot's entire infrastructure exists to compensate for the fact that their model is too large to run on-device. If the model were small enough to run locally (like Morpheus), none of that infrastructure would be needed. Morpheus's on-device approach eliminates the entire class of latency problems that Copilot's engineering team spent years solving.

### 3.2 Copilot Validates the Smart Compose Paradigm

Copilot is essentially **Smart Compose for code** — multi-token ghost text continuation accepted with Tab. The UX is identical to Smart Compose and to Morpheus's greedy demo. The difference is:
- Smart Compose: ~80M LSTM, server-side, natural language
- Copilot: multi-billion Transformer, server-side, code (with FIM)
- Morpheus: 91M Mamba-2, on-device, natural language

Copilot's use of FIM (prefix + suffix) is more sophisticated than Morpheus's prefix-only approach, but FIM is a training technique, not a deployment constraint — Morpheus could adopt FIM in future training without changing the on-device deployment model.

---

## 4. Evaluation Methodology (Relevant to Morpheus)

Copilot's evaluation approach is instructive and parallels several of Morpheus's decisions:

### 4.1 Three-Layer Evaluation

1. **Offline evaluations:**
   - **Execution-based benchmark:** Tests against repositories with unit test coverage. Simulates real tasks, accepts suggestions, measures build-and-test pass rates. "Emphasizes functional correctness over surface fluency."
   - **LLM-judge scoring:** An independent LLM scores completions on Quality (syntax, style), Relevance (on-task, no hallucination), Helpfulness (reduces manual effort). This is the **LLM-as-a-Judge** approach Morpheus researched (see `docs/llm-judge-eval-research.md`)

2. **Pre-production:** Qualitative dogfooding with internal developers, side-by-side testing, structured feedback on "readability, trust, and taste." Language-specific expert evaluation.

3. **Production:** A/B testing with metrics: **accepted-and-retained characters**, acceptance rate, completion-shown rate, time-to-first-token, latency. "Ship only when statistically significant improvements hold up."

### 4.2 The "Accepted and Retained Characters" Metric

This is Copilot's primary metric — not just acceptance rate, but whether the accepted text *stays* in the final code. This is more sophisticated than CSR (which measures keystrokes saved at acceptance time) because it captures the **retention** dimension: a suggestion that is accepted but later deleted provided no real value.

This parallels Morpheus's completion logging with replay (§5.5.6), which logs acceptance events for offline analysis. Copilot's version is more mature: they track retention over the full editing session, not just the acceptance moment.

### 4.3 The Acceptance Rate Trap

GitHub explicitly learned that optimizing for acceptance rate alone was harmful:

> "The original Copilot was optimized for the highest acceptance rate possible. However, we realized that a heavy focus on acceptance rates could lead to incorrectly favoring a high volume of simple and short suggestions."

This validates Morpheus's finding that CSR is a fragile metric (§6.8, §6.14). Both systems found that the obvious "did the user accept it?" metric is misleading — it rewards short, safe suggestions over genuinely useful ones.

### 4.4 Language-Specific Expert Evaluation

> "We collect structured feedback on readability, trust, and 'taste.' Part of this process includes working with language experts to improve overall completion quality. This is unique: while execution-based testing, LLM-based evaluations, dogfood testing, and A/B testing are common, we find language-specific evaluations lead to better outcomes along quality and style preferences."

This directly validates Morpheus's decision to use **expert-authored Basque evaluation prompts** and the finding that assistant-authored ad-hoc prompts scored 13.3% vs expert's 60.0% (a 4.5× gap). GitHub independently arrived at the same conclusion: language experts are essential for quality evaluation.

---

## 5. Sources

1. **GitHub Blog** — "The road to better completions: Building a faster, smarter GitHub Copilot with a new custom model" (Shengyu Fu & John Mogensen). https://github.blog/ai-and-ml/github-copilot/the-road-to-better-completions-building-a-faster-smarter-github-copilot-with-a-new-custom-model/
   - Training pipeline (mid-training, SFT, RL, FIM)
   - Evaluation methodology (offline, pre-production, production)
   - Results (20% more retained chars, 12% higher acceptance, 3× throughput, 35% lower latency)
   - The acceptance rate trap

2. **InfoQ Presentation** — "How GitHub Copilot Serves 400 Million Completion Requests a Day" (David Cheney, tech lead, copilot-proxy, QCon San Francisco 2024). https://www.infoq.com/presentations/github-copilot/
   - Architecture: server-side, Azure-hosted LLMs, copilot-proxy
   - Scale: 400M+ req/day, 8000 req/s peak, <200ms mean latency
   - Infrastructure: HTTP/2 multiplexing, request cancellation, global deployment, GLB, octoDNS
   - ~45% typed-through (cancelled) rate
   - Proxy as middleware (auth, mutation, traffic splitting, A/B testing)

3. **Chen et al. (2021)** — "Evaluating Large Language Models Trained on Code" (arXiv:2107.03374). Original Codex model: 12B parameter decoder-only Transformer, pretrained on GitHub code.

4. **VSCode Docs** — "Inline suggestions from GitHub Copilot in VS Code." https://code.visualstudio.com/docs/editing/ai-powered-suggestions

5. **GitHub Docs** — "Changing the AI model for GitHub Copilot inline suggestions." https://docs.github.com/en/copilot/how-tos/use-ai-models/change-the-completion-model

6. **vnavarro.dev** — "Fill-in-the-Middle: The Magic Behind Smart Code Completion." https://vnavarro.dev/blog/fim. FIM token format explanation.
