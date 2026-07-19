# Morpheus on Android: Native Keyboard IME Research

**Date:** July 11, 2026
**Question:** Can we integrate the Morpheus v2 Mamba-2 model into a real
Android system keyboard (IME), and is it feasible?

## TL;DR

**Yes, it is feasible, and our model is unusually well-suited for it.** There
is a direct precedent — HuoziIME (ACL 2026 Demo Track) — that ships a working
on-device LLM-powered Android keyboard using llama.cpp + JNI. Our model is
**6.5× smaller** in parameters and **9× smaller** in file size than theirs,
and our Mamba-2 architecture eliminates their hardest engineering problem
(KV cache management) entirely. The fastest path is forking HuoziIME/YuyanIme
(~2–4 weeks). The **#1 risk** to verify first is whether Mamba-2's SSM_SCAN
runs correctly on ARM NEON SIMD — a 1-day spike.

---

## 1. Why This Project Exists

### The Goal

Morpheus v2 is a 91M-parameter Mamba-2 language model for Basque autocomplete,
trained on ~10B tokens. It produces next-word prediction chips and multi-token
ghost-text completions. The model is exported to GGUF (55MB Q4_K_M / 66MB
Q5_K_M) and published on HuggingFace at
[`itzune/morpheus-gguf`](https://huggingface.co/itzune/morpheus-gguf).

The existing demos are:
- **Docker demo** (`demo/`) — backend server + web frontend, requires hosting.
- **morpheus-wasm** (in progress) — browser-based PWA via wllama, runs
  client-side but cannot intercept typing in other apps.

Neither is a **system keyboard**. A real Android IME would let users type
Basque in *any* app — WhatsApp, browser, email — with Morpheus predictions
appearing as suggestion chips and ghost text, exactly like Gboard or SwiftKey,
but running fully on-device with no cloud, no network, no privacy concerns.

### Why It Matters for Basque

Basque is a low-resource, highly agglutinative language. Commercial keyboards
(Gboard, SwiftKey) provide minimal Basque prediction support. An on-device
model trained specifically on Basque text — with inference engineering tuned
for agglutinative morphology (retokenization fallback, byte-fallback garbage
detection) — could offer materially better Basque typing than any existing
keyboard. And because it runs on-device, it works offline and respects
language-community privacy norms.

---

## 2. The Precedent: HuoziIME

### What It Is

[**HuoziIME**](https://github.com/Shan-HIT/HuoziIME) is an open-source,
on-device LLM-enhanced Android input method, published as an ACL 2026 Demo
Track paper with a working APK. It does exactly what we want: ghost text +
suggestion chips, fully on-device, no cloud.

### Their Stack

| Component | Details |
|-----------|---------|
| Base IME | [YuyanIme](https://github.com/gurecn/YuyanIme) (3.5k stars, Chinese pinyin keyboard) |
| Inference | llama.cpp, compiled for Android via NDK + CMake |
| Model | Qwen3-0.6B (600M params, Q4_0 quantization = 485MB) |
| JNI bindings | C++ → Kotlin, with RadixTree-based KV cache reuse |
| License | GPL-3.0 (inherited from YuyanIme derivative) |
| Build | Android SDK 36, NDK 25.2.9519653, CMake 3.22.1 |
| Languages | C++ 53.5%, C 14.1%, Python 8.2%, Kotlin 7.4% |

### Their Performance

- **24–25 tok/s** decode on a MediaTek Dimensity 9000 (2022 flagship-class).
- Ghost text rendering "comfortably outpaces human typing speed."
- KV cache capped at 24MB with RadixTree prefix sharing for incremental typing.
- Hierarchical memory system (KV-Splice) for personalization injection.

### Their Engineering Challenges (and why we avoid them)

HuoziIME's hardest problems stem from the transformer KV cache:

1. **KV cache growth** — context length determines memory. They cap at 24MB.
2. **Prefix sharing** — when the user types one more character, the entire
   prompt changes. They use a RadixTree to find the longest reusable prefix
   and avoid recomputing the full prefill.
3. **KV-Splice** — injecting personalization memory without full recomputation.

**Mamba-2 has no KV cache.** The recurrent state is fixed-size regardless of
context length. When the user types one more character, you feed the new token
through the SSM scan and get the updated state — O(1) per token, no prefix
matching, no tree management. HuoziIME's three hardest engineering problems
simply do not exist for our architecture.

---

## 3. Why Morpheus v2 Is Better Suited Than HuoziIME

| Property | HuoziIME (Qwen3) | Morpheus v2 (Mamba-2) | Advantage |
|----------|------------------|----------------------|-----------|
| Parameters | 600M | **91M** | 6.5× smaller |
| Model file (Q4) | 485MB | **55MB** | 9× smaller download |
| Architecture | Transformer | **SSM (Mamba-2)** | No KV cache |
| Memory per token | O(context length) | **O(1)** | Fixed, predictable |
| KV cache management | RadixTree + 24MB cap | **None needed** | Simpler impl |
| Context recompute | Prefix-share to avoid | **Just feed new token** | Trivial |
| Target language | Chinese (isolating) | Basque (agglutinative) | Different, not worse |
| Inference engineering | General LLM chat | **Agglutinative-specific** | Tuned for morphology |

The size advantage is compounding: a 55MB model downloads faster, uses less
storage, loads faster, and leaves more RAM for the OS and other apps. On a
device with 4GB RAM (budget Android), a 485MB model + KV cache is tight; a
55MB model with O(1) state is comfortable.

---

## 4. Technical Architecture Options

### Option A: Fork HuoziIME (Fastest, ~2–4 weeks)

HuoziIME already has the llama.cpp JNI integration, the IME service, the
keyboard UI, and the suggestion chip framework. We'd:

1. Fork [HuoziIME](https://github.com/Shan-HIT/HuoziIME).
2. Replace Qwen3 GGUF with `morpheus-v2-mamba.Q4_K_M.gguf` (55MB).
3. Port our inference engineering from Python to Kotlin:
   - Token-ID prompt construction (no BOS).
   - Retokenization fallback (the `zaud`→`zaude` trap).
   - Sticky merge (`mergeCandidates()`).
   - Top-k token alternatives for digit repair.
   - Byte-fallback garbage detection (`>0xFF` threshold).
   - EOS repair at position 0.
   - Next-word candidate extraction.
4. Replace Chinese-specific UI/logic with Basque keyboard layout (ñ, ç, ü,
   accents) and our chip-ordering UX (center = most probable, Android-style).
5. Inherit GPL-3.0 license.

**Pros:** Fastest path, proven IME foundation, llama.cpp already wired.
**Cons:** GPL-3.0 (copyleft), Chinese-keyboard codebase to strip down, their
KV cache management code is dead weight (but harmless).

### Option B: Fork YuyanIme + Add llama.cpp (More Control, ~3–5 weeks)

YuyanIme (the base, without HuoziIME's LLM additions) is **BSD-3-Clause** —
more permissive. We'd build the llama.cpp JNI integration ourselves using
llama.cpp's official `examples/llama.android/` as reference.

1. Fork [YuyanIme](https://github.com/gurecn/YuyanIme).
2. Integrate llama.cpp Android build (see §5).
3. Build the JNI bridge (Kotlin ↔ C++ ↔ llama.cpp).
4. Implement our inference engineering in Kotlin.
5. BSD-3-Clause license (our code can be any license).

**Pros:** Permissive license, cleaner codebase, full control.
**Cons:** Must build JNI bridge ourselves (HuoziIME already did this).

### Option C: Build from Scratch (Full Control, ~4–8 weeks)

Use llama.cpp's official `examples/llama.android/` as the inference layer and
implement `InputMethodService` from scratch. No inherited keyboard code.

1. Start from `examples/llama.android/` (official llama.cpp Android binding).
2. Implement `InputMethodService` subclass for the IME lifecycle.
3. Build keyboard UI (virtual keys, chip bar, ghost text overlay).
4. Implement all inference engineering.
5. Any license.

**Pros:** No inherited technical debt, minimal APK, full design control.
**Cons:** Most work — must build the entire keyboard UI and IME lifecycle
from scratch.

### Option D: Build on FlorisBoard (Apache-2.0, ~4–6 weeks)

[FlorisBoard](https://github.com/florisboard/florisboard) is a popular
open-source Android keyboard (Apache-2.0) with a clean, modern architecture.
It doesn't have LLM integration, but it has a well-structured keyboard UI and
IME lifecycle that we could extend.

**Pros:** Modern Kotlin codebase, permissive license, active community.
**Cons:** No existing llama.cpp integration — must build from scratch;
FlorisBoard is in beta/rewrite state.

### Recommendation

**Start with Option A (fork HuoziIME)** for the fastest proof of concept.
Once the Mamba-2-on-ARM risk is validated and we have a working Basque
keyboard, evaluate whether to migrate to Option B (YuyanIme + custom JNI) for
license freedom and cleaner code. Option C/D are fallbacks if the forks prove
too entangled with Chinese-keyboard-specific logic.

---

## 5. llama.cpp on Android: Build Paths

llama.cpp has first-class Android support. The official docs
([`docs/android.md`](https://github.com/ggml-org/llama.cpp/blob/master/docs/android.md))
describe three paths:

### Path 1: Official Android Studio Binding (Recommended)

The `examples/llama.android/` directory is a full Android Studio project with:

- **Hardware acceleration** up to SME2 (Arm) and AMX (x86-64), with automatic
  runtime feature detection — runs on both premium and older devices.
- **`InferenceEngine` facade** — load model, prefill, batch decode.
- **`GgufMetadataReader`** — parse GGUF metadata from file or Uri.
- **Kotlin `Flow`** for streaming generated tokens.
- **Arm AI Chat** (Google Play) as a production reference app built on this.

This binding is the cleanest starting point for the inference layer.

### Path 2: Cross-Compile via NDK

```bash
cmake \
  -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-28 \
  -DCMAKE_C_FLAGS="-march=armv8.7a" \
  -DCMAKE_CXX_FLAGS="-march=armv8.7a" \
  -DGGML_OPENMP=OFF \
  -DGGML_LLAMAFILE=OFF \
  -B build-android

cmake --build build-android --config Release -j{n}
cmake --install build-android --prefix {install-dir} --config Release
```

Notes from the docs:
- OpenMP must be OFF (NDK install issues).
- `llamafile` does not support Android.
- `armv8.7a` flag enables latest ARM features; runtime checks handle older
  devices.
- Must set `LD_LIBRARY_PATH` at runtime (Android doesn't find `lib/` alone).

### Path 3: Termux (No Root, For Testing Only)

Termux provides a Linux environment on Android. Install llama.cpp directly:
```bash
apt install git cmake libandroid-spawn
# then standard cmake build
```
Useful for quick ARM correctness testing without building a full APK.

### Critical: llama.cpp Version Requirement

The llama.cpp build **must include commit `dc2187d48`** ("ggml: fix SSM_SCAN
for n_groups > 1", merged ~2025-07-04). Earlier builds produce **silently
incorrect greedy outputs** for Mamba-2 models. This is already documented in
our `demo/Dockerfile` and `docs/demo.md` for the server demo. The same
requirement applies to Android builds. The official `examples/llama.android/`
on `master` includes this fix.

---

## 6. Mamba-2 on ARM: The #1 Risk

### The Risk

We have verified Mamba-2 inference correctness on:
- **x86 CPU** (Docker demo, llama-server) — correct.
- **CUDA GPU** (GPU server, training + eval) — correct.

We have **not** verified it on **ARM**. The SSM_SCAN kernel has ARM NEON SIMD
implementations in ggml's CPU backend. The `dc2187d48` fix addressed
`n_groups > 1` in the general implementation, but ARM-specific code paths may
have separate issues.

### Why This Matters

If SSM_SCAN produces incorrect results on ARM, the model will *appear* to work
(generate text, not crash) but produce **silently wrong** predictions. This is
exactly what happened with the `dc2187d48` bug on x86 — outputs looked
plausible but were subtly incorrect, and we initially misdiagnosed it as a
"model regression" (the step 54K vs 32K false alarm).

### Mitigation: 1-Day ARM Spike

Before committing to any implementation path:

1. **Build llama.cpp for ARM64** using Path 2 (NDK cross-compile) or Path 3
   (Termux on a physical Android device).
2. **Run a set of known Basque prompts** and compare outputs against the x86
   reference (our Docker demo).
3. **Check greedy outputs match exactly** — same prompt → same tokens. If they
   diverge, there's an ARM-specific bug.
4. **Measure decode speed** (tok/s) on the test device.

If ARM outputs match x86, the risk is cleared and we proceed. If they don't,
we file an issue with llama.cpp/ggml and either wait for a fix or investigate
workarounds.

This spike requires no Android development — just cross-compilation and CLI
testing via `adb shell` or Termux.

---

## 7. Inference Engineering: What Must Be Ported

Our server demo (`demo/server.py`) contains significant inference engineering
that goes beyond "call llama.cpp and display output." This logic must be
ported to Kotlin for the Android keyboard.

### 7.1 Token-ID Prompt Construction

**Problem:** llama.cpp's SentencePiece tokenizer diverges slightly from the
reference SP model for our 4K vocab. On the server, we work around this by
constructing prompts as token IDs, not text.

**Port:** Use llama.cpp's `llama_tokenize()` (available in the Android
binding) and apply the same token-ID-level prompt construction. Alternatively,
bundle the reference SentencePiece model and use the C++ SP library via JNI.

### 7.2 Retokenization Fallback

**Problem:** When a word's incomplete prefix tokenizes differently from the
complete word (e.g., `zaud` → `['▁za', 'u', 'd']` vs `zaude` → `['▁zaude']`),
the model cannot bridge the gap and produces garbage.

**Port:** Query alternate shorter prefixes, filter by typed string prefix,
merge/dedup top candidates. The `_keyboard_candidates()` function in
`server.py` is the reference implementation (~150 lines of Python → Kotlin).

### 7.3 From-Scratch Fallback

**Problem:** Some prefixes cannot reach the whole-word tokenization at all
(e.g., `beti b` → `bezala`). The retokenization fallback queries from-scratch
(empty prefix) and filters results by the typed prefix.

**Port:** Parallelized fallback calls (use Kotlin coroutines instead of
`asyncio.gather`).

### 7.4 Digit Repair

**Problem:** The model sometimes predicts tokens whose text representation is
a digit (e.g., token id 123 → "123"), which corrupts Basque text. The repair
loop swaps digit tokens for the best non-digit alternative from `top_logprobs`.

**Port:** Request `logprobs` from llama.cpp, implement the swap logic. The
`_generate_with_repair()` function is the reference.

### 7.5 EOS Repair

**Problem:** When the current word is a complete token (e.g., `zaude` = token
3685), the model predicts EOS (empty text) as the most likely next token — the
sentence is grammatically complete. But the user expects a suggestion.

**Port:** At position 0, treat EOS/empty tokens as repairable — swap for the
best non-empty alternative. Condition:
`needs_repair = _token_has_digit(chosen_text) or (i == 0 and not chosen_text.strip())`.

### 7.6 Byte-Fallback Garbage Detection

**Problem:** Byte-fallback tokens (`<0xB5>`, `<0xD0>`) get mangled into
non-Basque characters (e.g., Cyrillic `е` = U+0435). All legitimate Basque
non-ASCII characters are in Latin-1 Supplement (U+0080–U+00FF). Characters
above U+00FF are always garbage.

**Port:** `fun hasByteFallbackGarbage(text: String): Boolean = text.any { it.code > 0xFF }`.
When garbage is detected, fall back to retokenization candidates.

### 7.7 Sticky Merge

**Problem:** Rapid typing causes suggestion flicker. The sticky merge carries
forward previous candidates matching the current typed prefix, with a +0.1
prob boost, merged with fresh candidates.

**Port:** `mergeCandidates()` function — straightforward in Kotlin. Constants:
`STICKY_BOOST = 0.1`, `FETCH_K = 5`, `POOL_SIZE = 5`, `DISPLAY_K = 3`.

### 7.8 Next-Word Candidate Extraction

**Problem:** When the model's continuation starts with whitespace, the
"completion" is actually the next word. We extract it and mark it as
`is_next_word` for the chip display logic.

**Port:** Check if generated text starts with `▁` (space), extract the
following word.

### Summary: Porting Effort

| Component | Python LOC | Kotlin Effort | Risk |
|-----------|-----------|---------------|------|
| Token-ID prompts | ~30 | Low | SP tokenizer divergence |
| Retokenization fallback | ~150 | Medium | Core logic |
| From-scratch fallback | ~40 | Low | Coroutines for parallelism |
| Digit repair | ~60 | Medium | Needs logprobs API |
| EOS repair | ~20 | Low | Extension of digit repair |
| Byte-fallback garbage | ~30 | Low | Simple threshold check |
| Sticky merge | ~50 | Low | Straightforward |
| Next-word extraction | ~30 | Low | String parsing |
| **Total** | **~410** | **~2–3 days** | Medium overall |

---

## 8. Tokenizer Considerations

### The SentencePiece Divergence Problem

Our 4K Unigram SentencePiece tokenizer, when loaded by llama.cpp, diverges
slightly from the reference SP model. On the server, we work around this by
using token-ID prompts (feeding integer token IDs directly, bypassing text
tokenization). This is documented in the paper (§5.4, Inference Engineering for Ghost-Text Autocomplete) and the companion futo-basque KEYBOARD_ENGINEERING.md.

### Options for Android

1. **Use llama.cpp's built-in tokenization** (`llama_tokenize()`) and apply
   the same token-ID-level workarounds as the server. Simplest, but inherits
   the divergence.

2. **Bundle the reference SentencePiece model** and use the C++ SP library
   via JNI for tokenization, then feed token IDs to llama.cpp. Most faithful
   to training, but adds a dependency.

3. **Port SentencePiece to pure Kotlin/Java** — Google provides a Java
   wrapper for SentencePiece, but it may not handle all Unigram model
   features. Needs testing.

**Recommendation:** Start with Option 1 (llama.cpp's tokenization) and apply
the same workarounds. If quality issues arise, escalate to Option 2.

### No BOS Token

Inference must match training semantics: **no BOS token**. The prompt is raw
text (or raw token IDs), no special prefix. This is already handled correctly
in our server demo and must be preserved in the Android port.

---

## 9. Performance Estimates

### Decode Speed

HuoziIME achieves 24–25 tok/s with a 600M Qwen3 transformer on a MediaTek
Dimensity 9000. Our model is 6.5× smaller (91M vs 600M). Rough estimate:

- **Linear scaling (naive):** 24 × 6.5 ≈ 156 tok/s — unlikely, memory
  bandwidth dominates, not just FLOPs.
- **Realistic estimate:** 60–100 tok/s on a 2022+ flagship, 30–50 tok/s on
  budget devices. Still far above human typing speed (~5 tok/s).

The Mamba-2 SSM scan has different compute characteristics than transformer
attention — it's a sequential scan, not parallelizable across sequence
position, but each step is cheap. For next-word prediction (generate 1–5
tokens), this is ideal: low latency per request, no prefill cost for the
prompt (state is maintained incrementally).

### Memory

| Component | HuoziIME | Morpheus v2 |
|-----------|----------|-------------|
| Model (loaded) | ~500MB | **~60MB** |
| KV cache | Up to 24MB | **0 (fixed SSM state)** |
| Working memory | ~50MB | ~50MB |
| **Total** | ~574MB | **~110MB** |

Our total memory footprint is ~5× smaller. On a 4GB RAM device, we use <3%
of available RAM.

### Latency Budget

Keyboard prediction has a hard latency requirement: suggestions must appear
within ~100ms of the keystroke to feel "instant." For next-word prediction
(generate 1 token + top-k alternatives from logprobs), the latency is:

1. Feed 1 token through SSM scan: ~1ms (91M params, 1 token).
2. Compute logits + top-k: ~1ms.
3. Apply repair/fallback if needed: ~5–10ms (1–2 extra forward passes).
4. UI update: ~5ms.

**Total: ~10–20ms** per prediction — well within the 100ms budget, even with
fallback paths.

---

## 10. Implementation Plan

### Phase 0: ARM Validation Spike (1 day, blocking)

- Cross-compile llama.cpp for ARM64 via NDK.
- Run known Basque prompts, compare outputs to x86 reference.
- Measure decode speed on a test device (via Termux or adb shell).
- **Gate:** ARM outputs match x86. If not, stop and investigate.

### Phase 1: Proof of Concept (1 week)

- Fork HuoziIME (or YuyanIme + llama.cpp Android binding).
- Load `morpheus-v2-mamba.Q4_K_M.gguf`.
- Implement basic greedy completion (no repair, no fallback).
- Display ghost text in a test activity (not yet a system IME).
- **Gate:** Model loads on Android and produces correct Basque text.

### Phase 2: IME Integration (1 week)

- Wire the inference into `InputMethodService`.
- Implement suggestion chips (top-k from logprobs).
- Implement ghost text overlay.
- Basque keyboard layout (ñ, ç, ü, accents, auto-shift).
- **Gate:** Keyboard appears in Android settings, can be selected, produces
  suggestions while typing in any app.

### Phase 3: Inference Engineering Port (1 week)

- Port retokenization fallback, from-scratch fallback.
- Port digit repair, EOS repair, byte-fallback garbage detection.
- Port sticky merge, next-word extraction.
- Test against the known trap cases (`Kaix`→`Kaixo`, `zaud`→`zaude`,
  `beti b`→`bezala`).
- **Gate:** All trap cases from the server regression test suite pass.

### Phase 4: Polish & UX (1 week)

- Chip ordering (center = most probable, Android-style).
- Auto-space on chip accept, punctuation attachment.
- One-shot auto-shift, caret visibility, real keyboard parity.
- Completion logging (for offline eval/replay).
- **Gate:** UX matches the web predictive keyboard demo.

### Phase 5: Testing & Release (1 week)

- Test on multiple devices (flagship + budget).
- APK signing and distribution (GitHub Releases or F-Droid).
- Documentation.
- **Gate:** Working APK, installable on any Android 8.0+ device.

**Total estimate: ~5 weeks** (Phase 0–5), with Phase 0 as a 1-day gate.

---

## 11. Risks and Mitigations

| # | Risk | Likelihood | Impact | Mitigation |
|---|------|-----------|--------|------------|
| 1 | Mamba-2 SSM_SCAN incorrect on ARM NEON | Unknown | **Critical** | Phase 0 spike (1 day). Gate everything on this. |
| 2 | Decode speed too slow on budget devices | Low | Medium | 91M is tiny; HuoziIME runs 600M at 24 tok/s. Measure in Phase 0. |
| 3 | SentencePiece divergence causes quality issues | Medium | Medium | Use token-ID prompts (same as server). Escalate to bundled SP if needed. |
| 4 | llama.cpp Android binding API gaps | Low | Medium | Use Termux CLI as fallback; build custom JNI if needed. |
| 5 | GPL-3.0 license (if forking HuoziIME) | Certain | Low | BSD-3-Clause path via YuyanIme exists (Option B). |
| 6 | Chinese-keyboard codebase complexity | Medium | Low | Strip Chinese-specific logic; keep IME framework. |
| 7 | Android IME lifecycle complexity | Medium | Medium | YuyanIme/HuoziIME already handle this; inherit their lifecycle code. |
| 8 | Memory pressure on low-end devices | Low | Low | 110MB total is well within budget. |
| 9 | Battery drain from inference | Low | Medium | Only infer on keystroke (not continuously). SSM is cheap per token. |
| 10 | App Store / Play Store distribution | Low | Low | Distribute via GitHub Releases + F-Droid; sideloading. |

---

## 12. Alternative: morpheus-wasm PWA (What Works Today)

Before investing in a native Android keyboard, the **morpheus-wasm** PWA
(in progress) provides a zero-effort path to mobile:

- Runs in Android Chrome via wllama (WebAssembly llama.cpp).
- Can be "installed to home screen" as a PWA.
- Uses the same GGUF model, same inference (via wllama's API).
- **Limitation:** Cannot intercept typing in other apps — it's a standalone
  writing tool, not a system keyboard.

The PWA validates the model in a mobile-browser context and can serve as a
fallback if the native IME proves too complex. But for the real keyboard
experience (predictions in any app), the native IME is necessary.

### Relationship Between the Two Projects

| Aspect | morpheus-wasm (PWA) | Native Android IME |
|--------|--------------------|--------------------|
| Effort | ~2 weeks (in progress) | ~5 weeks |
| System keyboard | ❌ | ✅ |
| Works in any app | ❌ | ✅ |
| Offline | ✅ | ✅ |
| Install complexity | Visit URL | Install APK |
| Inference backend | wllama (WASM) | llama.cpp (native ARM) |
| Model file | Same GGUF | Same GGUF |
| Inference engineering | Port to JS | Port to Kotlin |

The inference engineering (§7) must be ported to *both* JavaScript (for
morpheus-wasm) and Kotlin (for native IME). The logic is identical; only the
language differs. Work done on one port informs the other.

---

## 13. References

### Direct Precedent
- **HuoziIME**: [github.com/Shan-HIT/HuoziIME](https://github.com/Shan-HIT/HuoziIME) — On-device LLM-enhanced Android IME, ACL 2026 Demo Track. Fork of YuyanIme + llama.cpp + Qwen3.
- **YuyanIme**: [github.com/gurecn/YuyanIme](https://github.com/gurecn/YuyanIme) — Base IME (BSD-3-Clause), 3.5k stars, Chinese pinyin keyboard.
- **HuoziIME paper**: arXiv, on-device LLM IME with hierarchical memory, stylized post-training, efficiency optimization. Throughput/latency benchmarks included.

### llama.cpp Android
- **Official Android docs**: [llama.cpp/docs/android.md](https://github.com/ggml-org/llama.cpp/blob/master/docs/android.md) — Android Studio binding, NDK cross-compile, Termux.
- **`examples/llama.android/`**: Official Android Studio project with `InferenceEngine` facade, SME2/AMX hardware acceleration, Kotlin Flow streaming.
- **Arm AI Chat**: [Google Play](https://play.google.com/store/apps/details?id=com.arm.aichat) — Production app built on the llama.cpp Android binding.
- **SSM_SCAN fix**: Commit `dc2187d48` ("ggml: fix SSM_SCAN for n_groups > 1") — required for correct Mamba-2 inference. In llama.cpp `master`.

### Kotlin/JNI Bindings
- **kotlinllamacpp**: [github.com/ljcamargo/kotlinllamacpp](https://github.com/ljcamargo/kotlinllamacpp) — Kotlin bindings for llama.cpp on Android, pure Kotlin API (no C++ required).
- **Reddit: JNI + llama.cpp on Android**: [r/androiddev](https://www.reddit.com/r/androiddev/comments/1rgsqbv/jni_llamacpp_on_android_what_i_wish_i_knew_before/) — Practical gotchas from developers.
- **Reddit: Building llama.cpp for Android (Snapdragon)**: [r/LocalLLaMA](https://www.reddit.com/r/LocalLLaMA/comments/1o7p34f/for_those_building_llamacpp_for_android/) — Snapdragon-specific build tips.

### Alternative Keyboards
- **FlorisBoard**: [github.com/florisboard/florisboard](https://github.com/florisboard/florisboard) — Open-source Android keyboard (Apache-2.0), modern Kotlin.
- **fcitx5-android**: [github.com/fcitx5-android/fcitx5-android](https://github.com/fcitx5-android/fcitx5-android) — Another open-source Android IME.

### Our Model
- **HuggingFace (safetensors)**: [huggingface.co/itzune/morpheus](https://huggingface.co/itzune/morpheus)
- **HuggingFace (GGUF)**: [huggingface.co/itzune/morpheus-gguf](https://huggingface.co/itzune/morpheus-gguf)
- **Server demo inference engineering**: `demo/server.py` (reference implementation for all ported logic in §7).
- **Paper**: `morpheus-on-device-basque-autocompletion-full.md`, §5.4 "Inference Engineering for Ghost-Text Autocomplete".
- **Keyboard paradigm**: `futo-transformer-basque/KEYBOARD_ENGINEERING.md` (retokenization fallback, sticky merge, next-word CSR).

---

## 14. Conclusion

A native Android Basque keyboard powered by Morpheus v2 is **feasible and
well-positioned**. The direct precedent (HuoziIME) proves the architecture
works. Our model's small size (91M, 55MB) and Mamba-2 architecture (no KV
cache) give us compounding advantages in memory, speed, and implementation
simplicity.

The critical first step is a **1-day ARM validation spike** to confirm
Mamba-2's SSM_SCAN produces correct output on ARM NEON. Everything else is
engineering: porting ~410 lines of inference engineering to Kotlin, wiring it
into an IME service, and building the Basque keyboard UI.

**Recommended next action:** Run the Phase 0 ARM spike. If it passes, fork
HuoziIME and begin Phase 1.
