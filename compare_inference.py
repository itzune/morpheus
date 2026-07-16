#!/usr/bin/env python3
"""
Fair head-to-head inference: morpheus (Mamba-2 91M GGUF) vs futo-basque (Llama 25M GGUF).

Both loaded via llama-cpp-python. Both tokenized via SentencePiece (avoids llama.cpp
tokenizer divergence — critical for morpheus per its README). Token-ID prompts fed
directly to create_completion (bypasses BOS auto-prepend).

morpheus: NO BOS (trained without it)
futo-basque: BOS=1 prepended (matches FUTO app behavior)
"""
import os, json, time
import sentencepiece as spm
import llama_cpp

# ── Paths (server) ──────────────────────────────────────────────────────────
MORPHEUS_GGUF = "/root/morpheus-mamba/exports/step_0074000.Q5_K_M.gguf"
MORPHEUS_SP   = "/root/morpheus-mamba/tokenizer/basque_unigram_4000.model"
FUTO_GGUF     = "/root/futo-transformer-basque/gguf/eu_futo_v2.gguf"
FUTO_SP       = "/root/futo-transformer-basque/tokenizer/spm_eu.model"

# ── Unified eval set ────────────────────────────────────────────────────────
NEXT_WORD_TESTS = [
    ("Egun on, zer",            ["moduz", "berri", "da", "nola"]),
    ("Ni euskara",              ["ikasten", "hitzen", "dakit", "maite"]),
    ("Bai, gustatu",            ["zait", "zaizu", "da"]),
    ("Ez dut",                  ["ahaztu", "dakit", "maite", "nahi", "ikusi"]),
    ("Zein da zure",            ["izena", "adina", "etxea"]),
    ("Bihar goizean",           ["etorriko", "joango", "izango", "izanen"]),
    ("Eskerrik asko",           ["guztiaz", "laguntzagatik", "denagatik", "gu"]),
    ("Non dago",                ["etxea", "trena", "garagardoa", "jana"]),
    ("Zer",                     ["da", "esan", "egin", "norena"]),
    ("Nola",                    ["zaude", "dago", "da", "esaten"]),
    ("Gaur ezin",               ["dut", "naiz", "da", "dugu"]),
    ("Barkatu, ez",             ["dakit", "nahi", "dut", "da"]),
    ("Euskal Herriko",          ["Unibertsitatea", "hizkuntza", "kultura"]),
    ("Gaur eguraldi",           ["ona", "txarra", "politikoa"]),
    ("Atzo etxera",             ["joan", "etorri", "heldu"]),
    ("Lagun batek",             ["esan", "egin", "idatzi"]),
]

# ── Model wrapper ───────────────────────────────────────────────────────────

class Model:
    def __init__(self, gguf_path, sp_path, name, add_bos, bos_id=1):
        self.name = name
        self.add_bos = add_bos
        self.bos_id = bos_id
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(sp_path)
        self.llm = llama_cpp.Llama(
            model_path=gguf_path, n_ctx=512, n_threads=4,
            verbose=False, logits_all=True,
        )
        print(f"  {name}: vocab={self.sp.GetPieceSize()}, add_bos={add_bos}")

    def _encode(self, text):
        ids = self.sp.Encode(text)
        if self.add_bos:
            ids = [self.bos_id] + ids
        return ids

    def predict(self, prompt):
        """Greedy next-word + top-5 first-token logprobs."""
        ids = self._encode(prompt + " ")
        t0 = time.time()
        r = self.llm.create_completion(
            ids, max_tokens=15, temperature=0, top_p=1.0, logprobs=5,
        )
        elapsed = time.time() - t0
        choice = r["choices"][0]
        text = choice["text"]
        word = extract_word(text)

        lp = choice.get("logprobs", {})
        top_lp = lp.get("top_logprobs", [None])
        top5 = list(top_lp[0].keys()) if top_lp and top_lp[0] else []
        return word, top5, text, elapsed * 1000

    def close(self):
        del self.llm


# ── Scoring ─────────────────────────────────────────────────────────────────

def extract_word(text):
    text = text.strip()
    if not text:
        return ""
    parts = text.split()
    return parts[0].strip(".,!?;:\"'()[]{}") if parts else ""

def is_structural(text):
    return "<XBU>" in text or "<XBC>" in text or "<XEC>" in text or "<CHAR_" in text

def check_hit(word, plausible):
    if not word or is_structural(word):
        return False
    wl = word.lower().strip("▁<>")
    if not wl:
        return False
    for p in plausible:
        pl = p.lower()
        if wl == pl:
            return True
        if len(wl) >= 2 and (pl.startswith(wl) or wl.startswith(pl)):
            return True
    return False

def check_topk_hit(top5_tokens, plausible):
    if not top5_tokens:
        return False
    for tok in top5_tokens:
        w = tok.strip().replace("▁", "").strip(".,!?;:\"'()[]{}")
        if not w or w.startswith("<") or len(w) < 2:
            continue
        wl = w.lower()
        for p in plausible:
            pl = p.lower()
            if wl == pl or pl.startswith(wl) or wl.startswith(pl):
                return True
    return False


# ── Eval ────────────────────────────────────────────────────────────────────

def run_eval(model):
    print(f"\n{'─'*60}")
    print(f"NEXT-WORD PREDICTION: {model.name}")
    print(f"{'─'*60}")
    top1 = 0; top5 = 0; results = []
    n = len(NEXT_WORD_TESTS)
    for prompt, plausible in NEXT_WORD_TESTS:
        word, top5_toks, raw, ms = model.predict(prompt)
        h1 = check_hit(word, plausible)
        h5 = check_topk_hit(top5_toks, plausible)
        top1 += int(h1); top5 += int(h5)
        flag = "⚡" if is_structural(raw) else ""
        s = "✓" if h1 else "✗"
        results.append({"prompt": prompt, "plausible": plausible,
            "predicted": word, "raw": raw[:80], "top5": top5_toks,
            "top1_hit": h1, "top5_hit": h5, "ms": round(ms)})
        print(f"  {s} [{ms:.0f}ms] '{prompt}' → '{word}' {flag}  (gold: {plausible[:3]})")
        if not h1 and top5_toks:
            print(f"       top5: {top5_toks}")
    print(f"\n  RESULTS ({model.name}):")
    print(f"    Top-1:  {top1}/{n} = {top1/n*100:.1f}%")
    print(f"    Top-5:  {top5}/{n} = {top5/n*100:.1f}%")
    return {"top1": top1, "top5": top5, "n": n, "results": results}

def measure_latency(model, n_runs=5):
    ids = model._encode("Euskal Herriko Unibertsitatea hezkuntza publikoko ")
    model.predict("test")  # warmup
    times = []
    for _ in range(n_runs):
        t0 = time.time()
        model.llm.create_completion(ids, max_tokens=1, temperature=0)
        times.append(time.time() - t0)
    avg = sum(times) / len(times) * 1000
    print(f"  {model.name}: {avg:.1f}ms avg")
    return avg

def main():
    print("=" * 60)
    print("REAL INFERENCE: morpheus vs futo-basque")
    print("=" * 60)

    morpheus = Model(MORPHEUS_GGUF, MORPHEUS_SP, "morpheus", add_bos=False)
    futo = Model(FUTO_GGUF, FUTO_SP, "futo-basque", add_bos=True, bos_id=1)

    print("\n--- Latency ---")
    m_lat = measure_latency(morpheus)
    f_lat = measure_latency(futo)

    m_nw = run_eval(morpheus)
    f_nw = run_eval(futo)

    print(f"\n{'='*60}")
    print("FINAL COMPARISON (REAL INFERENCE)")
    print(f"{'='*60}")
    n = m_nw['n']
    print(f"{'Metric':<25} {'morpheus':>15} {'futo-basque':>15}")
    print(f"{'─'*55}")
    print(f"{'Next-word top-1':<25} {m_nw['top1']}/{n}={m_nw['top1']/n*100:.1f}%   {f_nw['top1']}/{n}={f_nw['top1']/n*100:.1f}%")
    print(f"{'Next-word top-5':<25} {m_nw['top5']}/{n}={m_nw['top5']/n*100:.1f}%   {f_nw['top5']}/{n}={f_nw['top5']/n*100:.1f}%")
    print(f"{'Latency (ms)':<25} {m_lat:>13.1f}   {f_lat:>13.1f}")

    print(f"\n--- Per-prompt ---")
    print(f"{'Prompt':<22} {'morpheus':>14} {'futo-basque':>14}")
    print(f"{'─'*55}")
    for i in range(n):
        p = NEXT_WORD_TESTS[i][0]
        mw = m_nw['results'][i]['predicted'][:12]
        fw = f_nw['results'][i]['predicted'][:12]
        mh = "✓" if m_nw['results'][i]['top1_hit'] else "✗"
        fh = "✓" if f_nw['results'][i]['top1_hit'] else "✗"
        print(f"{p:<22} {mh} {mw:>12}   {fh} {fw:>12}")

    results = {"morpheus": {"next_word": m_nw, "latency_ms": m_lat},
               "futo_basque": {"next_word": f_nw, "latency_ms": f_lat}}
    with open("/root/comparison_results.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to /root/comparison_results.json")
    morpheus.close(); futo.close()

if __name__ == "__main__":
    main()
