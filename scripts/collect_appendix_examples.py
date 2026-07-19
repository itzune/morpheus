#!/usr/bin/env python3
"""Collect qualitative completion examples for the appendix.

Uses ONLY authentic Basque sentences from demo/real_prompts.json
(Wikipedia + Berria) — no invented text.

Queries all 3 models (Morpheus, Kimu, Latxa) for:
  - AR (autoregressive) top-3 samples per prompt
  - FIM (fill-in-the-middle) for Morpheus only
"""

import json
import time
import httpx

SERVERS = {
    "morpheus": "http://10.2.121.210:9091",
    "kimu":     "http://10.2.121.210:9092",
    "latxa":    "http://10.2.121.210:9090",
}

# ── AR prompts: authentic sentences from real_prompts.json ──
# Each has: prompt (prefix), gold (actual continuation), full (full sentence), source
AR_PROMPTS = [
    {
        "domain": "Education",
        "source": "Wikipedia (Hezkuntza)",
        "prompt": "Estatua da",
        "gold": "hezkuntza sistemaren gestio eta erregulazioaren erantzule.",
        "full": "Estatua da hezkuntza sistemaren gestio eta erregulazioaren erantzule.",
    },
    {
        "domain": "News / politics",
        "source": "Berria",
        "prompt": "Segurtasun Batzordea ere sei hilabete",
        "gold": "barru bilduko da, adostutakoaren jarraipena egiteko.",
        "full": "Segurtasun Batzordea ere sei hilabete barru bilduko da, adostutakoaren jarraipena egiteko.",
    },
    {
        "domain": "Language & culture",
        "source": "Wikipedia (Euskara batua)",
        "prompt": "Batasuna giputz euskaran zergatik oinarritu zen azaltzeko,",
        "gold": "arrazoi demografikoak ematen dira gehienetan.",
        "full": "Batasuna giputz euskaran zergatik oinarritu zen azaltzeko, arrazoi demografikoak ematen dira gehienetan.",
    },
    {
        "domain": "Geography",
        "source": "Wikipedia (Bilbo)",
        "prompt": "Euriak udazkenean eta udaberrian ugariak izaten dira; negua, berriz,",
        "gold": "eztia, eta uda ez oso beroa.",
        "full": "Euriak udazkenean eta udaberrian ugariak izaten dira; negua, berriz, eztia, eta uda ez oso beroa.",
    },
    {
        "domain": "Music / arts",
        "source": "Wikipedia (Musika)",
        "prompt": "Ming dinastiaren garaian",
        "gold": "hasi zen jai eta ospakizunetako musika kodetzen eta kontserbatorioetan irakasten.",
        "full": "Ming dinastiaren garaian hasi zen jai eta ospakizunetako musika kodetzen eta kontserbatorioetan irakasten.",
    },
    {
        "domain": "History",
        "source": "Wikipedia (Euskal Herria)",
        "prompt": "Erdi Aroan gaztelaniaz idatzi zuen Gonzalo de Berceo Errioxako idazlea",
        "gold": "ziur aski euskalduna zen.",
        "full": "Erdi Aroan gaztelaniaz idatzi zuen Gonzalo de Berceo Errioxako idazlea ziur aski euskalduna zen.",
    },
    {
        "domain": "Email / workplace",
        "source": "iberba.eus (Epostak eta gutunak)",
        "prompt": "Iazko azken batzarrean adostu genuen bezala, urteko memoria",
        "gold": "martxoa baino lehen amaitu beharra dago.",
        "full": "Iazko azken batzarrean adostu genuen bezala, urteko memoria martxoa baino lehen amaitu beharra dago.",
    },
]

# ── FIM prompts: authentic sentences split into prefix/middle/suffix ──
# Morpheus only (Latxa/Kimu are base models without FIM training)
FIM_PROMPTS = [
    {
        "domain": "Language & culture",
        "source": "Wikipedia (Euskara batua)",
        "prefix": "Hirietako hainbat euskaldunek,",
        "middle": "euskalkien erreferentzia sendorik gabe,",
        "suffix": "euskara batua ama-hizkuntzatzat ikasi du.",
        "full": "Hirietako hainbat euskaldunek, euskalkien erreferentzia sendorik gabe, euskara batua ama-hizkuntzatzat ikasi du.",
    },
    {
        "domain": "Society / economy",
        "source": "Wikipedia (Donostia)",
        "prefix": "Langabeziari dagokionez,",
        "middle": "Euskal Autonomi Erkidegoko hiriburuetan tasa baxuena dago",
        "suffix": "Donostian.",
        "full": "Langabeziari dagokionez, Euskal Autonomi Erkidegoko hiriburuetan tasa baxuena dago Donostian.",
    },
    {
        "domain": "News / politics",
        "source": "Wikipedia (Nafarroa Garaia)",
        "prefix": "Hirukotea osatzeko lehen porrotaren ondoren,",
        "middle": "PSN-ko hainbat kideren kritika zorrotzak",
        "suffix": "egon ziren.",
        "full": "Hirukotea osatzeko lehen porrotaren ondoren, PSN-ko hainbat kideren kritika zorrotzak egon ziren.",
    },
    {
        "domain": "News / journalism",
        "source": "Berria",
        "prefix": "Plazaren ingurumarietan korrika zebiltzanek",
        "middle": "gomazko pilotak ez, su armen hotsak",
        "suffix": "aditu zituzten.",
        "full": "Plazaren ingurumarietan korrika zebiltzanek gomazko pilotak ez, su armen hotsak aditu zituzten.",
    },
    {
        "domain": "Email / workplace",
        "source": "iberba.eus (Epostak eta gutunak)",
        "prefix": "Iazko azken batzarrean adostu genuen bezala,",
        "middle": "urteko memoria martxoa baino lehen",
        "suffix": "amaitu beharra dago.",
        "full": "Iazko azken batzarrean adostu genuen bezala, urteko memoria martxoa baino lehen amaitu beharra dago.",
    },
]

N_SAMPLES = 3
TEMPERATURE = 0.7
MAX_TOKENS = 20


def query_ar(base_url: str, prefix: str) -> list[dict]:
    """Get N_SAMPLES AR completions via /v1/complete (suffix='')."""
    results = []
    for _ in range(N_SAMPLES):
        try:
            r = httpx.post(
                f"{base_url}/v1/complete",
                json={
                    "prefix": prefix,
                    "suffix": "",
                    "max_tokens": MAX_TOKENS,
                    "temperature": TEMPERATURE,
                    "n": 1,
                },
                timeout=30.0,
            )
            r.raise_for_status()
            data = r.json()
            results.append({
                "text": data.get("text", "").strip(),
                "confidence": round(data.get("confidence", 0), 3),
            })
        except Exception as e:
            results.append({"text": f"[ERROR: {e}]", "confidence": 0.0})
    return results


def query_fim(base_url: str, prefix: str, suffix: str) -> list[dict]:
    """Get N_SAMPLES FIM completions via /v1/complete."""
    results = []
    for _ in range(N_SAMPLES):
        try:
            r = httpx.post(
                f"{base_url}/v1/complete",
                json={
                    "prefix": prefix,
                    "suffix": suffix,
                    "max_tokens": MAX_TOKENS,
                    "temperature": TEMPERATURE,
                    "n": 1,
                },
                timeout=30.0,
            )
            r.raise_for_status()
            data = r.json()
            results.append({
                "text": data.get("text", "").strip(),
                "confidence": round(data.get("confidence", 0), 3),
            })
        except Exception as e:
            results.append({"text": f"[ERROR: {e}]", "confidence": 0.0})
    return results


def save_output(output):
    out_path = "eval/demo-results/appendix_examples.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"  ✓ Saved to {out_path}")


def main():
    output = {"ar": [], "fim": []}

    # ── AR completions ──
    print("=" * 60)
    print("AR COMPLETIONS (top-3 samples per model)")
    print("=" * 60)
    for i, p in enumerate(AR_PROMPTS):
        print(f"\n[{i+1}/{len(AR_PROMPTS)}] {p['domain']} — {p['source']}")
        print(f"  Prompt: {p['prompt']}")
        print(f"  Gold:   {p['gold']}")
        entry = {**p, "models": {}}
        for model_name, url in SERVERS.items():
            print(f"  → {model_name}...", end=" ", flush=True)
            samples = query_ar(url, p["prompt"])
            entry["models"][model_name] = samples
            for s in samples:
                print(f"\n      «{s['text']}» (conf={s['confidence']})", end="")
            print()
            time.sleep(0.3)
        output["ar"].append(entry)
        save_output(output)

    # ── FIM completions (Morpheus only) ──
    print("\n" + "=" * 60)
    print("FIM COMPLETIONS (Morpheus only — base LLMs lack FIM training)")
    print("=" * 60)
    for i, p in enumerate(FIM_PROMPTS):
        print(f"\n[{i+1}/{len(FIM_PROMPTS)}] {p['domain']} — {p['source']}")
        print(f"  Prefix: {p['prefix']}")
        print(f"  Middle: {p['middle']}")
        print(f"  Suffix: {p['suffix']}")
        entry = {**p, "models": {}}
        # Morpheus FIM
        print(f"  → morpheus (FIM)...", end=" ", flush=True)
        samples = query_fim(SERVERS["morpheus"], p["prefix"], p["suffix"])
        entry["models"]["morpheus"] = samples
        for s in samples:
            print(f"\n      «{s['text']}» (conf={s['confidence']})", end="")
        print()
        # Also try Latxa/Kimu FIM for the appendix (to show they fail)
        for model_name in ["kimu", "latxa"]:
            print(f"  → {model_name} (FIM, expected to fail)...", end=" ", flush=True)
            samples = query_fim(SERVERS[model_name], p["prefix"], p["suffix"])
            entry["models"][model_name] = samples
            for s in samples:
                print(f"\n      «{s['text']}» (conf={s['confidence']})", end="")
            print()
            time.sleep(0.3)
        output["fim"].append(entry)
        save_output(output)

    print(f"\n✓ All done. Results in eval/demo-results/appendix_examples.json")


if __name__ == "__main__":
    main()
