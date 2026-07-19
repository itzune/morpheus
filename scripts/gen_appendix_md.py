#!/usr/bin/env python3
"""Generate appendix markdown from appendix_examples.json."""

import json

data = json.load(open("eval/demo-results/appendix_examples.json"))

# Domain translations (brief, for non-Basque readers)
AR_TRANSLATIONS = {
    "Estatua da": "(The State is)",
    "Segurtasun Batzordea ere sei hilabete": "(The Security Committee, six months)",
    "Batasuna giputz euskaran zergatik oinarritu zen azaltzeko,": "(To explain why the unification was based on Gipuzkoan Basque,)",
    "Euriak udazkenean eta udaberrian ugariak izaten dira; negua, berriz,": "(Rains are abundant in autumn and spring; winter, however,)",
    "Ming dinastiaren garaian": "(During the Ming dynasty)",
    "Erdi Aroan gaztelaniaz idatzi zuen Gonzalo de Berceo Errioxako idazlea": "(Gonzalo de Berceo, the Riojan writer who wrote in Castilian in the Middle Ages,)",
    "Iazko azken batzarrean adostu genuen bezala, urteko memoria": "(As agreed in last year's meeting, the annual report)",
}

FIM_TRANSLATIONS = {
    "Hirietako hainbat euskaldunek,": "Many Basque speakers in the cities,",
    "euskalkien erreferentzia sendorik gabe,": "without a strong reference to dialects,",
    "euskara batua ama-hizkuntzatzat ikasi du.": "learned standard Basque as a mother tongue.",
    "Langabeziari dagokionez,": "Regarding unemployment,",
    "Euskal Autonomi Erkidegoko hiriburuetan tasa baxuena dago": "the lowest rate among the capitals of the Basque Autonomous Community is",
    "Donostian.": "in Donostia.",
    "Hirukotea osatzeko lehen porrotaren ondoren,": "After the first failure to form the tripartite,",
    "PSN-ko hainbat kideren kritika zorrotzak": "sharp criticism from several PSN members",
    "egon ziren.": "were heard.",
    "Plazaren ingurumarietan korrika zebiltzanek": "Those running around the square",
    "gomazko pilotak ez, su armen hotsak": "not rubber balls, but the sound of firearms",
    "aditu zituzten.": "heard.",
    "Iazko azken batzarrean adostu genuen bezala,": "As agreed in last year's meeting,",
    "urteko memoria martxoa baino lehen": "the annual report before March",
    "amaitu beharra dago.": "must be finished.",
}

GOLD_TRANSLATIONS = {
    "hezkuntza sistemaren gestio eta erregulazioaren erantzule.": "(responsible for the management and regulation of the education system.)",
    "barru bilduko da, adostutakoaren jarraipena egiteko.": "(will meet, to follow up on what was agreed.)",
    "arrazoi demografikoak ematen dira gehienetan.": "(demographic reasons are most often given.)",
    "eztia, eta uda ez oso beroa.": "(mild, and the summer not very hot.)",
    "hasi zen jai eta ospakizunetako musika kodetzen eta kontserbatorioetan irakasten.": "(began codifying festive and ceremonial music and teaching it in conservatories.)",
    "ziur aski euskalduna zen.": "(was almost certainly Basque-speaking.)",
    "martxoa baino lehen amaitu beharra dago.": "(must be finished before March.)",
}


def fmt(text, conf):
    """Format a completion sample."""
    if not text:
        return "*(empty)*"
    return f"`{text}` ({conf})"


def gen_ar_section():
    lines = []
    lines.append("## Appendix E: Three-Model Qualitative Comparison\n")
    lines.append("Top-3 sampled completions (temperature 0.7, 20 tokens) from each model.")
    lines.append("All prompts are **authentic Basque sentences** sourced from Wikipedia (eu),")
    lines.append("Berria newspaper, and the iberba.eus email-writing guide — no invented text.")
    lines.append("Gold continuations are the actual completions from the source documents.")
    lines.append("No evaluation is offered here; the reader is invited to compare.\n")
    lines.append("Models: **Morpheus** (91M Mamba-2, Q5_K_M), **Kimu 2B** (Gemma-2, Q6_K),")
    lines.append("**Latxa 8B** (Llama-3.1, Q6_K). All served via the same demo stack on an L40 GPU.\n")

    for i, entry in enumerate(data["ar"]):
        domain = entry["domain"]
        source = entry["source"]
        prompt = entry["prompt"]
        gold = entry["gold"]
        prompt_tr = AR_TRANSLATIONS.get(prompt, "")
        gold_tr = GOLD_TRANSLATIONS.get(gold, "")

        lines.append(f"### E.{i+1} {domain} — {source}\n")
        lines.append(f"> **Prompt:** {prompt}")
        if prompt_tr:
            lines.append(f"> {prompt_tr}")
        lines.append(f"> **Gold:** {gold}")
        if gold_tr:
            lines.append(f"> {gold_tr}")
        lines.append("")
        lines.append("| # | Morpheus | Kimu 2B | Latxa 8B |")
        lines.append("|---|----------|---------|----------|")
        for s in range(3):
            m = entry["models"]["morpheus"][s]
            k = entry["models"]["kimu"][s]
            l = entry["models"]["latxa"][s]
            lines.append(f"| {s+1} | {fmt(m['text'], m['confidence'])} | {fmt(k['text'], k['confidence'])} | {fmt(l['text'], l['confidence'])} |")
        lines.append("")

    return "\n".join(lines)


def gen_fim_section():
    lines = []
    lines.append("### FIM (Fill-in-the-Middle) Examples\n")
    lines.append("The prefix and suffix are given; the model must generate the **middle** (shown in bold in the full sentence).")
    lines.append("Only **Morpheus** was trained with FIM. Kimu 2B and Latxa 8B are base models without FIM")
    lines.append("training — they cannot attend to the suffix and typically generate from the prefix only,")
    lines.append("emit empty strings, or leak FIM sentinel tokens. Their columns are included to illustrate")
    lines.append("this capability gap (cf. §6.6 / §7.2).\n")

    for i, entry in enumerate(data["fim"]):
        domain = entry["domain"]
        source = entry["source"]
        prefix = entry["prefix"]
        middle = entry["middle"]
        suffix = entry["suffix"]
        full = entry["full"]

        # Build the display sentence with bold middle
        display = full.replace(middle, f"**{middle}**", 1)

        lines.append(f"### F.{i+1} {domain} — {source}\n")
        lines.append(f"> **Full sentence:** {display}")
        lines.append("")
        lines.append("| # | Morpheus (FIM) | Kimu 2B (no FIM) | Latxa 8B (no FIM) |")
        lines.append("|---|----------------|------------------|-------------------|")
        for s in range(3):
            m = entry["models"]["morpheus"][s]
            k = entry["models"]["kimu"][s]
            l = entry["models"]["latxa"][s]
            lines.append(f"| {s+1} | {fmt(m['text'], m['confidence'])} | {fmt(k['text'], k['confidence'])} | {fmt(l['text'], l['confidence'])} |")
        lines.append("")

    return "\n".join(lines)


md = gen_ar_section() + "\n---\n\n" + gen_fim_section()
print(md)
