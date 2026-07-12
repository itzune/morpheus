#!/usr/bin/env python3
"""
fetch_domain_corpus.py — Fetch domain-stratified Basque text for eval.

Domains:
  1. news       — Berria.eus (politics, sports, culture, economy)
  2. legal      — BOPV/EHAA from euskadi.eus (laws/administrative)
  3. education  — Jakinbai.eus content pages (technical/vocational)
  4. literature — Armiarma klasikoak (classical Basque literature)
  5. blog       — Berria.eus blog columns (informal/opinion)

Output: eval/domain_corpus/<domain>_<n>.txt
"""
import urllib.request
import re
import os
import time
from html.parser import HTMLParser
from pathlib import Path

OUT_DIR = Path(__file__).parent / "domain_corpus"
OUT_DIR.mkdir(exist_ok=True)

HEADERS = {"User-Agent": "Mozilla/5.0 (Morpheus research eval)"}


def fetch(url, timeout=30):
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
        # Detect encoding from Content-Type header or default to utf-8
        ct = resp.headers.get("Content-Type", "")
        encoding = "utf-8"
        if "charset=" in ct:
            encoding = ct.split("charset=")[-1].strip()
        try:
            return raw.decode(encoding, errors="replace")
        except (LookupError, TypeError):
            # Fall back: try latin-1 (common for older Basque sites)
            return raw.decode("latin-1", errors="replace")


class TextExtractor(HTMLParser):
    """Extract text from <p>, <div>, <li> tags, skip nav/script/style."""
    SKIP = {"script", "style", "nav", "header", "footer", "aside", "form", "button"}
    COLLECT = {"p", "li", "h1", "h2", "h3", "h4", "div"}

    def __init__(self):
        super().__init__()
        self.skip_depth = 0
        self.in_collect = False
        self.buf = []
        self.paras = []

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self.skip_depth += 1
        elif tag in self.COLLECT and self.skip_depth == 0:
            # If already collecting, save current paragraph first
            # (handles unclosed <p> tags common in old HTML)
            if self.in_collect:
                text = " ".join(self.buf).strip()
                text = re.sub(r"\s+", " ", text)
                if len(text) > 40:
                    self.paras.append(text)
            self.in_collect = True
            self.buf = []

    def handle_endtag(self, tag):
        if tag in self.SKIP:
            self.skip_depth = max(0, self.skip_depth - 1)
        elif tag in self.COLLECT and self.in_collect:
            self.in_collect = False
            text = " ".join(self.buf).strip()
            text = re.sub(r"\s+", " ", text)
            if len(text) > 40:
                self.paras.append(text)

    def handle_data(self, data):
        if self.in_collect:
            self.buf.append(data)


def extract_text(html):
    parser = TextExtractor()
    parser.feed(html)
    return "\n\n".join(parser.paras)


# Spanish indicator words (high frequency in Spanish, absent/rare in Basque)
_SPANISH_WORDS = frozenset(
    "que por del los las una para con mas ser ha su como pero cuando "
    "donde sin sobre hasta desde entre durante antes despues porque "
    "tambien ya ocho nueve diez veinte treinta cuarenta cincuenta sesenta "
    "setenta ochenta noventa cien mil millones "
    "directora director general salud servicio vasco resolutions "
    "mediante resolucion julio noviembre diciembre enero febrero marzo "
    "abril mayo junio agosto septiembre octubre".split()
)


def is_basque(text):
    """Heuristic: returns True if paragraph is likely Basque, not Spanish."""
    words = set(text.lower().split())
    if not words:
        return False
    spanish_hits = len(words & _SPANISH_WORDS)
    # If >15% of unique words are Spanish indicators, it's Spanish
    return spanish_hits / len(words) < 0.05


def clean_text(text, basque_only=False):
    """Remove JS artifacts, filter by language if requested."""
    lines = []
    for para in text.split("\n\n"):
        para = para.strip()
        # Skip JS/document.write artifacts
        if "document.write" in para or "function(" in para:
            continue
        if para.startswith("var ") or "$.ajax" in para:
            continue
        if len(para) < 40:
            continue
        if basque_only and not is_basque(para):
            continue
        lines.append(para)
    return "\n\n".join(lines)


def save(domain, idx, text):
    fname = OUT_DIR / f"{domain}_{idx}.txt"
    fname.write_text(text, encoding="utf-8")
    print(f"  {fname.name}: {len(text)} chars, {len(text.split())} words")
    return len(text)


# ─────────────────────────────────────────────────────
# 1. NEWS — Berria.eus (multiple sections)
# ─────────────────────────────────────────────────────
def fetch_berria():
    print("\n== NEWS: Berria.eus ==")
    urls = [
        # Politics / Euskal Herria
        "https://www.berria.eus/euskal-herria/hildako-bat-errenterian-arma-zuriz-egindako-eraso-bat-jasan-ostean_2160994_102.html",
        "https://www.berria.eus/euskal-herria/berotik-babesteko-azpiegiturak-inoiz-baino-beharrezkoago_2160955_102.html",
        # International / Mundua
        "https://www.berria.eus/mundua/aebek-eta-iranek-berriro-egin-diote-eraso-elkarri-eta-ormuzko-itsasartea-itxita-geratu-da_2160993_102.html",
        # Economy / Ekonomia
        "https://www.berria.eus/ekonomia/singapur-errusiako-petrolioa-garbitzeko-makina-isila_2160156_102.html",
        "https://www.berria.eus/ekonomia/volkswagen-taldeak-auto-modelo-gutxiago-egin-ditu_2160928_102.html",
        # Culture / Kultura
        "https://www.berria.eus/kultura/gasteiz-jazzaren-plaza-irekia_2160864_102.html",
        "https://www.berria.eus/kultura/zenbaki-baskoniko-baten-lehenbiziko-lekukotasuna-izan-daitekeen-inskripzio-bat-aurkitu-dute-irulegin_2160226_102.html",
        # Sports / Kirola
        "https://www.berria.eus/kirola/van-der-poelek-ihesaldia-borobildu-eta-hirugarren-etapa-garaipena-lortu-du-tourrean_2160999_102.html",
        "https://www.berria.eus/kirola/oriok-etxeko-estropada-irabazi-du-gizonezkoetan_2160997_102.html",
        # Opinion / Iritzia
        "https://www.berria.eus/iritzia/bira/eguzki-sartzeak_2160870_102.html",
        "https://www.berria.eus/iritzia/artikuluak/gurasoen-etxea_2160790_102.html",
        # Society / Bizigiro
        "https://www.berria.eus/bizigiro/ondareari-soinua-dario_2160840_102.html",
        # Literature / Udako narrazioak
        "https://www.berria.eus/udako-narrazioak/dena-has-daiteke-arreta-berreskuratzeko-metodoa_2160048_102.html",
    ]
    total = 0
    for i, url in enumerate(urls):
        try:
            html = fetch(url)
            text = extract_text(html)
            text = clean_text(text)
            if len(text) > 200:
                total += save("news", i, text)
            time.sleep(1)
        except Exception as e:
            print(f"  news_{i}: FAILED ({e})")
    return total


# ─────────────────────────────────────────────────────
# 2. LEGAL — BOPV/EHAA (euskadi.eus)
# ─────────────────────────────────────────────────────
def fetch_legal():
    print("\n== LEGAL: BOPV/EHAA ==")
    # BOPV parallel text viewer — uses Basque text
    # Try recent BOPV entries
    urls = [
        "https://www.euskadi.eus/bopv2/datos/2025/07/2501980e.pdf",  # PDF — skip
        # Try the web viewer with Basque language
        "https://www.euskadi.eus/web01-bopv/eu/p43aBOPVWebWar/VerParalelo.do?R01HPortal=y22&R01HPage=bopv&R01HLang=eu&ed2025003108",
        "https://www.euskadi.eus/web01-bopv/eu/p43aBOPVWebWar/VerParalelo.do?R01HPortal=y22&R01HPage=bopv&R01HLang=eu&ed2025003107",
        "https://www.euskadi.eus/web01-bopv/eu/p43aBOPVWebWar/VerParalelo.do?R01HPortal=y22&R01HPage=bopv&R01HLang=eu&ed2025003106",
    ]
    total = 0
    for i, url in enumerate(urls):
        if url.endswith(".pdf"):
            continue
        try:
            html = fetch(url, timeout=30)
            text = extract_text(html)
            text = clean_text(text, basque_only=True)
            if len(text) > 200:
                total += save("legal", i, text)
            time.sleep(1)
        except Exception as e:
            print(f"  legal_{i}: FAILED ({e})")
    return total


# ─────────────────────────────────────────────────────
# 3. EDUCATION — Jakinbai.eus content pages
# ─────────────────────────────────────────────────────
def fetch_education():
    print("\n== EDUCATION: Jakinbai.eus ==")
    urls = [
        "https://jakinbai.eus/edukiak/harrera-eta-erreserbak",
        "https://jakinbai.eus/edukiak/diseinu-grafiko-aplikatua",
        "https://jakinbai.eus/edukiak/marrazketa-eta-eraikuntza-teknikak-zurezko-egituretan",
        "https://jakinbai.eus/edukiak/enplegurako-ibilbide-pertsonala",
        "https://jakinbai.eus/edukiak/sareak-administratzea-eta-planifikatzea-bideotutorialak",
        "https://jakinbai.eus/edukiak/automatismo-hidraulikoak",
    ]
    total = 0
    for i, url in enumerate(urls):
        try:
            html = fetch(url)
            text = extract_text(html)
            text = clean_text(text)
            if len(text) > 200:
                total += save("education", i, text)
            time.sleep(1)
        except Exception as e:
            print(f"  education_{i}: FAILED ({e})")
    return total


# ─────────────────────────────────────────────────────
# 4. LITERATURE — Armiarma klasikoak (classical Basque)
# ─────────────────────────────────────────────────────
def fetch_literature():
    print("\n== LITERATURE: Armiarma klasikoak ==")
    # Fetch several text pages from the 19th century collection
    base = "https://klasikoak.armiarma.eus/testuak/"
    urls = [
        base + "testuak19001.htm",
        base + "testuak19002.htm",
        base + "testuak19017.htm",
        base + "testuak19020.htm",
        base + "testuak19035.htm",
        base + "testuak19042.htm",
        # Also try 17th century texts
        base + "testuak17001.htm",
        base + "testuak17016.htm",
    ]
    total = 0
    for i, url in enumerate(urls):
        try:
            html = fetch(url)
            text = extract_text(html)
            text = clean_text(text)
            if len(text) > 200:
                total += save("literature", i, text)
            time.sleep(1)
        except Exception as e:
            print(f"  literature_{i}: FAILED ({e})")
    return total


# ─────────────────────────────────────────────────────
# 5. WIKIPEDIA (additional articles not in current eval)
# ─────────────────────────────────────────────────────
def fetch_wikipedia():
    print("\n== WIKIPEDIA: Additional articles ==")
    titles = [
        "Euskal_Herriko_historia",
        "Nafarroako_Foru_Komunitatea",
        "Euskaltzaindia",
        "Bertsolaritza",
        "Euskal_Literatura",
        "Pello_Joxepe",
        "Lurraldearen_plangintza",
        "Biodibertsitate",
        "Energia_Berriztagarri",
        "Zientzia",
    ]
    total = 0
    for i, t in enumerate(titles):
        try:
            url = (f"https://eu.wikipedia.org/w/api.php?action=query"
                   f"&prop=extracts&explaintext=1&redirects=1&titles={t}&format=json")
            data = fetch(url)
            import json
            d = json.loads(data)
            pages = d.get("query", {}).get("pages", {})
            for pid, p in pages.items():
                text = p.get("extract", "")
                if len(text) > 200:
                    total += save("wiki", i, text)
            time.sleep(1)
        except Exception as e:
            print(f"  wiki_{i}: FAILED ({e})")
    return total


if __name__ == "__main__":
    print(f"Output directory: {OUT_DIR}")
    totals = {}
    totals["news"] = fetch_berria()
    totals["legal"] = fetch_legal()
    totals["education"] = fetch_education()
    totals["literature"] = fetch_literature()
    totals["wiki"] = fetch_wikipedia()

    print("\n" + "=" * 50)
    print("DOMAIN CORPUS SUMMARY")
    print("=" * 50)
    for domain, total in totals.items():
        files = list(OUT_DIR.glob(f"{domain}_*.txt"))
        print(f"  {domain:12s}: {len(files):2d} files, {total:>8d} chars")
    total_chars = sum(t for t in totals.values())
    print(f"  {'TOTAL':12s}: {total_chars:>8d} chars")
