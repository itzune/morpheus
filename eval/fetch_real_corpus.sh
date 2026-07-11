#!/usr/bin/env bash
# Fetch clean Basque prose (Wikipedia + Berria) for the real-corpus eval.
# Output: eval/real_corpus/*.txt
# Reproducible: fixed article set. Re-run anytime to refresh.
set -euo pipefail
cd "$(dirname "$0")/.."   # repo root
mkdir -p eval/real_corpus

echo "== Wikipedia extracts (MediaWiki API, plain text) =="
WIKI_TITLES="Euskara Euskal_Herria Donostia Bilbo Nafarroa_Garaia Gipuzkoa Euskara_batua Hezkuntza Musika"
for t in $WIKI_TITLES; do
  curl -s --max-time 30 \
    "https://eu.wikipedia.org/w/api.php?action=query&prop=extracts&explaintext=1&redirects=1&titles=${t}&format=json" \
    | python3 -c "import sys,json; d=json.load(sys.stdin); p=list(d['query']['pages'].values())[0]; print(p.get('extract','(no extract)'))" \
    > "eval/real_corpus/wiki_${t}.txt"
  echo "  wiki_${t}.txt: $(wc -w < eval/real_corpus/wiki_${t}.txt) words"
done

echo "== Berria articles (HTML <p> extraction) =="
python3 - <<'PY'
import urllib.request, re
from html.parser import HTMLParser
URLS = [
    "https://www.berria.eus/bizigiro/eta-hala-ere-zutik-dirau-oroitarriak_2160788_102.html",
    "https://www.berria.eus/ekonomia/labek-ez-du-1500-eurotik-beherako-soldatarik-onartuko-sektore-hitzarmenetan_2160823_102.html",
    "https://www.berria.eus/euskal-herria/atzerritarrak-kanporatzeko-txostenak-egin-ahalko-ditu-ertzaintzak_2160848_102.html",
    "https://www.berria.eus/bizigiro/prozesua-obra-bihurtuta_2160796_102.html",
    "https://www.berria.eus/ekonomia/lan-harremanen-kontseiluaren-ustez-euskaraz-negoziatzea-ez-da-arazo-akordioetara-iristekoa_2160841_102.html",
]
class P(HTMLParser):
    def __init__(s):
        super().__init__(); s.in_p=False; s.buf=[]; s.paras=[]
    def handle_starttag(s,t,a):
        if t=="p": s.in_p=True; s.buf=[]
    def handle_endtag(s,t):
        if t=="p" and s.in_p:
            s.in_p=False; x=" ".join(s.buf).strip(); x=re.sub(r"\s+"," ",x)
            if len(x)>40: s.paras.append(x)
    def handle_data(s,d):
        if s.in_p: s.buf.append(d)
for i,u in enumerate(URLS):
    try:
        req=urllib.request.Request(u, headers={"User-Agent":"Mozilla/5.0"})
        raw=urllib.request.urlopen(req,timeout=25).read().decode("utf-8","replace")
        p=P(); p.feed(raw); txt="\n".join(p.paras)
        open(f"eval/real_corpus/berria_{i}.txt","w").write(txt)
        print(f"  berria_{i}.txt: {len(txt.split())} words")
    except Exception as e:
        print(f"  berria_{i}.txt: ERROR {e}")
PY

echo ""
echo "Done. Now run:  python3 demo/extract_real_prompts.py"
