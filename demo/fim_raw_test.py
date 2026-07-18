#!/usr/bin/env python3
"""Raw FIM test directly against llama-server (bypass proxy postprocessing)."""
import sentencepiece as spm
import json, urllib.request

sp = spm.SentencePieceProcessor()
sp.Load("tokenizer/basque_unigram_fim.model")

PRE = sp.piece_to_id("<PRE>")
SUF = sp.piece_to_id("<SUF>")
MID = sp.piece_to_id("<MID>")

tests = [
    ("Kaixo, ", " moduzu?", "zer"),
    ("Ni atzo amonaren etxera joan ", " bazkaltzera", "nintzen"),
    ("Zein da zure ", "? Ni Xabi naiz", "izena"),
    ("Bihar ", " elkartuko gara", "goizean"),
    ("Gaur egun, euskara ", " nagusia da", "mintzaira"),
    ("Nire ", " etxera etorri da.", "aita"),
    ("Euskera ", " mintzaira ofiziala da.", "da"),
]

print("── RAW llama-server FIM (no proxy, no postprocessing) ──\n")
for prefix, suffix, expected in tests:
    prefix_ids = sp.encode(prefix, out_type=int)
    suffix_ids = sp.encode(suffix, out_type=int)
    prompt_ids = [PRE] + prefix_ids + [SUF] + suffix_ids + [MID]

    payload = json.dumps({
        "prompt": prompt_ids,
        "n_predict": 30,
        "stream": False,
        "temperature": 0.0,
        "top_k": 0,
        "top_p": 1.0,
        "repeat_penalty": 1.0,
        "stop": ["<EOT>", "\n\n"],
    }).encode()

    req = urllib.request.Request("http://localhost:8080/completion", data=payload,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read())

    content = data.get("content", "")
    stop_str = data.get("stop", "")
    stopped_eot = "<EOT>" in str(stop_str)
    print(f"  Prefix:   {repr(prefix)}")
    print(f"  Suffix:   {repr(suffix)}")
    print(f"  Expected: {repr(expected)}")
    print(f"  Got:      {repr(content)}")
    print(f"  Stopped:  {'<EOT>' if stopped_eot else stop_str or 'max'}")
    print()
