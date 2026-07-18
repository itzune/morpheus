#!/usr/bin/env python3
"""Raw AR test directly against llama-server."""
import sentencepiece as spm
import json, urllib.request

sp = spm.SentencePieceProcessor()
sp.Load("tokenizer/basque_unigram_fim.model")

tests = ["Ni atzo ", "Euskal Herriko ", "Bihar goizean ", "Gaur egun, euskara "]

print("-- RAW AR mode (append-only) --\n")
for prompt in tests:
    prompt_ids = sp.encode(prompt, out_type=int)
    payload = json.dumps({
        "prompt": prompt_ids,
        "n_predict": 15,
        "stream": False,
        "temperature": 0.0,
        "top_k": 0,
        "repeat_penalty": 1.0,
        "stop": ["\n"],
    }).encode()
    req = urllib.request.Request("http://localhost:8080/completion", data=payload,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read())
    content = data.get("content", "")
    print(f"  [{prompt}] -> {repr(content)}")
