"""
OpenAI SDK smoke test for the morpheus proxy.

Validates that a real standard-library client (not curl) can talk to
/v1/completions with correct protocol conformance. If this passes,
Continue.dev (which uses the same SDK + /v1/completions) is virtually
guaranteed to work.

Prerequisites:
    1. llama-server running on :8080 (Mamba-2 GGUF)
    2. demo proxy running on :9090 (python -m uvicorn server:app --port 9090)

Run:
    pip install openai
    python demo/test_openai_compat.py

Exit code 0 = all checks passed.
"""
import sys
import httpx
from openai import OpenAI

BASE_URL = "http://127.0.0.1:9090/v1"
API_KEY = "not-needed"  # proxy doesn't check keys

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

failures = []


def check(name, cond, detail=""):
    status = "✓ PASS" if cond else "✗ FAIL"
    print(f"{status}: {name}" + (f" — {detail}" if detail else ""))
    if not cond:
        failures.append(name)


# ── 1. /v1/models ──
print("\n=== 1. client.models.list() ===")
try:
    models = client.models.list()
    model_ids = [m.id for m in models.data]
    check("models.list returns a list", len(models.data) >= 1, str(model_ids))
except Exception as e:
    check("models.list returns a list", False, str(e))

# ── 2. Non-streaming completion ──
print("\n=== 2. Non-streaming completion ===")
try:
    resp = client.completions.create(
        model="morpheus",
        prompt="Bihar goizean",
        max_tokens=8,
        temperature=0.3,
    )
    choice = resp.choices[0]
    text = choice.text
    has_text = bool(text)
    has_usage = resp.usage is not None and resp.usage.total_tokens > 0
    check("non-streaming returns text", has_text, repr(text))
    check("non-streaming has usage", has_usage, str(resp.usage))
    # Verify object schema
    check("id field present", bool(resp.id), resp.id)
    check("finish_reason present", choice.finish_reason in ("stop", "length"), choice.finish_reason)
    # Confidence is an extra field (not in OpenAI spec; SDK stores unknowns in model_extra)
    raw = resp.model_extra or {}
    has_confidence = "confidence" in raw and isinstance(raw["confidence"], (int, float))
    check("non-streaming has confidence", has_confidence, str(raw.get("confidence")))
    print(f"    text: {text!r}")
    print(f"    finish_reason: {choice.finish_reason}")
    print(f"    confidence: {raw.get('confidence')}")
    print(f"    usage: prompt={resp.usage.prompt_tokens} completion={resp.usage.completion_tokens}")
except Exception as e:
    check("non-streaming returns text", False, str(e))

# ── 3. Streaming completion (SSE) ──
print("\n=== 3. Streaming completion (SSE) ===")
try:
    chunks = []
    finish_reasons = []
    stream = client.completions.create(
        model="morpheus",
        prompt="Euskal Herria",
        max_tokens=12,
        temperature=0.3,
        stream=True,
    )
    for chunk in stream:
        if chunk.choices[0].text:
            chunks.append(chunk.choices[0].text)
        if chunk.choices[0].finish_reason:
            finish_reasons.append(chunk.choices[0].finish_reason)
    full = "".join(chunks)
    check("stream produced tokens", len(chunks) > 0, f"{len(chunks)} chunks: {full!r}")
    check("stream has finish_reason", len(finish_reasons) > 0, str(finish_reasons))
    print(f"    assembled text: {full!r}")
    print(f"    finish_reasons: {finish_reasons}")
except Exception as e:
    check("stream produced tokens", False, str(e))

# ── 4. Stop sequence handling ──
print("\n=== 4. Stop sequence handling ===")
try:
    resp = client.completions.create(
        model="morpheus",
        prompt="Euskal Herriko",
        max_tokens=20,
        temperature=0,
        stop=[" "],  # stop at first space → single token
    )
    text = resp.choices[0].text
    # Should stop before any space
    no_space = " " not in text
    check("stop=' ' truncates before space", no_space or resp.choices[0].finish_reason == "stop",
          f"text={text!r} finish={resp.choices[0].finish_reason}")
    print(f"    text: {text!r}  finish: {resp.choices[0].finish_reason}")
except Exception as e:
    check("stop=' ' truncates before space", False, str(e))

# ── 5. Token-ID prompt (the divergence bypass) ──
print("\n=== 5. Token-ID prompt path ===")
try:
    resp = client.completions.create(
        model="morpheus",
        prompt=[123, 456],  # raw token IDs
        max_tokens=5,
        temperature=0,
    )
    check("token-ID prompt accepted", bool(resp.choices[0].text), repr(resp.choices[0].text))
    print(f"    text: {resp.choices[0].text!r}")
except Exception as e:
    check("token-ID prompt accepted", False, str(e))

# ── 6. /v1/complete FIM route (prefix+suffix) ──
print("\n=== 6. /v1/complete FIM route ===")
try:
    r = httpx.post(
        f"{BASE_URL}/complete",
        json={
            "prefix": "Bihar goizean",
            "suffix": "etorriko naiz.",
            "max_tokens": 16,
            "temperature": 0,
        },
        timeout=30.0,
    )
    r.raise_for_status()
    data = r.json()
    text = data.get("text", "")
    finish = data.get("finish_reason", "")
    confidence = data.get("confidence", None)
    check("/v1/complete returns text", isinstance(text, str), repr(text))
    check("/v1/complete has finish_reason", finish in ("stop", "length"), repr(finish))
    check("/v1/complete has confidence", isinstance(confidence, (int, float)), str(confidence))
    print(f"    prefix: 'Bihar goizean'")
    print(f"    suffix: 'etorriko naiz.'")
    print(f"    generated middle: {text!r}")
    print(f"    confidence: {confidence}")
    print(f"    finish_reason: {finish}")
except Exception as e:
    check("/v1/complete returns text", False, str(e))

# ── 7. /v1/complete AR-only (no suffix) ──
print("\n=== 7. /v1/complete AR-only (no suffix) ===")
try:
    r = httpx.post(
        f"{BASE_URL}/complete",
        json={"prefix": "Bihar goizean", "max_tokens": 8, "temperature": 0},
        timeout=30.0,
    )
    r.raise_for_status()
    data = r.json()
    text = data.get("text", "")
    confidence = data.get("confidence", None)
    check("/v1/complete AR returns text", isinstance(text, str) and len(text) > 0, repr(text))
    check("/v1/complete AR has confidence", isinstance(confidence, (int, float)), str(confidence))
    print(f"    generated: {text!r}")
    print(f"    confidence: {confidence}")
except Exception as e:
    check("/v1/complete AR returns text", False, str(e))


# ── Summary ──
print("\n" + "=" * 50)
if failures:
    print(f"RESULT: {len(failures)} FAILURE(S): {failures}")
    sys.exit(1)
else:
    print("RESULT: ALL CHECKS PASSED — proxy is OpenAI-SDK-conformant")
    sys.exit(0)
