"""Unit tests for demo/server.py filter logic.

Run:  python3 demo/test_filters.py
"""

import sys
sys.path.insert(0, '.')
from demo.server import smart_context, ghost_suffix, filter_suggestion


# ---------------------------------------------------------------------------
# smart_context — strips only non-▁ tokens (true subword fragments)
# ---------------------------------------------------------------------------
def test_smart_context():
    cases = [
        # (input, expected)
        ("Kaixo, zer mod",   "Kaixo, zer mo"),     # mo ▁ token kept, d stripped
        ("Kaixo, zer moduz", "Kaixo, zer modu"),   # z stripped
        ("Kaixo",            "Kaixo"),              # single word kept
        ("Kaixo ",           "Kaixo "),             # trailing space → keep all
        ("Gaur nire ama berandu iri",  "Gaur nire ama berandu iri"),   # all ▁ tokens
        ("Gaur nire ama berandu irit", "Gaur nire ama berandu iri"),   # t stripped
        ("", ""),
        ("  ", "  "),
        ("Eskerrik asko",    "Eskerrik asko"),      # full word
    ]
    failures = []
    for text, expected in cases:
        result = smart_context(text)
        if result != expected:
            failures.append(f"  smart_context({text!r}) = {result!r}, expected {expected!r}")
    return failures


# ---------------------------------------------------------------------------
# ghost_suffix — computes the non-overlapping suffix for ghost display
# ---------------------------------------------------------------------------
def test_ghost_suffix():
    cases = [
        # (text, smart_ctx, suggestion, expected_ghost)
        # Same punct at boundary → strip from suggestion
        ("Kaixo?",  "Kaixo?",  "? zer moduz", " zer moduz"),
        ("Kaixo!",  "Kaixo!",  "! zer moduz", " zer moduz"),
        ("Kaixo.",  "Kaixo.",  ". Agur",      " Agur"),
        # Different punct → keep
        ("Kaixo.",  "Kaixo.",  "? zer moduz", "? zer moduz"),
        # Overlap (user typed part of prediction) — Smart Compose style
        ("Kaixo, zer mod", "Kaixo, zer", " moduz?", "uz?"),
        # Excluded text overlaps with suggestion prefix
        ("Euskal He", "Euskal", " Herria", "rria"),
    ]
    failures = []
    for text, ctx, suggestion, expected in cases:
        result = ghost_suffix(text, ctx, suggestion)
        if result != expected:
            failures.append(f"  ghost_suffix({text!r}, {ctx!r}, {suggestion!r}) = {result!r}, expected {expected!r}")
    return failures


# ---------------------------------------------------------------------------
# filter_suggestion — cleans junk from model predictions
# ---------------------------------------------------------------------------
def test_filter_suggestion():
    cases = [
        # (input, expected)

        # Leading space preserved (word boundary signal)
        (" bezala..",                            " bezala."),

        # Runs of same punct collapsed (anywhere)
        ("da??) eta zer da??",                   "da? eta zer da?"),
        ("koskorra da??) eta zer da??",          "koskorra da? eta zer da?"),
        ("..",                                   "."),
        ("...",                                  "."),
        ("????",                                 "?"),
        ("eskerrik asko..",                      "eskerrik asko."),

        # Mixed punct at end → keep first only
        ("LLC.,",                                "LLC."),
        ("z?,.",                                 "z?"),
        ("test!.,?;",                            "test!"),

        # Space-separated punct junk → stripped
        ("kaixo .,",                             "kaixo"),
        ("hello . ,",                            "hello"),
        ("eskerrik asko . ,",                    "eskerrik asko"),

        # Pure punct junk → empty
        (" .,",                                  ""),
        (" , . ;",                               ""),

        # Bare punct without whitespace → keep (legit sentence end)
        (".",                                    "."),
        ("?",                                    "?"),

        # Whitespace + punct → reject (junk)
        (" .",                                   ""),
        (" ,",                                   ""),

        # Normal text untouched
        ("kaixo",                                "kaixo"),
        ("kaixo?",                               "kaixo?"),
        ("no.",                                  "no."),
        ("koskorra",                             "koskorra"),

        # U+FFFD replacement char stripped
        ("\ufffd",                               ""),
        ("..\ufffd",                             "."),
        ("kaixo\ufffd!",                         "kaixo!"),
        ("\ufffd\ufffd\ufffd",                   ""),

        # ▁ markers → spaces
        ("▁▁hi",                                 " hi"),
    ]
    failures = []
    for inp, expected in cases:
        result = filter_suggestion(inp)
        if result != expected:
            failures.append(f"  filter_suggestion({inp!r}) = {result!r}, expected {expected!r}")
    return failures


# ---------------------------------------------------------------------------
# Integration: smart_context + filter_suggestion + ghost_suffix pipeline
# ---------------------------------------------------------------------------
def test_pipeline():
    """Simulate what the API does: smart_ctx → filter(suggestion) → ghost."""
    cases = [
        # (user_text, raw_model_output, expected_ghost)
        ("Kaixo, zer ",                  "koskorra",               "koskorra"),
        ("Kaixo, zer moduz? Ni ondo, beti", " bezala..",           " bezala."),
        ("Gaur nire ama berandu irit",   "si da.",                 "si da."),
        ("Eskerrik asko",                " asko.",                 " asko."),
    ]
    failures = []
    for text, raw, expected_ghost in cases:
        ctx = smart_context(text)
        suggestion = filter_suggestion(raw)
        ghost = ghost_suffix(text, ctx, suggestion)
        if ghost != expected_ghost:
            failures.append(
                f"  pipeline({text!r}):\n"
                f"    smart_ctx={ctx!r}\n"
                f"    suggestion={suggestion!r}\n"
                f"    ghost={ghost!r}, expected {expected_ghost!r}"
            )
    return failures


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    suites = [
        ("smart_context", test_smart_context),
        ("ghost_suffix", test_ghost_suffix),
        ("filter_suggestion", test_filter_suggestion),
        ("pipeline (integration)", test_pipeline),
    ]

    total = 0
    for name, fn in suites:
        failures = fn()
        status = "PASS" if not failures else f"FAIL ({len(failures)})"
        print(f"\n{'─'*60}")
        print(f"  {name}: {status}")
        print(f"{'─'*60}")
        for f in failures:
            print(f)
            total += 1

    print(f"\n{'='*60}")
    if total == 0:
        print("  ALL TESTS PASSED ✓")
    else:
        print(f"  {total} FAILURE(S) ✗")
    print(f"{'='*60}\n")
    sys.exit(0 if total == 0 else 1)
