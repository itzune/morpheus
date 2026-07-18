"""Unit tests for server-side best-of-n sampling.

Tests _pick_best_of_n — the selection logic that picks the highest-
confidence non-empty result from n parallel samples. No model, no
Docker, no network — pure function tests.

Run (from demo/):
    uv run pytest tests/ -v
"""
from __future__ import annotations

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from server import _pick_best_of_n


def test_single_result_returned():
    """best-of-1 just returns the single result."""
    r = _pick_best_of_n([("hello", 0.5, "stop")])
    assert r == ("hello", 0.5, "stop")


def test_picks_highest_confidence():
    """Among n samples, the highest-confidence one wins."""
    results = [
        ("zer", 0.3, "stop"),
        ("best", 0.6, "stop"),
        ("other", 0.4, "stop"),
    ]
    text, conf, _ = _pick_best_of_n(results)
    assert text == "best"
    assert conf == 0.6


def test_empty_results_skipped():
    """Empty-text samples (model returned nothing) are skipped — we only
    consider samples that actually produced text."""
    results = [
        ("", 0.0, "stop"),      # empty — skip
        ("real", 0.4, "stop"),   # only non-empty — pick this
        ("", 0.0, "stop"),      # empty — skip
    ]
    text, conf, _ = _pick_best_of_n(results)
    assert text == "real"
    assert conf == 0.4


def test_all_empty_returns_empty():
    """If all samples are empty, return empty (not an exception)."""
    results = [("", 0.0, "stop"), ("", 0.0, "stop")]
    text, conf, finish = _pick_best_of_n(results)
    assert text == ""
    assert conf == 0.0
    assert finish == "stop"


def test_exceptions_skipped():
    """Failed parallel requests (exceptions) are skipped — one bad sample
    shouldn't sink the whole best-of-n request."""
    results = [
        Exception("connection error"),
        ("survivor", 0.5, "stop"),
        Exception("timeout"),
    ]
    text, conf, _ = _pick_best_of_n(results)
    assert text == "survivor"


def test_all_exceptions_returns_empty():
    """If all requests fail, return empty gracefully."""
    results = [Exception("err1"), Exception("err2")]
    text, conf, finish = _pick_best_of_n(results)
    assert text == ""
    assert conf == 0.0
    assert finish == "stop"


def test_tie_picks_first_max():
    """On a confidence tie, max() returns the first occurrence (stable)."""
    results = [
        ("first", 0.5, "stop"),
        ("second", 0.5, "stop"),
    ]
    text, _, _ = _pick_best_of_n(results)
    assert text == "first"
