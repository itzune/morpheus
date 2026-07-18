"""Pytest fixtures for editor browser tests.

Serves demo/static/ on a local port and provides a fresh Playwright
Chromium page per test. The /v1/complete API is mocked per-test via
``mock_complete`` so tests are fully deterministic — no Docker demo, no
GPU, no model. We test the *client logic*, not the model output.
"""
from __future__ import annotations

import http.server
import json
import pathlib
import socketserver
import threading

import pytest
from playwright.sync_api import sync_playwright

STATIC_DIR = pathlib.Path(__file__).resolve().parent.parent / "static"


class _Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(STATIC_DIR), **kw)

    def log_message(self, *_a):
        pass


@pytest.fixture(scope="session")
def base_url():
    """Serve demo/static/ on an ephemeral port for the whole session."""
    httpd = socketserver.ThreadingTCPServer(("127.0.0.1", 0), _Handler)
    httpd.daemon_threads = True
    port = httpd.server_address[1]
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{port}"
    httpd.shutdown()
    httpd.server_close()


@pytest.fixture
def page(base_url):
    """Fresh headless Chromium page per test."""
    with sync_playwright() as p:
        browser = p.chromium.launch(args=["--no-sandbox"])
        pg = browser.new_page()
        yield pg
        pg.close()
        browser.close()


@pytest.fixture
def editor_url(base_url):
    return f"{base_url}/editor.html"


@pytest.fixture
def mock_complete():
    """Install a deterministic /v1/complete route on the page.

    Usage::

        recorder = []
        mock_complete(page, [
            {"text": " zer moduz", "confidence": 0.5},
            {"text": " beste bat", "confidence": 0.4},
        ], recorder=recorder)
        page.goto(editor_url)
        # ... recorder[i] is the parsed request body of the i-th call

    Each request returns the next response in the list (cycling if
    exhausted). A response may be a callable ``(req, idx) -> dict`` for
    full flexibility. The parsed request body (prefix, suffix,
    temperature, ...) of every call is appended to ``recorder``.
    """
    def install(page, responses, recorder=None):
        state = {"i": 0}

        def handler(route):
            body = route.request.post_data
            try:
                req = json.loads(body) if body else {}
            except Exception:
                req = {}
            if recorder is not None:
                recorder.append(req)
            resp = responses[state["i"] % len(responses)]
            if callable(resp):
                resp = resp(req, state["i"])
            state["i"] += 1
            route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps(resp),
            )

        page.route("**/v1/complete", handler)
        return handler

    return install
