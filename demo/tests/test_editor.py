"""Editor behavior tests (deterministic, mocked API).

These test the *client logic* of demo/static/editor.html: ghost-text
rendering, keybindings (Tab / Ctrl+Right / Esc / Alt+] / Alt+[), FIM vs
AR auto-selection, and the confidence gate. The /v1/complete API is
mocked so results are deterministic and independent of the model.

Run (from demo/):
    uv run pytest tests/ -v
"""
from __future__ import annotations

from playwright.sync_api import expect

# ── locators ──────────────────────────────────────────────────────────
EDITOR = "#editor"
GHOST = "#ghost"
GHOST_TEXT = ".ghost-text"
STATUS = "#status"
STATS = "#stats"


def ghost_text(page):
    """The current suggestion text shown in the ghost span (or '')."""
    loc = page.locator(GHOST_TEXT)
    return loc.text_content() if loc.count() else ""


# ── rendering / regression ────────────────────────────────────────────


def test_textarea_is_transparent_ghost_renders_text(page, editor_url, mock_complete):
    """Regression for the FIM overlap bug: the textarea must be transparent
    (only provides the caret) and the ghost layer renders all visible text.
    If someone reverts the transparent-textarea fix, this catches it."""
    mock_complete(page, [{"text": "", "confidence": 0}])
    page.goto(editor_url)

    editor_color = page.locator(EDITOR).evaluate("el => getComputedStyle(el).color")
    ghost_color = page.locator(GHOST).evaluate("el => getComputedStyle(el).color")

    assert editor_color in ("rgba(0, 0, 0, 0)", "transparent"), (
        f"textarea text must be transparent, got {editor_color!r}"
    )
    # ghost layer should render real text in a visible (non-transparent) color
    assert ghost_color not in ("rgba(0, 0, 0, 0)", "transparent"), (
        f"ghost layer must be visible, got {ghost_color!r}"
    )


def test_initial_suggestion_appears(page, editor_url, mock_complete):
    """On load the editor seeds text and auto-requests a suggestion, which
    must render in the ghost span."""
    mock_complete(page, [{"text": " zer moduz", "confidence": 0.5}])
    page.goto(editor_url)

    expect(page.locator(GHOST_TEXT)).to_contain_text("zer moduz", timeout=5000)


# ── keybindings ───────────────────────────────────────────────────────


def test_tab_accepts_full_suggestion(page, editor_url, mock_complete):
    mock_complete(page, [
        {"text": " zer moduz", "confidence": 0.5},   # initial
        {"text": "", "confidence": 0},                # after accept (no new ghost)
    ])
    page.goto(editor_url)
    expect(page.locator(GHOST_TEXT)).to_contain_text("zer moduz", timeout=5000)

    page.locator(EDITOR).press("Tab")

    # The suggestion text is now committed into the editor value...
    value = page.locator(EDITOR).input_value()
    assert "zer moduz" in value
    # ...and the ghost is cleared.
    expect(page.locator(GHOST_TEXT)).to_have_count(0, timeout=3000)


def test_ctrl_right_accepts_next_word(page, editor_url, mock_complete):
    """Ctrl+Right inserts only the first word and keeps the remainder as
    the new ghost (VS Code / Copilot convention)."""
    mock_complete(page, [{"text": " zer moduz?", "confidence": 0.5}])
    page.goto(editor_url)
    expect(page.locator(GHOST_TEXT)).to_contain_text("zer moduz", timeout=5000)

    page.locator(EDITOR).press("Control+ArrowRight")

    # First word ("zer") committed; remainder ("moduz?") stays as ghost.
    value = page.locator(EDITOR).input_value()
    assert "zer" in value
    expect(page.locator(GHOST_TEXT)).to_contain_text("moduz", timeout=3000)
    # And no extra request was made (remainder kept, not re-requested).
    # Ghost is the remainder, not a fresh suggestion.


def test_escape_dismisses_suggestion(page, editor_url, mock_complete):
    mock_complete(page, [{"text": " zer moduz", "confidence": 0.5}])
    page.goto(editor_url)
    expect(page.locator(GHOST_TEXT)).to_contain_text("zer moduz", timeout=5000)
    value_before = page.locator(EDITOR).input_value()

    page.locator(EDITOR).press("Escape")

    expect(page.locator(GHOST_TEXT)).to_have_count(0, timeout=3000)
    # Editor text unchanged.
    assert page.locator(EDITOR).input_value() == value_before


# ── FIM vs AR auto-selection ──────────────────────────────────────────


def test_ar_mode_when_cursor_at_end(page, editor_url, mock_complete):
    """Cursor at end of text → suffix is empty → AR (append) path."""
    recorder = []
    mock_complete(page, [{"text": " gehiago", "confidence": 0.5}], recorder=recorder)
    page.goto(editor_url)
    expect(page.locator(GHOST_TEXT)).to_contain_text("gehiago", timeout=5000)

    assert any(r.get("suffix", "__missing__") == "" for r in recorder), (
        "expected at least one AR request with empty suffix"
    )


def test_fim_mode_when_cursor_in_middle(page, editor_url, mock_complete):
    """Cursor mid-sentence → suffix is non-empty → FIM (infill) path."""
    recorder = []
    mock_complete(page, [{"text": " zer", "confidence": 0.5}], recorder=recorder)
    page.goto(editor_url)

    # Place the cursor in the middle of the text and trigger a request.
    page.locator(EDITOR).evaluate(
        """() => {
            const e = document.getElementById('editor');
            e.value = 'Kaixo, zer moduz? Ni atzo etorri nintzen.';
            e.setSelectionRange(13, 13);  // cursor mid-sentence
            e.dispatchEvent(new Event('input'));
        }"""
    )
    expect(page.locator(GHOST_TEXT)).to_contain_text("zer", timeout=5000)

    assert any(r.get("suffix") for r in recorder), (
        "expected at least one FIM request with non-empty suffix"
    )


# ── confidence gate ───────────────────────────────────────────────────


def test_low_confidence_suggestion_is_suppressed(page, editor_url, mock_complete):
    """Below the confidence threshold (0.2) the ghost is not shown."""
    mock_complete(page, [{"text": " zerbait", "confidence": 0.10}])
    page.goto(editor_url)

    # Give the request time to resolve, then assert no ghost rendered.
    expect(page.locator(STATUS)).to_contain_text("Suppressed", timeout=5000)
    expect(page.locator(GHOST_TEXT)).to_have_count(0)


def test_empty_response_shows_no_suggestion(page, editor_url, mock_complete):
    """When the model returns empty text, the editor shows 'No suggestion'
    rather than a ghost. (Documents that 'no suggestion' = empty model
    response, not a client bug.)"""
    mock_complete(page, [{"text": "", "confidence": 0}])
    page.goto(editor_url)

    expect(page.locator(STATUS)).to_contain_text("No suggestion", timeout=5000)
    expect(page.locator(GHOST_TEXT)).to_have_count(0)


# ── alternative cycling (Alt+] / Alt+[) — Copilot convention ──────────


def test_alt_bracket_cycles_alternatives(page, editor_url, mock_complete):
    """Alt+] fetches a new alternative at elevated temperature; Alt+[
    walks back through history. (RED until cycling is implemented.)"""
    recorder = []
    mock_complete(page, [
        {"text": " zer moduz", "confidence": 0.50},
        {"text": " beste bat", "confidence": 0.40},
        {"text": " hirugarrena", "confidence": 0.30},
    ], recorder=recorder)
    page.goto(editor_url)
    expect(page.locator(GHOST_TEXT)).to_contain_text("zer moduz", timeout=5000)

    # First Alt+] → second alternative, fetched at elevated temperature.
    page.locator(EDITOR).press("Alt+BracketRight")
    expect(page.locator(GHOST_TEXT)).to_contain_text("beste bat", timeout=5000)
    expect(page.locator(STATS)).to_contain_text("2/2")

    # Second Alt+] → third alternative.
    page.locator(EDITOR).press("Alt+BracketRight")
    expect(page.locator(GHOST_TEXT)).to_contain_text("hirugarrena", timeout=5000)
    expect(page.locator(STATS)).to_contain_text("3/3")

    # Alt+[ → walk back to the second alternative (no new request).
    page.locator(EDITOR).press("Alt+BracketLeft")
    expect(page.locator(GHOST_TEXT)).to_contain_text("beste bat", timeout=3000)
    expect(page.locator(STATS)).to_contain_text("2/3")

    # The auto request used baseline temp; cycle requests used elevated temp.
    assert recorder[0]["temperature"] == 0.2
    assert recorder[1]["temperature"] == 0.7
    assert recorder[2]["temperature"] == 0.7


# ── regression: the original Alt+] bug ────────────────────────────────


def test_alt_bracket_empty_cycle_does_not_wipe_suggestion(page, editor_url, mock_complete):
    """REGRESSION: The original cycling implementation wiped currentSuggestion
    when a cycle request returned empty text (common at elevated temperature).
    This caused the suggestion to vanish intermittently — the exact regression
    the user reported. The fix: on empty cycle response, keep the current
    suggestion and show 'No alternative found'."""
    mock_complete(page, [
        {"text": " zer moduz", "confidence": 0.50},   # initial auto-request
        {"text": "", "confidence": 0},                  # Alt+] returns empty
    ])
    page.goto(editor_url)
    expect(page.locator(GHOST_TEXT)).to_contain_text("zer moduz", timeout=5000)

    # Alt+] returns an empty response.
    page.locator(EDITOR).press("Alt+BracketRight")

    # The original suggestion must STILL be visible (not wiped).
    expect(page.locator(GHOST_TEXT)).to_contain_text("zer moduz", timeout=5000)
    # And the status should report that no alternative was found.
    expect(page.locator(STATUS)).to_contain_text("No alternative", timeout=5000)
