/*
 * Morpheus ghost-text extension for Obsidian (CodeMirror 6).
 *
 * Renders inline "ghost text" suggestions at the cursor, like GitHub Copilot
 * or Google Smart Compose. On typing pause, sends the text before/after the
 * cursor to the Morpheus demo server's /v1/complete endpoint (which handles
 * FIM templating, tokenization, and cleanup per-backend).
 *
 * Keybindings:
 *   Tab              Accept full suggestion
 *   Ctrl+Right       Accept next word (partial acceptance, Copilot convention)
 *   Alt+]            Cycle to next alternative (re-fetch at elevated temp)
 *   Alt+[            Cycle to previous alternative (from history)
 *   Escape           Dismiss suggestion
 *
 * Telemetry: every suggestion lifecycle event (suggested, accepted,
 * partially_accepted, rejected, ignored) is tracked via TelemetryService.
 * Partial acceptance sends accepted_length (cumulative, Copilot's
 * didPartiallyAcceptCompletion convention). Cycling sends rejected
 * (reason: cycled/cycled_back) for the displaced suggestion.
 *
 * Adapted from the CodeMirror 6 inline-suggestion pattern
 * (cf. Leoyishou/obsidian-ai-autocomplete), but targeting the Morpheus
 * demo server's {prefix, suffix} -> {text, confidence} convenience route
 * instead of an OpenAI chat API.
 */

import {
  ViewPlugin,
  ViewUpdate,
  EditorView,
  Decoration,
  DecorationSet,
  WidgetType,
  keymap,
} from "@codemirror/view";
import {
  StateEffect,
  StateField,
  Text,
  Prec,
} from "@codemirror/state";
import { requestUrl } from "obsidian";
import type MorpheusPlugin from "./main";

// ── State effects ──────────────────────────────────────────────────────

/** New suggestion from a fetch (has doc-stale check in the StateField). */
const InlineSuggestionEffect = StateEffect.define<{
  text: string | null;
  doc: Text;
  suggestionId: string;
  confidence: number;
  model: string;
}>();

/** Set suggestion directly (NO doc-stale check). Used by partial acceptance
 *  (to update the remainder after inserting a word) and cycle-prev (to
 *  restore a suggestion from history). Bypasses the stale-check because
 *  the doc HAS changed in these cases, but the suggestion is still valid. */
const SetSuggestionEffect = StateEffect.define<{
  text: string | null;
  suggestionId: string;
  confidence: number;
  model: string;
  acceptedLength: number;
  totalLength: number;
}>();

const ClearSuggestionEffect = StateEffect.define<null>();

/** Keymap / command → fetch plugin: trigger Alt+] / Alt+[ / Ctrl+Right.
 *  Exported so main.ts can dispatch them from Obsidian commands (which
 *  take priority over CodeMirror keymaps and are configurable via
 *  Settings → Hotkeys). */
export const CycleNextEffect = StateEffect.define<null>();
export const CyclePrevEffect = StateEffect.define<null>();
export const AcceptNextWordEffect = StateEffect.define<null>();

// ── State ──────────────────────────────────────────────────────────────

interface SuggestionState {
  suggestion: string | null;
  suggestionId: string | null;
  confidence: number;
  model: string;
  /** Cumulative accepted length (UTF-16 codepoints). Reset to 0 when a
   *  new suggestion arrives. Incremented on each Ctrl+Right. */
  acceptedLength: number;
  /** Original suggestion length (for telemetry: fraction accepted). */
  totalLength: number;
}

const EMPTY_STATE: SuggestionState = {
  suggestion: null,
  suggestionId: null,
  confidence: 0,
  model: "",
  acceptedLength: 0,
  totalLength: 0,
};

const InlineSuggestionState = StateField.define<SuggestionState>({
  create() {
    return EMPTY_STATE;
  },
  update(value, tr) {
    // 1. Explicit clear
    for (const effect of tr.effects) {
      if (effect.is(ClearSuggestionEffect)) {
        return EMPTY_STATE;
      }
    }
    // 2. SetSuggestionEffect (no doc-stale check — partial accept + cycle-prev)
    for (const effect of tr.effects) {
      if (effect.is(SetSuggestionEffect)) {
        const v = effect.value;
        if (v.text === null) return EMPTY_STATE;
        return {
          suggestion: v.text,
          suggestionId: v.suggestionId,
          confidence: v.confidence,
          model: v.model,
          acceptedLength: v.acceptedLength,
          totalLength: v.totalLength,
        };
      }
    }
    // 3. InlineSuggestionEffect (with doc-stale check — new fetches)
    for (const effect of tr.effects) {
      if (effect.is(InlineSuggestionEffect)) {
        if (tr.state.doc === effect.value.doc) {
          return {
            suggestion: effect.value.text,
            suggestionId: effect.value.suggestionId,
            confidence: effect.value.confidence,
            model: effect.value.model,
            acceptedLength: 0,
            totalLength: effect.value.text?.length ?? 0,
          };
        }
      }
    }
    // 4. Any doc change or cursor move clears the suggestion
    if (tr.docChanged || tr.selection) {
      return EMPTY_STATE;
    }
    return value;
  },
});

// ── Ghost text widget ──────────────────────────────────────────────────

class GhostTextWidget extends WidgetType {
  constructor(readonly text: string) {
    super();
  }

  eq(other: GhostTextWidget) {
    return other.text === this.text;
  }

  toDOM() {
    const span = document.createElement("span");
    span.className = "morpheus-ghost-text";
    span.textContent = this.text;
    return span;
  }

  get lineBreaks() {
    return this.text.split("\n").length - 1;
  }
}

// ── Render plugin (decoration -> ghost text at cursor) ─────────────────

const renderGhostTextPlugin = ViewPlugin.fromClass(
  class {
    decorations: DecorationSet = Decoration.none;

    update(update: ViewUpdate) {
      const suggestion = update.state.field(InlineSuggestionState)?.suggestion;

      if (!suggestion) {
        this.decorations = Decoration.none;
        return;
      }

      const pos = update.state.selection.main.head;
      const widget = Decoration.widget({
        widget: new GhostTextWidget(suggestion),
        side: 1,
      });
      this.decorations = Decoration.set([widget.range(pos)]);
    }
  },
  { decorations: (v) => v.decorations }
);

// ── Fetch plugin (triggers completion + handles cycling/partial-accept) ─

interface HistoryEntry {
  text: string;
  suggestionId: string;
  confidence: number;
  model: string;
}

function createFetchPlugin(plugin: MorpheusPlugin) {
  return ViewPlugin.fromClass(
    class {
      private timer: ReturnType<typeof setTimeout> | null = null;
      private generation = 0;
      /** Cycling history: all alternatives seen for the current context. */
      private history: HistoryEntry[] = [];
      private historyIndex = -1;

      update(update: ViewUpdate) {
        // ── Handle explicit action effects (from keymap) ──
        for (const tr of update.transactions) {
          for (const effect of tr.effects) {
            if (effect.is(CycleNextEffect)) {
              // Defer: can't call view.dispatch() from within update().
              // queueMicrotask runs after the current update cycle.
              queueMicrotask(() => this.cycleNext(update.view));
              return;
            }
            if (effect.is(CyclePrevEffect)) {
              queueMicrotask(() => this.cyclePrev(update.view));
              return;
            }
            if (effect.is(AcceptNextWordEffect)) {
              queueMicrotask(() => this.acceptNextWord(update.view));
              return;
            }
          }
        }

        // ── Check if this update is from partial acceptance ──
        // (Ctrl+Right inserts text but we keep the remainder — don't
        // trigger a new fetch or mark the suggestion as ignored)
        const isPartialAccept = update.transactions.some((tr) =>
          tr.isUserEvent("input.complete.partial")
        );

        if ((update.docChanged || update.selectionSet) && !isPartialAccept) {
          plugin.telemetry?.markIgnoredIfOutstanding();
          // Context changed — reset cycling history
          this.history = [];
          this.historyIndex = -1;
        }

        if (!update.docChanged) return;
        if (isPartialAccept) return; // keep remainder, don't re-fetch
        if (!plugin.settings.enabled) return;
        if (!update.state.selection.main.empty) return;

        if (this.timer) clearTimeout(this.timer);
        this.generation++;
        const currentGen = this.generation;
        const delay = plugin.settings.triggerDelay;

        this.timer = setTimeout(() => {
          void this.fetchCompletion(update.view, update.state, currentGen, false);
        }, delay);
      }

      /** Fetch a completion from the server. Shared by normal-typing trigger
       *  and Alt+] cycling. `isCycle=true` uses elevated temperature. */
      private async fetchCompletion(
        view: EditorView,
        requestState: { doc: Text; selection: { main: { head: number } } },
        expectedGen: number,
        isCycle: boolean
      ): Promise<void> {
        const settings = plugin.settings;
        const doc = requestState.doc;
        const cursor = requestState.selection.main.head;
        const fullText = doc.toString();

        const prefix = fullText.slice(
          Math.max(0, cursor - settings.contextBefore),
          cursor
        );
        const suffix = fullText.slice(cursor, cursor + settings.contextAfter);

        if (prefix.trim().length < 3) return;

        const reqStart = performance.now();

        try {
          const res = await requestUrl({
            url: `${settings.serverUrl.replace(/\/$/, "")}/v1/complete`,
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              prefix,
              suffix,
              max_tokens: settings.maxTokens,
              temperature: isCycle
                ? settings.cycleTemperature
                : settings.temperature,
              n: settings.bestOfN,
            }),
          });

          // Stale checks
          if (expectedGen !== this.generation) return;
          if (view.state.doc !== doc) return;

          const data = res.json;
          const text: string = data?.text ?? "";
          const confidence: number = data?.confidence ?? 0;

          if (text.trim() && confidence >= settings.confidenceThreshold) {
            const latency = performance.now() - reqStart;
            const suggestionId = plugin.telemetry?.newSuggestionId() ?? "";
            const model = plugin.currentModel ?? "unknown";

            plugin.telemetry?.trackSuggested(
              model,
              suggestionId,
              latency,
              confidence,
              text,
              prefix.length,
              prefix
            );

            // Push to cycling history
            this.history.push({ text, suggestionId, confidence, model });
            this.historyIndex = this.history.length - 1;

            view.dispatch({
              effects: InlineSuggestionEffect.of({
                text,
                doc,
                suggestionId,
                confidence,
                model,
              }),
            });
          }
        } catch (e) {
          if (e instanceof Error) {
            console.debug("Morpheus: fetch error", e.message);
          }
        }
      }

      /** Alt+]: reject current, fetch new alternative at elevated temp. */
      private cycleNext(view: EditorView): void {
        if (!plugin.settings.enabled) return;

        const state = view.state.field(InlineSuggestionState);

        // Reject current suggestion (if any) — user wants a different one
        if (state.suggestion && state.suggestionId) {
          plugin.telemetry?.trackRejected(
            state.suggestionId,
            state.model,
            "cycled"
          );
        }

        // Clear current ghost text immediately
        view.dispatch({ effects: ClearSuggestionEffect.of(null) });

        // Fetch a new alternative
        this.generation++;
        const currentGen = this.generation;
        const snapshot = {
          doc: view.state.doc,
          selection: { main: { head: view.state.selection.main.head } },
        };
        void this.fetchCompletion(view, snapshot, currentGen, true);
      }

      /** Alt+[: walk back through cycling history. */
      private cyclePrev(view: EditorView): void {
        if (this.historyIndex <= 0) return;

        const state = view.state.field(InlineSuggestionState);

        // Reject current (user wants the previous one)
        if (state.suggestion && state.suggestionId) {
          plugin.telemetry?.trackRejected(
            state.suggestionId,
            state.model,
            "cycled_back"
          );
        }

        this.historyIndex--;
        const entry = this.history[this.historyIndex];

        // Restore via SetSuggestionEffect (bypasses doc-stale check).
        // Do NOT fire trackSuggested — this suggestion was already counted
        // when it was first fetched (avoids inflating the denominator).
        view.dispatch({
          effects: SetSuggestionEffect.of({
            text: entry.text,
            suggestionId: entry.suggestionId,
            confidence: entry.confidence,
            model: entry.model,
            acceptedLength: 0, // reset: fresh show of full suggestion
            totalLength: entry.text.length,
          }),
        });
      }

      /** Ctrl+Right: accept next word from the current suggestion.
       *  Follows the VS Code / Copilot "accept next word" convention.
       *  Fires partially_accepted telemetry with cumulative acceptedLength. */
      private acceptNextWord(view: EditorView): void {
        const state = view.state.field(InlineSuggestionState);
        if (!state.suggestion) return;

        // Match leading whitespace + a word + trailing whitespace
        const m = state.suggestion.match(/^\s*\S+\s*/);
        if (!m) return;

        const word = m[0];
        const rest = state.suggestion.slice(word.length);
        const head = view.state.selection.main.head;

        if (rest) {
          // ── Partial acceptance: insert word, keep remainder as ghost ──
          const newAcceptedLength = state.acceptedLength + word.length;

          view.dispatch({
            changes: { from: head, to: head, insert: word },
            selection: { anchor: head + word.length },
            effects: SetSuggestionEffect.of({
              text: rest,
              suggestionId: state.suggestionId!,
              confidence: state.confidence,
              model: state.model,
              acceptedLength: newAcceptedLength,
              totalLength: state.totalLength,
            }),
            userEvent: "input.complete.partial",
          });

          // Telemetry: partially_accepted with cumulative length (Copilot
          // convention — acceptedLength is total so far, not just this word)
          plugin.telemetry?.trackPartiallyAccepted(
            state.suggestionId!,
            state.model,
            newAcceptedLength,
            state.totalLength
          );
        } else {
          // ── Last word consumed: full acceptance via word-by-word ──
          view.dispatch({
            changes: { from: head, to: head, insert: word },
            selection: { anchor: head + word.length },
            effects: ClearSuggestionEffect.of(null),
            userEvent: "input.complete",
          });

          const fullText = view.state.doc.toString();
          plugin.telemetry?.trackAccepted(
            state.suggestionId ?? "",
            state.model,
            state.suggestion,
            fullText.slice(Math.max(0, head - 200), head)
          );

          // Reset cycling history (suggestion fully consumed)
          this.history = [];
          this.historyIndex = -1;
        }
      }

      destroy() {
        if (this.timer) clearTimeout(this.timer);
        plugin.telemetry?.markIgnoredIfOutstanding();
      }
    }
  );
}

// ── Public API ─────────────────────────────────────────────────────────

export function createMorpheusExtension(plugin: MorpheusPlugin) {
  const ghostTextKeymap = Prec.highest(
    keymap.of([
      {
        key: "Tab",
        run: (view: EditorView) => {
          const state = view.state.field(InlineSuggestionState);
          if (!state?.suggestion) return false;

          const head = view.state.selection.main.head;
          view.dispatch({
            changes: { from: head, to: head, insert: state.suggestion },
            selection: { anchor: head + state.suggestion.length },
            userEvent: "input.complete",
          });

          const fullText = view.state.doc.toString();
          plugin.telemetry?.trackAccepted(
            state.suggestionId ?? "",
            state.model,
            state.suggestion,
            fullText.slice(Math.max(0, head - 200), head)
          );

          return true;
        },
      },
      {
        key: "Escape",
        run: (view: EditorView) => {
          const state = view.state.field(InlineSuggestionState);
          if (!state?.suggestion) return false;
          plugin.telemetry?.trackRejected(
            state.suggestionId ?? "",
            state.model,
            "dismissed"
          );
          view.dispatch({ effects: ClearSuggestionEffect.of(null) });
          return true;
        },
      },
    ])
  );

  return [
    InlineSuggestionState,
    createFetchPlugin(plugin),
    renderGhostTextPlugin,
    ghostTextKeymap,
  ];
}

// ── Helpers for Obsidian commands ───────────────────────────────────────

/** Check whether a ghost-text suggestion is currently showing.
 *  Used by Obsidian commands (registered in main.ts) to decide whether
 *  to handle the key or let it fall through to the default behavior. */
export function hasSuggestion(view: EditorView): boolean {
  const state = view.state.field(InlineSuggestionState);
  return !!state?.suggestion;
}
