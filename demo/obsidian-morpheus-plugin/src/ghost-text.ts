/*
 * Morpheus ghost-text extension for Obsidian (CodeMirror 6).
 *
 * Renders inline "ghost text" suggestions at the cursor, like GitHub Copilot
 * or Google Smart Compose. On typing pause, sends the text before/after the
 * cursor to the Morpheus demo server's /v1/complete endpoint (which handles
 * FIM templating, tokenization, and cleanup per-backend). Tab accepts, Esc
 * dismisses.
 *
 * Telemetry: every suggestion lifecycle event (suggested, accepted, rejected,
 * ignored) is tracked via the TelemetryService for cross-model analysis.
 * See telemetry.ts.
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

// ── State management ───────────────────────────────────────────────────

/** Carries a new suggestion (or null to clear) + the doc snapshot it was
 *  computed against, so stale responses (doc changed since request) are
 *  discarded by the StateField. Also carries telemetry metadata
 *  (suggestionId, confidence, model) so the keymap can log accept/reject. */
const InlineSuggestionEffect = StateEffect.define<{
  text: string | null;
  doc: Text;
  suggestionId: string;
  confidence: number;
  model: string;
}>();

const ClearSuggestionEffect = StateEffect.define<null>();

interface SuggestionState {
  suggestion: string | null;
  suggestionId: string | null;
  confidence: number;
  model: string;
}

const InlineSuggestionState = StateField.define<SuggestionState>({
  create() {
    return { suggestion: null, suggestionId: null, confidence: 0, model: "" };
  },
  update(value, tr) {
    // Explicit clear
    for (const effect of tr.effects) {
      if (effect.is(ClearSuggestionEffect)) {
        return { suggestion: null, suggestionId: null, confidence: 0, model: "" };
      }
    }
    // New suggestion arrived — only accept if doc hasn't changed since request
    for (const effect of tr.effects) {
      if (effect.is(InlineSuggestionEffect)) {
        if (tr.state.doc === effect.value.doc) {
          return {
            suggestion: effect.value.text,
            suggestionId: effect.value.suggestionId,
            confidence: effect.value.confidence,
            model: effect.value.model,
          };
        }
      }
    }
    // Any doc change or cursor move clears the suggestion
    if (tr.docChanged || tr.selection) {
      return { suggestion: null, suggestionId: null, confidence: 0, model: "" };
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

  /** Let CodeMirror know how many line breaks the widget spans so it can
   *  calculate viewport height correctly. */
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
        side: 1, // render after the cursor
      });
      this.decorations = Decoration.set([widget.range(pos)]);
    }
  },
  { decorations: (v) => v.decorations }
);

// ── Fetch plugin (triggers completion on typing pause) ─────────────────

function createFetchPlugin(plugin: MorpheusPlugin) {
  return ViewPlugin.fromClass(
    class {
      private timer: ReturnType<typeof setTimeout> | null = null;
      /** Monotonic counter — each new keystroke increments it. When a
       *  response arrives, we check if it's still the latest request; if
       *  not, discard it (requestUrl doesn't support AbortController). */
      private generation = 0;

      update(update: ViewUpdate) {
        // Mark any outstanding suggestion as ignored on any change
        if (update.docChanged || update.selectionSet) {
          plugin.telemetry?.markIgnoredIfOutstanding();
        }

        if (!update.docChanged) return;
        if (!plugin.settings.enabled) return;

        // Don't trigger when there's an active selection
        if (!update.state.selection.main.empty) return;

        if (this.timer) clearTimeout(this.timer);
        this.generation++;
        const currentGen = this.generation;

        const delay = plugin.settings.triggerDelay;

        this.timer = setTimeout(() => {
          void (async () => {
            const settings = plugin.settings;
            // Snapshot the doc at request time — the StateField will
            // discard the response if the doc has since changed.
            const doc = update.state.doc;
            const cursor = update.state.selection.main.head;
            const fullText = doc.toString();

            const prefix = fullText.slice(
              Math.max(0, cursor - settings.contextBefore),
              cursor
            );
            const suffix = fullText.slice(
              cursor,
              cursor + settings.contextAfter
            );

            // Don't trigger on very short prefixes
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
                  temperature: settings.temperature,
                  n: settings.bestOfN,
                }),
              });

              // Stale check: a newer request was fired while we were waiting
              if (currentGen !== this.generation) return;
              // Stale check: doc changed since the request
              if (update.view.state.doc !== doc) return;

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
                  prefix.length
                );

                update.view.dispatch({
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
              // Silently ignore — don't spam errors on every keystroke.
              // The status bar (in main.ts) surfaces connection issues.
              if (e instanceof Error) {
                console.debug("Morpheus: fetch error", e.message);
              }
            }
          })();
        }, delay);
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
  // Keymap is created inside the factory so it can close over `plugin`
  // for settings access (serverUrl) and telemetry.
  const ghostTextKeymap = Prec.highest(
    keymap.of([
      {
        key: "Tab",
        run: (view: EditorView) => {
          const state = view.state.field(InlineSuggestionState);
          if (!state?.suggestion) return false; // let default Tab (indent) run

          const head = view.state.selection.main.head;
          view.dispatch({
            changes: { from: head, to: head, insert: state.suggestion },
            selection: { anchor: head + state.suggestion.length },
            userEvent: "input.complete",
          });

          // Telemetry: track acceptance
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
            state.model
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
