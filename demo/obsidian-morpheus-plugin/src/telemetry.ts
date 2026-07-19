/*
 * Morpheus Telemetry Service.
 *
 * Collects autocomplete interaction events (suggested, accepted, rejected,
 * ignored) and sends them in batches to a configurable telemetry endpoint.
 * Inspired by GitHub Copilot's telemetry: tracks acceptance rates, latency,
 * and confidence to measure how useful each model's suggestions are.
 *
 * Privacy: opt-in (disabled by default). The endpoint URL is configurable
 * so users can self-host. Suggestion/context text is sent only when the
 * user explicitly enables it.
 *
 * Events are buffered and flushed every 5 seconds (or when 20 events
 * accumulate). Best-effort: if the telemetry server is down, events are
 * silently dropped.
 */

import { requestUrl } from "obsidian";
import type { MorpheusSettings } from "./main";

export type TelemetryEventType =
  | "suggested"
  | "accepted"
  | "partially_accepted"
  | "rejected"
  | "ignored";

interface TelemetryEventData {
  model: string;
  event_type: TelemetryEventType;
  suggestion_id: string;
  latency_ms?: number;
  confidence?: number;
  suggestion_length?: number;
  prompt_length?: number;
  suggestion_text?: string;
  context?: string;
  /** Cumulative accepted length (UTF-16 codepoints), Copilot convention.
   *  Set on partially_accepted events. */
  accepted_length?: number;
  /** Why the suggestion was rejected: "dismissed" (Esc), "cycled" (Alt+]),
   *  "cycled_back" (Alt+[). Only set on rejected events. */
  reject_reason?: string;
}

interface TelemetryEvent extends TelemetryEventData {
  timestamp: string;
  session_id: string;
}

export class TelemetryService {
  private buffer: TelemetryEvent[] = [];
  private flushTimer: number | null = null;
  private readonly sessionId: string;
  /** The currently-shown suggestion, tracked so we can fire "ignored"
   *  when the user types past it without accepting or rejecting. */
  private outstanding: { suggestionId: string; model: string } | null = null;

  constructor(private getSettings: () => MorpheusSettings) {
    this.sessionId = uuid();
    this.flushTimer = window.setInterval(() => this.flush(), 5_000);
  }

  /** Generate a unique ID for a new suggestion. */
  newSuggestionId(): string {
    return uuid();
  }

  /** Called when a ghost-text suggestion is shown to the user. */
  trackSuggested(
    model: string,
    suggestionId: string,
    latencyMs: number,
    confidence: number,
    suggestionText: string,
    promptLength: number
  ): void {
    // Any outstanding suggestion that wasn't accepted/rejected was ignored
    this.markIgnoredIfOutstanding();

    this.outstanding = { suggestionId, model };

    const settings = this.getSettings();
    this.enqueue({
      event_type: "suggested",
      model,
      suggestion_id: suggestionId,
      latency_ms: Math.round(latencyMs * 10) / 10,
      confidence: Math.round(confidence * 1000) / 1000,
      suggestion_length: suggestionText.length,
      prompt_length: promptLength,
      suggestion_text: settings.telemetryIncludeText
        ? suggestionText.slice(0, 500)
        : undefined,
    });
  }

  /** Called when the user presses Tab to accept a suggestion. */
  trackAccepted(
    suggestionId: string,
    model: string,
    suggestionText: string,
    context: string
  ): void {
    this.outstanding = null;
    const settings = this.getSettings();
    this.enqueue({
      event_type: "accepted",
      model,
      suggestion_id: suggestionId,
      suggestion_text: settings.telemetryIncludeText
        ? suggestionText.slice(0, 500)
        : undefined,
      context: settings.telemetryIncludeText
        ? context.slice(0, 500)
        : undefined,
    });
  }

  /** Called when the user accepts part of a suggestion (Ctrl+Right).
   *  Follows the Copilot convention: acceptedLength is CUMULATIVE — it
   *  includes everything accepted so far, not just the latest word. */
  trackPartiallyAccepted(
    suggestionId: string,
    model: string,
    acceptedLength: number,
    totalLength: number
  ): void {
    // Partial acceptance does NOT clear `outstanding` — the suggestion
    // is still showing (the remainder). It will be cleared by a final
    // `accepted` (Tab) or `rejected`/`ignored` later.
    this.enqueue({
      event_type: "partially_accepted",
      model,
      suggestion_id: suggestionId,
      accepted_length: acceptedLength,
      suggestion_length: totalLength,
    });
  }

  /** Called when the user presses Esc to dismiss a suggestion, or cycles
   *  to another alternative (Alt+]/[). */
  trackRejected(
    suggestionId: string,
    model: string,
    reason: "dismissed" | "cycled" | "cycled_back" = "dismissed"
  ): void {
    this.outstanding = null;
    this.enqueue({
      event_type: "rejected",
      model,
      suggestion_id: suggestionId,
      reject_reason: reason,
    });
  }

  /** Called when the user types or moves the cursor, causing an outstanding
   *  suggestion to be cleared without being accepted or rejected. */
  markIgnoredIfOutstanding(): void {
    if (!this.outstanding) return;
    const { suggestionId, model } = this.outstanding;
    this.outstanding = null;
    this.enqueue({
      event_type: "ignored",
      model,
      suggestion_id: suggestionId,
    });
  }

  private enqueue(event: TelemetryEventData): void {
    const settings = this.getSettings();
    if (!settings.telemetryEnabled || !settings.telemetryEndpoint) return;

    this.buffer.push({
      ...event,
      timestamp: new Date().toISOString(),
      session_id: this.sessionId,
    });

    if (this.buffer.length >= 20) {
      void this.flush();
    }
  }

  async flush(): Promise<void> {
    if (this.buffer.length === 0) return;
    const settings = this.getSettings();
    if (!settings.telemetryEnabled || !settings.telemetryEndpoint) {
      this.buffer = [];
      return;
    }

    const events = this.buffer;
    this.buffer = [];

    try {
      await requestUrl({
        url: `${settings.telemetryEndpoint.replace(/\/$/, "")}/api/events`,
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ events }),
      });
    } catch (e) {
      // Drop on failure — don't retry to avoid unbounded buffer growth.
      // For research telemetry, losing a few events is acceptable.
      console.debug("Morpheus: telemetry flush failed", e);
    }
  }

  destroy(): void {
    if (this.flushTimer) window.clearInterval(this.flushTimer);
    this.flushTimer = null;
    void this.flush();
  }
}

function uuid(): string {
  if (typeof crypto !== "undefined" && crypto.randomUUID) {
    return crypto.randomUUID();
  }
  return `id-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
}
