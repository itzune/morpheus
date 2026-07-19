"""SQLite database for Morpheus telemetry events.

Uses WAL mode for concurrent read/write (dashboard reads while plugin writes).
Thread-safe via a global lock around writes (SQLite serializes writes anyway).
"""
import sqlite3
import threading
from pathlib import Path
from contextlib import contextmanager
from typing import Any

_DB_PATH: Path = Path(__file__).resolve().parent / "telemetry.db"
_lock = threading.Lock()

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    session_id TEXT NOT NULL,
    model TEXT NOT NULL,
    event_type TEXT NOT NULL,
    suggestion_id TEXT,
    latency_ms REAL,
    confidence REAL,
    suggestion_length INTEGER,
    prompt_length INTEGER,
    suggestion_text TEXT,
    context TEXT,
    accepted_length INTEGER,
    reject_reason TEXT
);
CREATE INDEX IF NOT EXISTS idx_events_model ON events(model);
CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type);
CREATE INDEX IF NOT EXISTS idx_events_ts ON events(timestamp);
CREATE INDEX IF NOT EXISTS idx_events_sid ON events(suggestion_id);
"""

# Columns added after initial release. Each is added via ALTER TABLE if missing.
_MIGRATIONS = [
    ("accepted_length", "ALTER TABLE events ADD COLUMN accepted_length INTEGER"),
    ("reject_reason", "ALTER TABLE events ADD COLUMN reject_reason TEXT"),
]


def _migrate(conn) -> None:
    """Add columns that were introduced after the initial schema."""
    existing = {
        row[1]
        for row in conn.execute("PRAGMA table_info(events)").fetchall()
    }
    for col, sql in _MIGRATIONS:
        if col not in existing:
            conn.execute(sql)


def init_db(db_path: str | None = None) -> None:
    """Create the database file and tables if they don't exist."""
    global _DB_PATH
    if db_path:
        _DB_PATH = Path(db_path)
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _get_conn() as conn:
        conn.executescript(SCHEMA)
        _migrate(conn)


@contextmanager
def _get_conn():
    conn = sqlite3.connect(str(_DB_PATH), timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def _normalize_ts(ts: str) -> str:
    """Convert ISO 8601 (2024-01-01T12:00:00.000Z) to SQLite format
    (2024-01-01 12:00:00) so datetime() comparisons work correctly."""
    # 2024-01-01T12:00:00.000Z -> 2024-01-01 12:00:00
    return ts.replace("T", " ").split(".")[0]


def insert_events(events: list[dict[str, Any]]) -> int:
    """Insert a batch of telemetry events. Thread-safe."""
    if not events:
        return 0
    # Normalize timestamps to SQLite format
    for e in events:
        if "timestamp" in e and e["timestamp"]:
            e["timestamp"] = _normalize_ts(e["timestamp"])
    with _lock:
        with _get_conn() as conn:
            conn.executemany(
                """INSERT INTO events
                   (timestamp, session_id, model, event_type, suggestion_id,
                    latency_ms, confidence, suggestion_length, prompt_length,
                    suggestion_text, context, accepted_length, reject_reason)
                   VALUES
                   (:timestamp, :session_id, :model, :event_type, :suggestion_id,
                    :latency_ms, :confidence, :suggestion_length, :prompt_length,
                    :suggestion_text, :context, :accepted_length, :reject_reason)""",
                events,
            )
            return len(events)


def get_events(model: str | None = None, limit: int = 100) -> list[dict]:
    """Query raw events (most recent first)."""
    with _get_conn() as conn:
        if model:
            rows = conn.execute(
                "SELECT * FROM events WHERE model = ? ORDER BY id DESC LIMIT ?",
                (model, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM events ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]


def get_models() -> list[str]:
    """Distinct model names seen in the telemetry data."""
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT DISTINCT model FROM events ORDER BY model"
        ).fetchall()
        return [r["model"] for r in rows]


def get_total_count() -> int:
    with _get_conn() as conn:
        row = conn.execute("SELECT COUNT(*) as count FROM events").fetchone()
        return row["count"]


def get_suggestion_groups(
    hours: int = 24, model: str | None = None, limit: int = 200
) -> list[dict]:
    """Group events by suggestion_id, returning the lifecycle of each
    suggestion. Each group has the suggested event (latency, confidence,
    text, context) plus the outcome (accepted / partially_accepted /
    rejected / ignored).
    """
    cutoff = f"-{hours} hours"
    with _get_conn() as conn:
        if model:
            rows = conn.execute(
                """SELECT * FROM events
                   WHERE model = ? AND timestamp >= datetime('now', ?)
                   ORDER BY id DESC""",
                (model, cutoff),
            ).fetchall()
        else:
            rows = conn.execute(
                """SELECT * FROM events
                   WHERE timestamp >= datetime('now', ?)
                   ORDER BY id DESC""",
                (cutoff,),
            ).fetchall()

    # Group by suggestion_id, preserving most-recent-first order
    groups: dict[str, list[dict]] = {}
    order: list[str] = []
    for r in rows:
        sid = r["suggestion_id"]
        if not sid:
            continue
        if sid not in groups:
            groups[sid] = []
            order.append(sid)
        groups[sid].append(dict(r))

    result = []
    for sid in order:
        events = groups[sid]
        # The suggested event carries latency, confidence, text, context
        suggested = next(
            (e for e in events if e["event_type"] == "suggested"), None
        )
        if not suggested:
            continue  # orphan events without a preceding suggested

        # Determine outcome from the non-suggested events
        outcome = "ignored"  # default: only suggested, no interaction
        reject_reason = None
        accepted_length = None
        for e in events:
            et = e["event_type"]
            if et == "accepted":
                outcome = "accepted"
            elif et == "partially_accepted":
                outcome = "partially_accepted"
                accepted_length = e.get("accepted_length")
            elif et == "rejected":
                outcome = "rejected"
                reject_reason = e.get("reject_reason")
            elif et == "ignored":
                outcome = "ignored"

        # Context: prefer the suggested event's context; fall back to
        # the accepted event's context (older plugin versions only sent
        # context on accept).
        context = suggested.get("context")
        if not context:
            accepted_ev = next(
                (e for e in events if e["event_type"] == "accepted"), None
            )
            if accepted_ev:
                context = accepted_ev.get("context")

        result.append(
            {
                "suggestion_id": sid,
                "model": suggested["model"],
                "timestamp": suggested["timestamp"],
                "suggestion_text": suggested.get("suggestion_text"),
                "context": context,
                "latency_ms": suggested.get("latency_ms"),
                "confidence": suggested.get("confidence"),
                "prompt_length": suggested.get("prompt_length"),
                "suggestion_length": suggested.get("suggestion_length"),
                "outcome": outcome,
                "reject_reason": reject_reason,
                "accepted_length": accepted_length,
                "event_count": len(events),
            }
        )
        if len(result) >= limit:
            break

    return result


def get_stats(hours: int = 24) -> dict:
    """Aggregated stats for the dashboard: per-model breakdown + timeline."""
    cutoff = f"-{hours} hours"
    # Use finer time buckets for short ranges
    if hours <= 1:
        bucket_sql = "strftime('%Y-%m-%d %H:%M:00', timestamp)"
    else:
        bucket_sql = "strftime('%Y-%m-%d %H:00:00', timestamp)"

    with _get_conn() as conn:
        # Per-model aggregate stats
        rows = conn.execute(
            """SELECT
                model,
                SUM(CASE WHEN event_type = 'suggested'  THEN 1 ELSE 0 END) as suggested,
                SUM(CASE WHEN event_type = 'accepted'   THEN 1 ELSE 0 END) as accepted,
                SUM(CASE WHEN event_type = 'rejected'   THEN 1 ELSE 0 END) as rejected,
                SUM(CASE WHEN event_type = 'ignored'    THEN 1 ELSE 0 END) as ignored,
                SUM(CASE WHEN event_type = 'partially_accepted' THEN 1 ELSE 0 END) as partial_acceptances,
                AVG(CASE WHEN event_type = 'suggested'  THEN latency_ms END) as avg_latency_ms,
                AVG(CASE WHEN event_type = 'suggested'  THEN confidence END) as avg_confidence
               FROM events
               WHERE timestamp >= datetime('now', ?)
               GROUP BY model
               ORDER BY model""",
            (cutoff,),
        ).fetchall()

        models = []
        for r in rows:
            d = dict(r)
            suggested = d["suggested"] or 0
            accepted = d["accepted"] or 0
            d["acceptance_rate"] = (
                round(accepted / suggested, 4) if suggested > 0 else 0.0
            )
            d["partial_acceptances"] = d["partial_acceptances"] or 0
            # “engagement rate” = fraction of suggestions the user
            # interacted with (accepted, partially accepted, or explicitly
            # rejected via Esc/cycle), excluding pure ignores. Counted by
            # DISTINCT suggestion_id so multiple partial-accept events on
            # the same suggestion don't inflate the metric.
            interacted_ids = conn.execute(
                """SELECT COUNT(DISTINCT suggestion_id) as n
                   FROM events
                   WHERE model = ?
                     AND event_type IN ('accepted','partially_accepted','rejected')
                     AND suggestion_id IS NOT NULL
                     AND timestamp >= datetime('now', ?)""",
                (d["model"], cutoff),
            ).fetchone()
            d["engagement_rate"] = (
                round(interacted_ids["n"] / suggested, 4) if suggested > 0 else 0.0
            )
            d["avg_latency_ms"] = (
                round(d["avg_latency_ms"], 1) if d["avg_latency_ms"] else 0.0
            )
            d["avg_confidence"] = (
                round(d["avg_confidence"], 3) if d["avg_confidence"] else 0.0
            )
            models.append(d)

        # Per-model latency percentiles (small dataset, compute in Python)
        for m in models:
            lat_rows = conn.execute(
                """SELECT latency_ms FROM events
                   WHERE model = ? AND event_type = 'suggested'
                     AND latency_ms IS NOT NULL
                     AND timestamp >= datetime('now', ?)
                   ORDER BY latency_ms""",
                (m["model"], cutoff),
            ).fetchall()
            lats = [r["latency_ms"] for r in lat_rows]
            if lats:
                m["p50_latency_ms"] = round(lats[len(lats) // 2], 1)
                m["p95_latency_ms"] = round(
                    lats[min(int(len(lats) * 0.95), len(lats) - 1)], 1
                )
            else:
                m["p50_latency_ms"] = 0.0
                m["p95_latency_ms"] = 0.0

        # Timeline: time-bucketed counts per model
        tl_rows = conn.execute(
            f"""SELECT
                {bucket_sql} as hour_bucket,
                model,
                SUM(CASE WHEN event_type = 'suggested' THEN 1 ELSE 0 END) as suggested,
                SUM(CASE WHEN event_type = 'accepted'  THEN 1 ELSE 0 END) as accepted
               FROM events
               WHERE timestamp >= datetime('now', ?)
               GROUP BY hour_bucket, model
               ORDER BY hour_bucket""",
            (cutoff,),
        ).fetchall()

        timeline = [
            {
                "hour": r["hour_bucket"],
                "model": r["model"],
                "suggested": r["suggested"] or 0,
                "accepted": r["accepted"] or 0,
            }
            for r in tl_rows
        ]

        total_suggested = sum(m["suggested"] for m in models)
        total_accepted = sum(m["accepted"] for m in models)
        total_partial = sum(m["partial_acceptances"] for m in models)
        overall_rate = (
            round(total_accepted / total_suggested, 4)
            if total_suggested > 0
            else 0.0
        )

        return {
            "hours": hours,
            "overall": {
                "suggested": total_suggested,
                "accepted": total_accepted,
                "partial_acceptances": total_partial,
                "acceptance_rate": overall_rate,
                "models": len(models),
            },
            "models": models,
            "timeline": timeline,
        }
