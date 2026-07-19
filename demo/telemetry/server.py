"""Morpheus Telemetry Server.

A standalone FastAPI service that collects autocomplete interaction events
from the Obsidian plugin and serves a dashboard for cross-model analysis.

Architecture:
    Obsidian plugin ──POST /api/events──→ this server ──→ SQLite (WAL)
                        GET /dashboard  ──→ Chart.js dashboard

The telemetry endpoint is separate from the completion server, following
the GitHub Copilot pattern: "execution" (completion) and "observation"
(telemetry) are decoupled. This lets a single dashboard compare models
running on different servers (Morpheus on localhost, Kimu/Latxa on GPU).

Run:
    cd demo/telemetry
    pip install -r requirements.txt
    uvicorn server:app --host 0.0.0.0 --port 9100

Env vars:
    TELEMETRY_DB   Path to SQLite file (default: ./telemetry.db)
    TELEMETRY_PORT Port (default: 9100; or pass --port to uvicorn)
"""
import os
from pathlib import Path

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from typing import Optional

from db import init_db, insert_events, get_events, get_stats, get_models, get_total_count

app = FastAPI(title="Morpheus Telemetry", version="1.0.0")

# Allow cross-origin requests (plugin uses requestUrl which bypasses CORS,
# but this helps if someone queries the API from a browser on another origin)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize database
DB_PATH = os.environ.get(
    "TELEMETRY_DB", str(Path(__file__).resolve().parent / "telemetry.db")
)
init_db(DB_PATH)


# ── Pydantic models ────────────────────────────────────────────────────

class TelemetryEvent(BaseModel):
    timestamp: str
    session_id: str
    model: str
    event_type: str = Field(
        ..., description="suggested | accepted | rejected | ignored"
    )
    suggestion_id: Optional[str] = None
    latency_ms: Optional[float] = None
    confidence: Optional[float] = None
    suggestion_length: Optional[int] = None
    prompt_length: Optional[int] = None
    suggestion_text: Optional[str] = None
    context: Optional[str] = None
    accepted_length: Optional[int] = None
    reject_reason: Optional[str] = Field(
        None, description="dismissed | cycled | cycled_back"
    )


class EventsBatch(BaseModel):
    events: list[TelemetryEvent]


# ── API endpoints ──────────────────────────────────────────────────────

@app.post("/api/events")
def receive_events(batch: EventsBatch):
    """Receive a batch of telemetry events from the plugin."""
    events = [e.model_dump() for e in batch.events]
    count = insert_events(events)
    return {"ok": True, "received": count}


@app.get("/api/events")
def list_events(
    model: Optional[str] = Query(None),
    limit: int = Query(100, le=1000),
):
    """Query raw events (most recent first)."""
    return {"events": get_events(model=model, limit=limit)}


@app.get("/api/stats")
def stats(hours: int = Query(24, le=720)):
    """Aggregated stats for the dashboard."""
    return get_stats(hours=hours)


@app.get("/api/models")
def models():
    """Distinct model names seen in the telemetry data."""
    return {"models": get_models()}


@app.get("/health")
def health():
    return {"status": "ok", "events": get_total_count()}


@app.get("/", response_class=HTMLResponse)
@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    """Serve the telemetry dashboard (Chart.js, auto-refreshing)."""
    html_path = Path(__file__).resolve().parent / "static" / "dashboard.html"
    return HTMLResponse(html_path.read_text(encoding="utf-8"))
