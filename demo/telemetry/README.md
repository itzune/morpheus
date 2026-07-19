# Morpheus Telemetry

A standalone FastAPI service that collects autocomplete interaction events
from the Obsidian plugin and serves a dashboard for cross-model analysis.

## Architecture

```
Obsidian plugin ──POST /api/events──→ telemetry server ──→ SQLite (WAL)
                    GET /dashboard  ──→ Chart.js dashboard
```

The telemetry endpoint is **separate** from the completion server, following
the GitHub Copilot pattern: "execution" (completion) and "observation"
(telemetry) are decoupled. This lets a single dashboard compare models
running on different servers:

```
Obsidian plugin ──→ Morpheus (localhost:9090)    ← completion
                ──→ Kimu     (GPU:9092)          ← completion
                ──→ Latxa    (GPU:9090)          ← completion
                ──→ telemetry (localhost:9100)   ← observation (all models)
```

## Events Tracked

| Event | When | Key fields |
|-------|------|------------|
| `suggested` | Ghost text appeared | `latency_ms`, `confidence`, `model` |
| `accepted` | User pressed Tab | `suggestion_id` (links to `suggested`) |
| `rejected` | User pressed Esc | `suggestion_id` |
| `ignored` | User kept typing (ghost faded) | `suggestion_id` |

The headline metric is **acceptance rate** = `accepted / suggested`, computed
per model. This directly answers "which model's suggestions are most useful?"

## Quick Start

### Option 1: Docker Compose (recommended)

The telemetry server is included in `docker-compose.yml` alongside the
completion server. Both start together:

```bash
cd demo
docker compose -f docker-compose.yml -f docker-compose.local.yml up -d --build
```

- Completion server: `http://localhost:9090`
- Telemetry server: `http://localhost:9100`
- Dashboard: `http://localhost:9100/dashboard`

The SQLite database persists in a Docker volume (`morpheus-telemetry`).

### Option 2: Standalone

```bash
cd demo/telemetry
pip install -r requirements.txt
uvicorn server:app --host 0.0.0.0 --port 9100
```

Then open `http://localhost:9100/dashboard`.

## Configuration

| Env var | Default | Description |
|---------|---------|-------------|
| `TELEMETRY_DB` | `./telemetry.db` | Path to SQLite database file |

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/events` | Receive a batch of events `{ events: [...] }` |
| `GET` | `/api/events?model=X&limit=100` | Query raw events (newest first) |
| `GET` | `/api/stats?hours=24` | Aggregated stats for the dashboard |
| `GET` | `/api/models` | Distinct model names |
| `GET` | `/health` | Health check + total event count |
| `GET` | `/dashboard` | HTML dashboard (Chart.js, auto-refreshing) |

## Plugin Configuration

In Obsidian → Settings → Morpheus Autocomplete → Telemetry:

1. **Enable telemetry** — toggle on (opt-in, off by default)
2. **Telemetry endpoint** — `http://localhost:9100` (or your server URL)
3. **Include suggestion text** — if on, sends the actual suggestion and
   context text (for qualitative replay). If off, sends metrics only
   (latency, confidence, lengths).

The plugin buffers events and flushes every 5 seconds (or when 20 events
accumulate). Events are best-effort: if the telemetry server is down,
events are silently dropped.

## Privacy

- **Opt-in**: telemetry is disabled by default.
- **Self-hostable**: the endpoint URL is configurable. Point it at your own
  server for full data control.
- **Content toggle**: suggestion/context text is sent only when explicitly
  enabled. With it off, only metrics (latency, confidence, event type,
  model name) are sent.
- **No user identification**: events carry a random session ID (regenerated
  per Obsidian launch), not a user account or email.

## Database

SQLite in WAL mode (concurrent reads during writes). The database file is
a single `telemetry.db` — back it up by copying the file. For data
retention, periodically delete old events:

```sql
DELETE FROM events WHERE timestamp < datetime('now', '-30 days');
```
