# Morpheus Telemetry Server
#
# A lightweight FastAPI + SQLite service that collects autocomplete
# interaction events from the Obsidian plugin and serves a cross-model
# analytics dashboard. No model inference — just event storage + charts.
#
# Build (from repo root):
#   docker build -f demo/telemetry/Dockerfile -t morpheus-telemetry .
#
# Run:
#   docker run -p 9100:9100 -v morpheus-telemetry:/data morpheus-telemetry
#
# Or via docker-compose (see docker-compose.local.yml).

FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (cached layer)
COPY demo/telemetry/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY demo/telemetry/server.py demo/telemetry/db.py /app/
COPY demo/telemetry/static/ /app/static/

# Default env (overridable)
# TELEMETRY_DB: path to SQLite file. Mount a volume here to persist data.
ENV TELEMETRY_DB=/data/telemetry.db

EXPOSE 9100

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "9100"]
