# Morpheus Autocomplete for Obsidian

Ghost-text autocompletion for [Obsidian](https://obsidian.md), powered by a
[Morpheus](https://github.com/itzune/morpheus) demo server. Type naturally —
suggestions appear as transparent inline text. **Tab** to accept, **Esc** to
dismiss.

The server URL is configurable, so the **same plugin** works with either tier
of the Morpheus two-tier architecture:

| Backend | Server URL | Hardware | Latency |
|---------|-----------|----------|---------|
| **Morpheus** (Mamba-2, 91M) | `http://localhost:9090` | Consumer laptop CPU | ~75 ms |
| **Latxa 8B** (Llama-3.1 base) | `http://<gpu-host>:9090` | NVIDIA L40 GPU | ~115 ms |

Just change the **Server URL** setting — no plugin reinstall needed.

## How it works

```
Obsidian editor ──prefix+suffix──▶ Morpheus demo server ──▶ llama-server ──▶ Mamba-2 / Latxa 8B
       ▲                               (FIM template,            (GGUF)
       │  ghost text                    tokenization, cleanup)
       └──── text + confidence ────────┘
```

1. You type. After a short pause (default 500 ms), the plugin sends the text
   before and after the cursor to the server's `POST /v1/complete` endpoint.
2. The server applies the FIM template (prefix + suffix → infill), tokenizes,
   calls `llama-server`, and cleans up the output — all per-backend.
3. The plugin renders the returned text as inline ghost text at the cursor.
   Low-confidence suggestions (below the threshold) are suppressed.
4. **Tab** inserts the suggestion; **Esc** dismisses it.

The plugin is **backend-agnostic**: it speaks the simple `{prefix, suffix}
→ {text, confidence}` protocol. All model-specific concerns (tokenization,
FIM sentinels, byte-fallback cleanup) live in the server's backend layer
(`demo/backends.py`).

## Dev testing (hot reload)

The dev script watches `src/` and rebuilds `main.js` on every save. Pair it
with a symlink into your vault so Obsidian picks up changes instantly:

```bash
# 1. One-time: symlink the plugin into your vault
VAULT=~/Dev/xezpeleta/obsidian   # ← your vault path
ln -sfn "$(pwd)" "$VAULT/.obsidian/plugins/morpheus-autocomplete"

# 2. Start the Morpheus server (if not already running)
cd ../../.. && docker compose -f demo/docker-compose.local.yml up -d

# 3. Start the watcher (leave it running in a terminal)
npm run dev
```

Then in Obsidian (with that vault open):
1. Reload: **Ctrl/Cmd+R** (or restart Obsidian)
2. **Settings → Community plugins** → enable **Morpheus Autocomplete**
3. Open a note, start typing — ghost text should appear after ~500ms
4. **Settings → Morpheus Autocomplete** → **Test connection** to verify the server

**Edit loop:** Edit `src/*.ts` → save → esbuild rebuilds → reload Obsidian
(Ctrl/Cmd+R). For automatic reload-on-save (no manual reload), install the
[Hot-Reload](https://github.com/pjeby/obsidian-hot-reload) community plugin.

## Install

### From source (development)

```bash
cd demo/obsidian-morpheus-plugin
npm install
npm run build        # outputs main.js (production, minified)
# or: npm run dev   # watches src/ and rebuilds on save
```

Then copy these four files into your Obsidian vault's plugins folder
(`<vault>/.obsidian/plugins/morpheus-autocomplete/`):

- `main.js`
- `manifest.json`
- `styles.css`

Restart Obsidian (or reload — `Ctrl/Cmd+R`), then enable **Morpheus
Autocomplete** in Settings → Community plugins.

> **Dev loop:** Symlink the plugin folder into your vault and run `npm run dev`
> to rebuild on every save. Reload Obsidian (`Ctrl/Cmd+R`) to pick up changes.

### Manual install (no build)

If you have a prebuilt `main.js`, copy `main.js`, `manifest.json`, and
`styles.css` into `<vault>/.obsidian/plugins/morpheus-autocomplete/` and
enable the plugin in Obsidian.

## Configuration

Open **Settings → Morpheus Autocomplete**:

| Setting | Default | Description |
|---------|---------|-------------|
| **Server URL** | `http://localhost:9090` | Morpheus demo server endpoint. Change this to point at the GPU server for Latxa 8B. |
| **Enabled** | on | Toggle autocomplete. (Also: click the status bar item, or run the "Toggle autocomplete" command.) |
| **Trigger delay** | 500 ms | Pause length after typing before fetching. Lower = snappier but more requests. |
| **Max tokens** | 16 | Suggestion length cap. Ghost text longer than ~1 line is rarely useful. |
| **Temperature** | 0.2 | 0 = greedy/deterministic. Higher = more creative but less predictable. |
| **Confidence threshold** | 0.15 | Suppress suggestions below this confidence. Higher = fewer but better. |
| **Best-of-N** | 1 | Fire N parallel samples, keep highest-confidence. Only useful with temperature > 0. |
| **Context before cursor** | 1500 chars | How much text before the cursor to send as prefix. |
| **Context after cursor** | 400 chars | How much text after the cursor to send as suffix (FIM infill). Set to 0 for append-only mode. |

Use **Test connection** to verify the server is reachable and see which model
is loaded.

## Starting a server

### Morpheus (on-device, local)

```bash
cd demo
docker compose -f docker-compose.local.yml up -d
# serves at http://localhost:9090, CPU mode
```

### Latxa 8B (GPU server)

```bash
# on the GPU server (e.g. 10.2.121.210):
/root/morpheus-mamba/scripts/serve_latxa.sh
# serves at http://10.2.121.210:9090, Latxa 8B Q6_K on L40
```

Then set **Server URL** in the plugin settings to match.

## Architecture notes

- **CodeMirror 6**: The plugin uses CM6's `StateField` + `WidgetType` +
  `ViewPlugin` pattern to render inline decorations. Suggestions live in a
  state field cleared on any doc/selection change, so they never drift from
  the cursor.
- **Stale-response guard**: Each keystroke increments a generation counter.
  When a response arrives, it's discarded if either (a) a newer request was
  fired, or (b) the doc has changed since the request was issued.
- **Acceptance logging**: On Tab-accept, the plugin fire-and-forgets an event
  to the server's `POST /api/log` endpoint. These logs build the eval dataset
  used for CSR (Character Saved Ratio) measurement — see `demo/csr_eval.py`.
- **`requestUrl`**: Uses Obsidian's built-in HTTP client (not `fetch`) to
  avoid CORS issues and work within the Electron sandbox.

## License

MIT. See the parent [Morpheus](https://github.com/itzune/morpheus) project.
