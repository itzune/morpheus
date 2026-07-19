/*
 * Morpheus Autocomplete — Obsidian plugin entry point.
 *
 * Connects to a Morpheus demo server (demo/server.py) and renders inline
 * ghost-text suggestions in the editor. The server URL is configurable, so
 * the same plugin works with:
 *   - Morpheus Mamba-2 (91M, on-device) — point at http://localhost:9090
 *   - Latxa 8B (GPU server)            — point at http://<gpu-host>:9090
 *
 * The server handles all model-specific concerns (tokenization, FIM
 * templating, output cleanup) per-backend; this plugin is backend-agnostic.
 */

import {
  App,
  Notice,
  Plugin,
  PluginSettingTab,
  Setting,
  requestUrl,
} from "obsidian";
import { createMorpheusExtension } from "./ghost-text";

export interface MorpheusSettings {
  /** Morpheus demo server base URL (no trailing slash). */
  serverUrl: string;
  /** Toggle ghost-text autocomplete on/off. */
  enabled: boolean;
  /** Debounce: ms to wait after the user stops typing before fetching. */
  triggerDelay: number;
  /** Maximum suggestion length in tokens. */
  maxTokens: number;
  /** Sampling temperature (0 = greedy). */
  temperature: number;
  /** Suppress suggestions below this confidence (0.0–1.0). */
  confidenceThreshold: number;
  /** Best-of-N: fire N parallel samples, keep highest-confidence. */
  bestOfN: number;
  /** Characters of text before the cursor to send as prefix. */
  contextBefore: number;
  /** Characters of text after the cursor to send as suffix (FIM). 0 = append-only. */
  contextAfter: number;
}

const DEFAULT_SETTINGS: MorpheusSettings = {
  serverUrl: "http://localhost:9090",
  enabled: true,
  triggerDelay: 500,
  maxTokens: 16,
  temperature: 0.2,
  confidenceThreshold: 0.15,
  bestOfN: 1,
  contextBefore: 1500,
  contextAfter: 400,
};

export default class MorpheusPlugin extends Plugin {
  settings: MorpheusSettings = DEFAULT_SETTINGS;
  private statusEl: HTMLElement | null = null;

  async onload() {
    await this.loadSettings();

    // Status bar item (click to toggle on/off)
    this.statusEl = this.addStatusBarItem();
    this.statusEl.addClass("morpheus-status");
    this.registerDomEvent(this.statusEl, "click", () => {
      this.settings.enabled = !this.settings.enabled;
      this.saveSettings();
      this.updateStatusBar();
      new Notice(`Morpheus: ${this.settings.enabled ? "on" : "off"}`);
    });

    // Register the CodeMirror 6 ghost-text extension
    this.registerEditorExtension(createMorpheusExtension(this));

    // Settings tab
    this.addSettingTab(new MorpheusSettingTab(this.app, this));

    // Command: toggle autocomplete
    this.addCommand({
      id: "morpheus-toggle",
      name: "Toggle autocomplete",
      callback: () => {
        this.settings.enabled = !this.settings.enabled;
        this.saveSettings();
        this.updateStatusBar();
        new Notice(`Morpheus: ${this.settings.enabled ? "on" : "off"}`);
      },
    });

    // Initial status + periodic refresh
    this.updateStatusBar();
    this.registerInterval(
      window.setInterval(() => this.updateStatusBar(), 60_000)
    );
  }

  async loadSettings() {
    this.settings = Object.assign({}, DEFAULT_SETTINGS, await this.loadData());
  }

  async saveSettings() {
    await this.saveData(this.settings);
  }

  /** Query the server's /api/model endpoint and show the model name in the
   *  status bar. Shows "offline" if the server is unreachable. */
  async updateStatusBar() {
    if (!this.statusEl) return;

    if (!this.settings.enabled) {
      this.statusEl.setText("Morpheus: off");
      return;
    }

    try {
      const url = this.settings.serverUrl.replace(/\/$/, "");
      const res = await requestUrl({ url: `${url}/api/model` });
      const name = res.json?.current?.name;
      if (name && name !== "none") {
        this.statusEl.setText(`Morpheus: ${name}`);
      } else {
        this.statusEl.setText("Morpheus: no model");
      }
    } catch {
      this.statusEl.setText("Morpheus: offline");
    }
  }
}

// ── Settings tab ───────────────────────────────────────────────────────

class MorpheusSettingTab extends PluginSettingTab {
  plugin: MorpheusPlugin;

  constructor(app: App, plugin: MorpheusPlugin) {
    super(app, plugin);
    this.plugin = plugin;
  }

  display(): void {
    const { containerEl } = this;
    containerEl.empty();

    containerEl.createEl("h3", { text: "Morpheus Autocomplete" });
    const desc = containerEl.createEl("p", {
      cls: "setting-item-description",
    });
    desc.setText(
      "Ghost-text autocompletion via a Morpheus demo server. " +
        "Point at localhost for the on-device Mamba-2 model, or a GPU " +
        "server URL for Latxa 8B. The server handles tokenization, FIM " +
        "templating, and output cleanup; this plugin is backend-agnostic."
    );

    // ── Server URL ──
    new Setting(containerEl)
      .setName("Server URL")
      .setDesc("Morpheus demo server endpoint (no trailing slash).")
      .addText((text) =>
        text
          .setPlaceholder("http://localhost:9090")
          .setValue(this.plugin.settings.serverUrl)
          .onChange(async (value) => {
            this.plugin.settings.serverUrl = value;
            await this.plugin.saveSettings();
            this.plugin.updateStatusBar();
          })
      );

    // ── Enabled ──
    new Setting(containerEl)
      .setName("Enabled")
      .setDesc(
        "Toggle ghost-text autocomplete. " +
          "(You can also click the status bar item or use the command palette.)"
      )
      .addToggle((toggle) =>
        toggle
          .setValue(this.plugin.settings.enabled)
          .onChange(async (value) => {
            this.plugin.settings.enabled = value;
            await this.plugin.saveSettings();
            this.plugin.updateStatusBar();
          })
      );

    // ── Trigger delay ──
    new Setting(containerEl)
      .setName("Trigger delay (ms)")
      .setDesc("How long to wait after you stop typing before fetching a suggestion.")
      .addSlider((slider) =>
        slider
          .setLimits(100, 2000, 50)
          .setValue(this.plugin.settings.triggerDelay)
          .setDynamicTooltip()
          .onChange(async (value) => {
            this.plugin.settings.triggerDelay = value;
            await this.plugin.saveSettings();
          })
      );

    // ── Max tokens ──
    new Setting(containerEl)
      .setName("Max tokens")
      .setDesc("Maximum length of each suggestion (in tokens).")
      .addSlider((slider) =>
        slider
          .setLimits(1, 64, 1)
          .setValue(this.plugin.settings.maxTokens)
          .setDynamicTooltip()
          .onChange(async (value) => {
            this.plugin.settings.maxTokens = value;
            await this.plugin.saveSettings();
          })
      );

    // ── Temperature ──
    new Setting(containerEl)
      .setName("Temperature")
      .setDesc("0 = deterministic (greedy), higher = more creative. Use 0 for predictable ghost text.")
      .addSlider((slider) =>
        slider
          .setLimits(0, 1, 0.05)
          .setValue(this.plugin.settings.temperature)
          .setDynamicTooltip()
          .onChange(async (value) => {
            this.plugin.settings.temperature = value;
            await this.plugin.saveSettings();
          })
      );

    // ── Confidence threshold ──
    new Setting(containerEl)
      .setName("Confidence threshold")
      .setDesc("Suppress suggestions below this confidence. Higher = fewer but better suggestions.")
      .addSlider((slider) =>
        slider
          .setLimits(0, 1, 0.05)
          .setValue(this.plugin.settings.confidenceThreshold)
          .setDynamicTooltip()
          .onChange(async (value) => {
            this.plugin.settings.confidenceThreshold = value;
            await this.plugin.saveSettings();
          })
      );

    // ── Best-of-N ──
    new Setting(containerEl)
      .setName("Best-of-N")
      .setDesc("Fire N parallel samples and keep the highest-confidence one. Only useful with temperature > 0.")
      .addSlider((slider) =>
        slider
          .setLimits(1, 5, 1)
          .setValue(this.plugin.settings.bestOfN)
          .setDynamicTooltip()
          .onChange(async (value) => {
            this.plugin.settings.bestOfN = value;
            await this.plugin.saveSettings();
          })
      );

    // ── Context before cursor ──
    new Setting(containerEl)
      .setName("Context before cursor (chars)")
      .setDesc("How many characters before the cursor to send as prefix context.")
      .addSlider((slider) =>
        slider
          .setLimits(100, 8000, 100)
          .setValue(this.plugin.settings.contextBefore)
          .setDynamicTooltip()
          .onChange(async (value) => {
            this.plugin.settings.contextBefore = value;
            await this.plugin.saveSettings();
          })
      );

    // ── Context after cursor ──
    new Setting(containerEl)
      .setName("Context after cursor (chars)")
      .setDesc("How many characters after the cursor to send as suffix (enables FIM infill). Set to 0 for append-only mode.")
      .addSlider((slider) =>
        slider
          .setLimits(0, 4000, 50)
          .setValue(this.plugin.settings.contextAfter)
          .setDynamicTooltip()
          .onChange(async (value) => {
            this.plugin.settings.contextAfter = value;
            await this.plugin.saveSettings();
          })
      );

    // ── Test connection ──
    new Setting(containerEl)
      .setName("Test connection")
      .setDesc("Check if the server is reachable and which model is loaded.")
      .addButton((btn) =>
        btn
          .setButtonText("Test")
          .setCta()
          .onClick(async () => {
            btn.setButtonText("Testing...");
            btn.setDisabled(true);
            try {
              const url = this.plugin.settings.serverUrl.replace(/\/$/, "");
              const res = await requestUrl({ url: `${url}/api/model` });
              const name = res.json?.current?.name || "unknown";
              const available = res.json?.available?.length ?? 0;
              new Notice(`✓ Connected — model: ${name} (${available} available)`, 5000);
            } catch (e) {
              new Notice(
                `✗ Cannot reach server at ${this.plugin.settings.serverUrl}`,
                5000
              );
            } finally {
              btn.setButtonText("Test");
              btn.setDisabled(false);
            }
          })
      );
  }
}
