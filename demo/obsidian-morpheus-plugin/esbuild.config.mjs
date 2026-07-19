import esbuild from "esbuild";
import process from "process";

const prod = process.argv[2] === "production";

/** Resolve the plugin dir relative to this config file, so the script
 *  works no matter where it's invoked from. */
import { fileURLToPath } from "url";
import { dirname, resolve } from "path";
const __dirname = dirname(fileURLToPath(import.meta.url));

const config = {
  entryPoints: ["src/main.ts"],
  bundle: true,
  external: [
    "obsidian",
    "electron",
    "@codemirror/autocomplete",
    "@codemirror/collab",
    "@codemirror/commands",
    "@codemirror/language",
    "@codemirror/lint",
    "@codemirror/search",
    "@codemirror/state",
    "@codemirror/view",
    "@lezer/common",
    "@lezer/highlight",
    "@lezer/lr",
  ],
  format: "cjs",
  target: "es2018",
  logLevel: "info",
  sourcemap: prod ? false : "inline",
  treeShaking: true,
  outfile: resolve(__dirname, "main.js"),
  minify: prod,
};

if (prod) {
  // One-shot production build (minified, no sourcemap)
  esbuild.build(config).catch(() => process.exit(1));
} else {
  // Dev mode: watch src/ and rebuild on every save.
  // Pair with Obsidian's Ctrl/Cmd+R reload (or the hot-reload plugin)
  // to see changes without restarting Obsidian.
  const ctx = await esbuild.context(config);
  await ctx.watch();
  console.log("watching src/ for changes... (Ctrl+C to stop)");
}
