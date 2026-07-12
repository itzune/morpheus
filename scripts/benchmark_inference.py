#!/usr/bin/env python3
"""
Benchmark Morpheus v2 Mamba-2 model inference across hardware configurations.

Measures:
  - Memory footprint (peak RSS for CPU, VRAM for GPU)
  - Prompt processing speed (prefill tok/s)
  - Generation speed (decode tok/s)
  - Time-to-first-token (TTFT, user-facing latency)
  - End-to-end autocomplete latency (short prompt + few tokens)

Uses the llama-server /completion endpoint, which returns timing data:
  {
    "timings": {
      "prompt_n", "prompt_ms", "prompt_per_second",
      "predicted_n", "predicted_ms", "predicted_per_second"
    }
  }

Usage:
    # Run against a running llama-server
    python3 scripts/benchmark_inference.py --host http://localhost:8080 --label "CPU-i7-8550U"

    # With GPU VRAM measurement
    python3 scripts/benchmark_inference.py --host http://localhost:8080 --label "GPU-L40" --gpu
"""
import argparse
import json
import time
import statistics
import subprocess
import sys
import os
from pathlib import Path

import httpx

# ── Test prompts designed for autocomplete scenarios ─────────────────────────

# Short context, typical autocomplete (user typing in editor)
PROMPTS = {
    "short_10tok": "Euskal Herriko",  # ~3 tokens, minimal context
    "medium_30tok": "Gaur eguraldi ona egiten du eta kalean paseatzera",  # ~10 tokens
    "long_100tok": (
        "Euskaltzaindia euskara zaindu, aztertu, zabaldu, batu eta hobetzea helburu "
        "duen erakunde ofiziala da. 1919an sortu zuten, eta gaur egun ere bere "
        "lana egiten jarraitzen du euskara batuaren alde. Euskaltzaindiaren "
        "helburu nagusia euskaraaren batasuna eta"
    ),  # ~40 tokens
    "very_long_200tok": (
        "Euskaltzaindia euskara zaindu, aztertu, zabaldu, batu eta hobetzea helburu "
        "duen erakunde ofiziala da. 1919an sortu zuten, eta gaur egun ere bere "
        "lana egiten jarraitzen du euskara batuaren alde. Euskaltzaindiaren "
        "helburu nagusia euskaraaren batasuna eta garapena bermatzea da, eta "
        "horretarako hainbat lan ildo ditu: gramatika, hiztegia, ortografia, "
        "eta literatura. Euskaltzaindiak argitaratutako lanen artean, Euskal "
        "Gramatika osoa eta Hiztegi Batua nabarmentzen dira. Erakundeak bere "
        "historian zehar paper garrantzitsua jokatu du euskararen нормализazioan "
        "eta bere eginkizuna"
    ),  # ~80+ tokens
}

# Autocomplete-specific: short prompt, few tokens (next-word prediction)
AUTOCOMPLETE_TESTS = [
    ("Euskal Herriko", 3),        # 3-token continuation
    ("Gaur eguraldi ona egiten du eta", 5),  # 5-token continuation
    ("Bihar goizean lanera joan beharko dut", 5),
    ("Euskara ikasten ari naiz baina oraindik", 5),
    ("Lagunekin afaria egitea asteburuan", 5),
]


def query_completion(host, prompt, n_predict=10, temperature=0.0, timeout=30):
    """Send a completion request and return the response with timings."""
    r = httpx.post(
        f"{host}/completion",
        json={
            "prompt": prompt,
            "n_predict": n_predict,
            "temperature": temperature,
            "stream": False,
            "top_k": 1,  # greedy
        },
        timeout=timeout,
    )
    return r.json()


def get_process_rss(pid):
    """Get RSS (resident set size) in MB for a process."""
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024  # KB -> MB
    except Exception:
        pass
    return None


def get_gpu_vram():
    """Get GPU VRAM usage in MB via nvidia-smi."""
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        return int(r.stdout.strip())
    except Exception:
        return None


def benchmark_prompt_speed(host, label, n_runs=5):
    """Measure prompt processing and generation speed across prompt lengths."""
    print(f"\n{'='*70}")
    print(f"  PROMPT PROCESSING & GENERATION SPEED ({label})")
    print(f"{'='*70}")
    print(f"  {'Prompt':<20s} {'Prompt':>7s} {'Prefill':>10s} {'Decode':>10s} {'TTFT':>8s}")
    print(f"  {'Length':<20s} {'Tokens':>7s} {'tok/s':>10s} {'tok/s':>10s} {'(ms)':>8s}")
    print(f"  {'-'*60}")

    results = []
    for name, prompt in PROMPTS.items():
        prefill_speeds = []
        decode_speeds = []
        ttfts = []
        prompt_n = 0

        for _ in range(n_runs):
            resp = query_completion(host, prompt, n_predict=20)
            t = resp.get("timings", {})
            prompt_n = t.get("prompt_n", 0)
            prompt_ms = t.get("prompt_ms", 0)
            pred_ms = t.get("predicted_ms", 0)
            pred_n = t.get("predicted_n", 0)

            if prompt_ms > 0:
                prefill_speeds.append(prompt_n / (prompt_ms / 1000))
            if pred_ms > 0 and pred_n > 0:
                decode_speeds.append(pred_n / (pred_ms / 1000))
            # TTFT = time to generate first token ≈ prompt_ms + (1 token decode time)
            if prompt_ms > 0 and pred_n > 0:
                ttft_ms = prompt_ms + (pred_ms / pred_n)
                ttfts.append(ttft_ms)

        avg_prefill = statistics.mean(prefill_speeds) if prefill_speeds else 0
        avg_decode = statistics.mean(decode_speeds) if decode_speeds else 0
        avg_ttft = statistics.mean(ttfts) if ttfts else 0

        print(f"  {name:<20s} {prompt_n:>7d} {avg_prefill:>10.1f} {avg_decode:>10.1f} {avg_ttft:>8.1f}")
        results.append({
            "prompt_name": name,
            "prompt_tokens": prompt_n,
            "prefill_tok_s": round(avg_prefill, 1),
            "decode_tok_s": round(avg_decode, 1),
            "ttft_ms": round(avg_ttft, 1),
        })

    return results


def benchmark_autocomplete_latency(host, label, n_runs=10):
    """Measure end-to-end autocomplete latency (short prompt, few tokens)."""
    print(f"\n{'='*70}")
    print(f"  AUTOCOMPLETE LATENCY ({label})")
    print(f"  Short prompt + few tokens = realistic next-word prediction")
    print(f"{'='*70}")
    print(f"  {'Prompt':<45s} {'Gen':>4s} {'Latency':>8s} {'Decode':>10s}")
    print(f"  {'':45s} {'tok':>4s} {'(ms)':>8s} {'tok/s':>10s}")
    print(f"  {'-'*70}")

    results = []
    for prompt, n_pred in AUTOCOMPLETE_TESTS:
        latencies = []
        decode_speeds = []
        generated_text = ""

        for _ in range(n_runs):
            t0 = time.perf_counter()
            resp = query_completion(host, prompt, n_predict=n_pred)
            wall_ms = (time.perf_counter() - t0) * 1000

            t = resp.get("timings", {})
            pred_ms = t.get("predicted_ms", 0)
            pred_n = t.get("predicted_n", 0)
            generated_text = resp.get("content", "")[:30]

            latencies.append(wall_ms)
            if pred_ms > 0 and pred_n > 0:
                decode_speeds.append(pred_n / (pred_ms / 1000))

        avg_lat = statistics.mean(latencies)
        avg_decode = statistics.mean(decode_speeds) if decode_speeds else 0
        p95_lat = sorted(latencies)[int(len(latencies) * 0.95)] if len(latencies) > 1 else latencies[0]

        preview = prompt[:42] + "..." if len(prompt) > 42 else prompt
        print(f"  {preview:<45s} {n_pred:>4d} {avg_lat:>8.1f} {avg_decode:>10.1f}")
        results.append({
            "prompt": prompt,
            "n_predict": n_pred,
            "wall_latency_ms": round(avg_lat, 1),
            "p95_latency_ms": round(p95_lat, 1),
            "decode_tok_s": round(avg_decode, 1),
            "sample_output": generated_text,
        })

    # Summary
    all_lat = [r["wall_latency_ms"] for r in results]
    all_decode = [r["decode_tok_s"] for r in results]
    print(f"\n  Average wall latency: {statistics.mean(all_lat):.1f} ms")
    print(f"  Average decode speed: {statistics.mean(all_decode):.1f} tok/s")

    return results


def benchmark_memory(host, label, llama_pid=None, use_gpu=False):
    """Measure memory footprint."""
    print(f"\n{'='*70}")
    print(f"  MEMORY FOOTPRINT ({label})")
    print(f"{'='*70}")

    # Get model info from server
    try:
        r = httpx.get(f"{host}/props", timeout=5)
        props = r.json()
        model_path = props.get("model_path", "?")
        n_params = props.get("n_params", 0)
        n_ctx = props.get("n_ctx", 0)
        model_size_mb = os.path.getsize(model_path) / (1024 * 1024) if os.path.exists(model_path) else 0
    except Exception:
        model_path = "?"
        n_params = 0
        n_ctx = 0
        model_size_mb = 0

    print(f"  Model file: {Path(model_path).name if model_path != '?' else '?'}")
    print(f"  Model size: {model_size_mb:.1f} MB")
    print(f"  Parameters: {n_params / 1e6:.1f}M" if n_params else "  Parameters: ?")
    print(f"  Context:    {n_ctx} tokens")

    # CPU memory (RSS of llama-server process)
    if llama_pid:
        rss = get_process_rss(llama_pid)
        if rss:
            print(f"  Process RSS (CPU RAM): {rss:.1f} MB")

    # GPU memory
    if use_gpu:
        vram = get_gpu_vram()
        if vram is not None:
            print(f"  GPU VRAM used: {vram:.1f} MB")
        else:
            print(f"  GPU VRAM: (nvidia-smi not available)")

    return {
        "model_file": Path(model_path).name if model_path != "?" else "?",
        "model_size_mb": round(model_size_mb, 1),
        "n_params": n_params,
        "n_ctx": n_ctx,
        "process_rss_mb": round(get_process_rss(llama_pid), 1) if llama_pid else None,
        "gpu_vram_mb": get_gpu_vram() if use_gpu else None,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark Morpheus inference")
    parser.add_argument("--host", default="http://localhost:8080",
                        help="llama-server URL")
    parser.add_argument("--label", default="unknown",
                        help="Hardware label for output (e.g., 'CPU-i7-8550U', 'GPU-L40')")
    parser.add_argument("--pid", type=int, default=None,
                        help="llama-server PID for memory measurement")
    parser.add_argument("--gpu", action="store_true",
                        help="Measure GPU VRAM via nvidia-smi")
    parser.add_argument("--output", default=None,
                        help="Save results to JSON file")
    parser.add_argument("--runs", type=int, default=5,
                        help="Number of runs per benchmark")
    args = parser.parse_args()

    print(f"\n{'#'*70}")
    print(f"  Morpheus v2 Inference Benchmark")
    print(f"  Hardware: {args.label}")
    print(f"  Server: {args.host}")
    print(f"{'#'*70}")

    # Wait for server to be ready
    print("\n  Waiting for server...", end=" ", flush=True)
    for i in range(30):
        try:
            r = httpx.get(f"{args.host}/health", timeout=2)
            if r.status_code == 200:
                print("ready!")
                break
        except Exception:
            time.sleep(1)
    else:
        print("FAILED to connect!")
        sys.exit(1)

    # Run benchmarks
    mem_results = benchmark_memory(args.host, args.label, args.pid, args.gpu)
    speed_results = benchmark_prompt_speed(args.host, args.label, args.runs)
    latency_results = benchmark_autocomplete_latency(args.host, args.label, args.runs)

    # Summary
    print(f"\n{'='*70}")
    print(f"  SUMMARY: {args.label}")
    print(f"{'='*70}")
    print(f"  Model: {mem_results['model_file']} ({mem_results['model_size_mb']:.1f} MB)")

    # Best-case decode speed (short prompt, many tokens)
    best_decode = max(r["decode_tok_s"] for r in speed_results if r["decode_tok_s"] > 0)
    best_prefill = max(r["prefill_tok_s"] for r in speed_results if r["prefill_tok_s"] > 0)
    avg_lat = statistics.mean(r["wall_latency_ms"] for r in latency_results)

    print(f"  Peak decode speed:    {best_decode:.1f} tok/s")
    print(f"  Peak prefill speed:   {best_prefill:.1f} tok/s")
    print(f"  Avg autocomplete latency: {avg_lat:.1f} ms")
    if mem_results["process_rss_mb"]:
        print(f"  Process RSS:          {mem_results['process_rss_mb']:.1f} MB")
    if mem_results["gpu_vram_mb"] is not None:
        print(f"  GPU VRAM:             {mem_results['gpu_vram_mb']:.1f} MB")

    # Save results
    if args.output:
        out = {
            "label": args.label,
            "host": args.host,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "memory": mem_results,
            "speed": speed_results,
            "latency": latency_results,
            "summary": {
                "peak_decode_tok_s": round(best_decode, 1),
                "peak_prefill_tok_s": round(best_prefill, 1),
                "avg_autocomplete_latency_ms": round(avg_lat, 1),
            },
        }
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
