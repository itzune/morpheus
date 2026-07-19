#!/usr/bin/env python3
"""Latency + resource benchmark for the Morpheus demo autocomplete endpoint.

Measures user-facing latency (the /api/autocomplete/greedy product path) and
concurrently samples compute/memory resources for the model-serving processes.

Usage:
  python3 bench_latency.py --host H --ports 9090:Latxa 9091:Morpheus \
      --pids 100485:Latxa 101157:Morpheus --gpu
"""
import argparse, json, statistics, subprocess, sys, threading, time, urllib.request, urllib.parse
from concurrent.futures import ThreadPoolExecutor

PROMPTS = [
    "Egun on! Astelehenean bilera bat egitea proposatzen",
    "Adimen artifizialak Hezkuntzan izango duen eragina",
    "Suhesia sareko komunikazio guztiak",
    "Euskara ikasten ari naiz eta",
    "Gaur egun, teknologiak gure bizimodua",
    "Barkatu, baina ez dut",
]

def greedy(host, port, text, max_tokens=8, timeout=60):
    url = f"http://{host}:{port}/api/autocomplete/greedy"
    q = urllib.parse.urlencode({"text": text, "max_tokens": max_tokens})
    t0 = time.perf_counter()
    with urllib.request.urlopen(f"{url}?{q}", timeout=timeout) as r:
        body = json.loads(r.read())
    wall = (time.perf_counter() - t0) * 1000
    return wall, body

def bench_model(host, port, reps, max_tokens):
    # warmup
    for p in PROMPTS[:2]:
        try: greedy(host, port, p, max_tokens)
        except Exception: pass
    samples = []
    for _ in range(reps):
        for p in PROMPTS:
            try:
                wall, body = greedy(host, port, p, max_tokens)
                rep = body.get("latency_ms", wall)
                toks = len(body.get("tokens", [])) or max_tokens
                samples.append({"prompt": p, "wall_ms": wall, "reported_ms": rep, "tokens": toks})
            except Exception as e:
                samples.append({"prompt": p, "error": str(e)})
            time.sleep(0.05)
    return samples

def pct(xs, p):
    if not xs: return float("nan")
    xs = sorted(xs); k = (len(xs)-1) * p/100; f = int(k); c = min(f+1, len(xs)-1)
    return xs[f] + (xs[c]-xs[f])*(k-f)

def summarize(samples):
    walls = [s["wall_ms"] for s in samples if "wall_ms" in s]
    reps = [s["reported_ms"] for s in samples if "reported_ms" in s]
    toks = [s["tokens"] for s in samples if "tokens" in s]
    total_toks = sum(toks); total_wall = sum(walls)/1000
    return {
        "n": len(walls),
        "wall_mean": statistics.mean(walls),
        "wall_p50": pct(walls, 50),
        "wall_p95": pct(walls, 95),
        "wall_min": min(walls), "wall_max": max(walls),
        "reported_mean": statistics.mean(reps) if reps else 0,
        "tok_per_s": total_toks/total_wall if total_wall > 0 else 0,
        "errors": sum(1 for s in samples if "error" in s),
    }

# ---- resource sampler ----
def sample_resources(pids, with_gpu, stop):
    rows = []
    while not stop.is_set():
        row = {"t": time.time()}
        for pid, name in pids:
            try:
                out = subprocess.check_output(
                    ["ps","-p",str(pid),"-o","%cpu,%mem,rss,vsz","--no-headers"],
                    text=True, stderr=subprocess.DEVNULL).split()
                rows.append({**row, "name": name, "pid": pid,
                             "cpu": float(out[0]), "mem_pct": float(out[1]),
                             "rss_mb": int(out[2])/1024, "vsz_mb": int(out[3])/1024})
            except Exception: pass
        if with_gpu:
            try:
                out = subprocess.check_output(
                    ["nvidia-smi","--query-gpu=utilization.gpu,memory.used,memory.total",
                     "--format=csv,noheader,nounits"], text=True).strip().split(", ")
                rows.append({**row, "name":"<gpu>", "util": int(out[0]),
                             "vram_used_mb": int(out[1]), "vram_total_mb": int(out[2])})
            except Exception: pass
            try:
                out = subprocess.check_output(
                    ["nvidia-smi","--query-compute-apps=pid,used_memory",
                     "--format=csv,noheader,nounits"], text=True).strip()
                for line in out.splitlines():
                    pid_u, mem = line.split(", ")
                    rows.append({**row, "name":"<gpu-proc>", "pid": int(pid_u),
                                 "proc_vram_mb": int(mem)})
            except Exception: pass
        stop.wait(0.5)
    return rows

def agg_resources(rows, pids, with_gpu):
    res = {}
    for name in [n for _,n in pids] + (["<gpu>"] if with_gpu else []):
        sub = [r for r in rows if r.get("name")==name]
        if not sub: continue
        if name == "<gpu>":
            res[name] = {
                "util_mean": statistics.mean(r["util"] for r in sub),
                "util_max": max(r["util"] for r in sub),
                "vram_used_mb": sub[-1]["vram_used_mb"],
                "vram_total_mb": sub[-1]["vram_total_mb"],
            }
        else:
            res[name] = {
                "cpu_mean": statistics.mean(r["cpu"] for r in sub),
                "cpu_max": max(r["cpu"] for r in sub),
                "rss_peak_mb": max(r["rss_mb"] for r in sub),
            }
    # gpu per-proc
    procs = [r for r in rows if r.get("name")=="<gpu-proc>"]
    if procs:
        pid_to_name = {p:n for p,n in pids}
        res["gpu_per_proc"] = {}
        for pid in pid_to_name:
            sub = [r for r in procs if r["pid"]==pid]
            if sub:
                res["gpu_per_proc"][pid_to_name[pid]] = max(r["proc_vram_mb"] for r in sub)
    return res

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--ports", nargs="+", required=True, help="PORT:NAME ...")
    ap.add_argument("--pids", nargs="+", default=[], help="PID:NAME ...")
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--reps", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=8)
    args = ap.parse_args()

    models = [(int(p.split(":")[0]), p.split(":")[1]) for p in args.ports]
    pids = [(int(p.split(":")[0]), p.split(":")[1]) for p in args.pids]

    stop = threading.Event()
    rrows = []
    def _samp(): rrows.extend(sample_resources(pids, args.gpu, stop))
    rt = threading.Thread(target=_samp, daemon=True); rt.start()

    results = {}
    for port, name in models:
        print(f"\n>>> benchmarking {name} ({args.host}:{port}) ...", flush=True)
        results[name] = bench_model(args.host, port, args.reps, args.max_tokens)

    stop.set(); rt.join()
    print("\n" + "="*72)
    print(f"{'LATENCY BENCHMARK':^72}")
    print("="*72)
    print(f"prompts={len(PROMPTS)} reps={args.reps} max_tokens={args.max_tokens} "
          f"hardware={'GPU (L40)' if args.gpu else 'CPU'}")
    print("-"*72)
    hdr = f"{'Model':<14}{'n':>4}{'mean':>9}{'p50':>9}{'p95':>9}{'min':>8}{'max':>8}{'tok/s':>8}"
    print(hdr); print("-"*72)
    for port, name in models:
        s = summarize(results[name])
        print(f"{name:<14}{s['n']:>4}{s['wall_mean']:>8.0f}ms{s['wall_p50']:>7.0f}ms"
              f"{s['wall_p95']:>7.0f}ms{s['wall_min']:>6.0f}ms{s['wall_max']:>6.0f}ms"
              f"{s['tok_per_s']:>7.1f}")
        if s["errors"]:
            print(f"  (! {s['errors']} errors)")

    if rrows:
        print("\n" + "-"*72); print("RESOURCES (during benchmark)"); print("-"*72)
        rr = agg_resources(rrows, pids, args.gpu)
        for name, m in rr.items():
            if name == "<gpu>":
                print(f"  GPU: util mean={m['util_mean']:.0f}% max={m['util_max']}%  "
                      f"VRAM={m['vram_used_mb']}/{m['vram_total_mb']} MiB")
            elif name == "gpu_per_proc":
                print(f"  GPU per-process VRAM:")
                for proc_name, mb in m.items():
                    print(f"    {proc_name:<14}{mb:>7} MiB")
            else:
                print(f"  {name:<14}CPU mean={m['cpu_mean']:.0f}% max={m['cpu_max']:.0f}%  "
                      f"RSS peak={m['rss_peak_mb']:.0f} MiB")

    # dump json
    out = {
        "config": {"host": args.host, "reps": args.reps, "max_tokens": args.max_tokens, "gpu": args.gpu},
        "latency": {name: summarize(results[name]) for _, name in models},
        "resources": agg_resources(rrows, pids, args.gpu) if rrows else {},
    }
    fn = f"/tmp/bench_{int(time.time())}.json"
    with open(fn, "w") as f: json.dump(out, f, indent=2)
    print(f"\n(raw json: {fn})")

if __name__ == "__main__":
    main()
