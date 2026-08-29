#!/usr/bin/env python3
"""
bench.py, the SparkDuet honesty harness. Implements docs/BENCHMARK-PROTOCOL.md.

Protocol compliance, in code rather than in promises:
  rule 1  throughput = usage.completion_tokens / wall_time. Tokens come from the
          final SSE usage frame (stream_options.include_usage). SSE chunks are
          counted only to DETECT chunk-counting mistakes, never to report.
  rule 2  acceptance is captured as a before/after delta of the engine's
          speculative-decoding counters over the same window as the cell.
  rule 3  every cell carries (prompt_tokens, concurrency, thinking).
  rule 5  TTFT p50/p95 measured from the streaming first content chunk.
  rule 7  refuses to publish runs shorter than 30 s.

Usage:
  python3 scripts/bench.py --suite standard --output results/
  python3 scripts/bench.py --base-url http://127.0.0.1:30000/v1 --model X --suite spec
  python3 scripts/bench.py --suite quick --prompts 256,2048 --conc 1,2 --trials 2
"""
from __future__ import annotations
import argparse, asyncio, json, re, shutil, statistics, subprocess, time, urllib.request
from pathlib import Path

SUITES = {
    "standard":   {"prompts": [256, 2048, 8192, 32768, 131072], "conc": [1, 2, 4, 6],
                   "gen": 128, "thinking": False},
    "quick":      {"prompts": [256, 2048, 8192], "conc": [1, 2, 4], "gen": 128,
                   "thinking": False},
    "fleet":      {"prompts": [512, 4096], "conc": [2, 4, 8, 12, 16], "gen": 256,
                   "thinking": False, "stagger_s": 0.4},
    "mixed-long": {"prompts": [32768, 131072], "conc": [6], "gen": 128,
                   "thinking": False},
    "spec":       {"conc": [1], "gen": 384, "thinking": False, "classes": True},
}

# Workload classes for the spec suite. Distinct enough that draft acceptance
# genuinely differs (code/math accept high, prose low, the SpecAdvisor premise).
CLASS_PROMPTS = {
    "code":  "Write a complete Python module implementing a thread-safe LRU cache "
             "with TTL eviction, full type hints, and docstrings. Code only.",
    "math":  "Compute step by step, showing every intermediate result: the sum of "
             "squares of the first 40 integers, then its prime factorization.",
    "prose": "Write an imaginative short story about a lighthouse keeper who "
             "discovers the fog itself is alive. Vivid, unpredictable language.",
    "tool":  "You have a tool `search(query: str) -> str`. Produce exactly five "
             "JSON tool calls, one per line, planning research about tidal energy.",
}

# Exactly the *_total counters. The metric family also exposes *_created
# (a unix-timestamp gauge) and *_per_pos_total (per-position breakdown that
# sums to the total); matching those doubles the acceptance ratio. Measured
# that mistake here first: results/2026-08-25 standard artifact, corrected.
# Totals only. *_created gauges and *_per_pos_total series sit next to these
# and double the ratio if matched (the 2026-08-25 bug). SGLang NEXTN uses the
# sglang: prefix; vLLM uses vllm:. Either engine can serve a lane.
ACCEPT_PAT = re.compile(
    r'^(?:vllm|sglang):(?:spec_decode_)?num_accepted_tokens_total\{[^}]*\}\s+([0-9.e+]+)',
    re.M,
)
DRAFT_PAT = re.compile(
    r'^(?:vllm|sglang):(?:spec_decode_)?num_draft_tokens_total\{[^}]*\}\s+([0-9.e+]+)',
    re.M,
)
SGLANG_ACCEPT_LEN = re.compile(r'^sglang:spec_accept_length\{[^}]*\}\s+([0-9.e+]+)', re.M)
SGLANG_ACCEPT_RATE = re.compile(r'^sglang:spec_accept_rate\{[^}]*\}\s+([0-9.e+]+)', re.M)


def synth_prompt(n_tokens: int, salt: str) -> str:
    """Unique cold prefix of ~n tokens (~4 chars/token) ending in a real task."""
    body = (f"{salt} " + "lorem ipsum dolor sit amet " * (n_tokens // 6 + 1))[: n_tokens * 4 - 128]
    return body + "\n\nSummarize the above in exactly three sentences."


def http_get(url: str, timeout: int = 10) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read().decode()


def scrape_spec_counters(metrics_url: str) -> tuple[float, float] | None:
    """(accepted_total, drafted_total) from the engine's Prometheus metrics."""
    try:
        text = http_get(metrics_url)
    except Exception:
        return None
    acc = [float(v) for v in ACCEPT_PAT.findall(text)]
    dra = [float(v) for v in DRAFT_PAT.findall(text)]
    if acc and dra:
        return (sum(acc), sum(dra))
    # SGLang NEXTN often exposes accept rate / length gauges instead of the
    # vLLM-style running totals. Encode rate as a synthetic (rate, 1.0) pair
    # so the existing delta math still yields the rate when the gauge moves.
    rates = [float(v) for v in SGLANG_ACCEPT_RATE.findall(text)]
    if rates:
        return (sum(rates) / len(rates), 1.0)
    return None


def capture_environment() -> dict:
    """Clocks and GPU identity when running on the node (protocol: record clocks)."""
    env: dict = {}
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,clocks.sm,clocks.max.sm",
                 "--format=csv,noheader"],
                capture_output=True, text=True, timeout=10).stdout.strip()
            env["gpu"] = out
        except Exception:
            pass
    return env


async def one_request(base: str, model: str, prompt: str, gen: int, thinking: bool):
    """Streaming request. Returns wall time, TTFT, and usage-counted tokens."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": gen, "min_tokens": gen, "ignore_eos": True,
        "temperature": 0.0,
        # Flash-Next / Qwen4 reads enable_thinking (thinking on by default).
        # DeepSeek V4 reads thinking. Send both so a suite that says "off"
        # actually turns thinking off on whichever lane is resident.
        "chat_template_kwargs": {"thinking": thinking, "enable_thinking": thinking},
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    def _call():
        req = urllib.request.Request(base + "/chat/completions",
                                     data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        t0 = time.perf_counter()
        ttft = None
        chunks = 0
        usage = {}
        with urllib.request.urlopen(req, timeout=7200) as r:
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if obj.get("usage"):
                    usage = obj["usage"]
                for ch in obj.get("choices", []):
                    delta = ch.get("delta") or {}
                    if delta.get("content") or delta.get("reasoning_content"):
                        chunks += 1
                        if ttft is None:
                            ttft = time.perf_counter() - t0
        return time.perf_counter() - t0, ttft, chunks, usage

    wall, ttft, chunks, usage = await asyncio.to_thread(_call)
    ct = usage.get("completion_tokens", 0)
    if ct <= 0:
        raise RuntimeError("no usage.completion_tokens in the final stream frame, "
                           "refusing to measure (protocol rule 1)")
    return {"wall_s": wall, "ttft_s": ttft, "sse_chunks": chunks,
            "completion_tokens": ct, "tok_s": ct / wall,
            "prompt_tokens": usage.get("prompt_tokens", 0)}


def pctl(vals: list[float], q: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    idx = min(len(s) - 1, max(0, round(q * (len(s) - 1))))
    return s[idx]


async def run_cell(base, metrics_url, model, prompt_maker, n_prompt, conc, gen,
                   thinking, trials, stagger_s=0.0, label=None):
    rates, ttfts, all_reqs = [], [], []
    spec0 = scrape_spec_counters(metrics_url) if metrics_url else None
    for t in range(trials):
        jobs = [one_request(base, model,
                            prompt_maker(n_prompt, f"cell{label or n_prompt}-c{conc}-t{t}-r{i}"),
                            gen, thinking) for i in range(conc)]
        if stagger_s:
            async def staggered(coro, i):
                await asyncio.sleep(i * stagger_s)
                return await coro
            jobs = [staggered(c, i) for i, c in enumerate(jobs)]
        res = await asyncio.gather(*jobs)
        all_reqs.extend(res)
        per_stream = [r["tok_s"] for r in res]
        rates.append({"aggregate_tok_s": round(sum(r["completion_tokens"] for r in res) /
                                               max(r["wall_s"] for r in res), 1),
                      "per_stream_median": round(statistics.median(per_stream), 1)})
        ttfts.extend(r["ttft_s"] for r in res if r["ttft_s"] is not None)
    spec1 = scrape_spec_counters(metrics_url) if metrics_url else None

    acceptance = None
    if spec0 and spec1:
        d_acc, d_dra = spec1[0] - spec0[0], spec1[1] - spec0[1]
        if d_dra > 0:
            acceptance = round(d_acc / d_dra, 3)
        elif spec1[1] == 1.0 and spec1[0] > 0:
            # SGLang accept-rate gauge fallback (encoded as (rate, 1.0)).
            acceptance = round(spec1[0], 3)

    # chunk-counting tripwire (protocol sanity check): under speculation, tokens
    # per SSE chunk should be well above 1. Ratio ~1 with speculation active means
    # someone is counting chunks somewhere.
    toks = sum(r["completion_tokens"] for r in all_reqs)
    chunks = sum(r["sse_chunks"] for r in all_reqs) or 1
    med = lambda key: statistics.median(r[key] for r in rates)
    return {"prompt_tokens_nominal": n_prompt,
            "prompt_tokens_actual": all_reqs[0]["prompt_tokens"] if all_reqs else None,
            "concurrency": conc, "thinking": thinking,
            "per_stream_tok_s": round(med("per_stream_median"), 1),
            "aggregate_tok_s": round(med("aggregate_tok_s"), 1),
            "ttft_p50_s": round(pctl(ttfts, 0.50), 3) if ttfts else None,
            "ttft_p95_s": round(pctl(ttfts, 0.95), 3) if ttfts else None,
            "acceptance": acceptance,
            "tokens_per_sse_chunk": round(toks / chunks, 2),
            "class": label}


async def amain():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:30000/v1")
    ap.add_argument("--metrics-url", default=None,
                    help="engine Prometheus endpoint; default derives <base>/../metrics")
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-V4-Flash-0731")
    ap.add_argument("--lane", default="auto")
    ap.add_argument("--suite", default="standard", choices=list(SUITES))
    ap.add_argument("--output", default="results")
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--prompts", default=None, help="override: comma-separated prompt sizes")
    ap.add_argument("--conc", default=None, help="override: comma-separated concurrencies")
    ap.add_argument("--gen", type=int, default=None, help="override: generation length")
    ap.add_argument("--require-acceptance", action="store_true",
                    help="refuse to publish when spec counters are unavailable")
    args = ap.parse_args()

    s = dict(SUITES[args.suite])
    if args.prompts:
        s["prompts"] = [int(x) for x in args.prompts.split(",")]
    if args.conc:
        s["conc"] = [int(x) for x in args.conc.split(",")]
    if args.gen:
        s["gen"] = args.gen

    metrics_url = args.metrics_url
    if metrics_url is None and args.base_url.endswith("/v1"):
        metrics_url = args.base_url[:-3] + "/metrics"

    cells = []
    t_start = time.time()
    if s.get("classes"):  # spec suite: per-class acceptance, fixed short prompts
        for cls, prompt in CLASS_PROMPTS.items():
            maker = lambda _n, salt, p=prompt: f"[{salt}] {p}"
            cells.append(await run_cell(args.base_url, metrics_url, args.model, maker,
                                        0, 1, s["gen"], s["thinking"], args.trials,
                                        label=cls))
    else:
        for n_prompt in s["prompts"]:
            for conc in s["conc"]:
                cells.append(await run_cell(args.base_url, metrics_url, args.model,
                                            synth_prompt, n_prompt, conc, s["gen"],
                                            s["thinking"], args.trials,
                                            s.get("stagger_s", 0.0)))
    elapsed = time.time() - t_start
    if elapsed < 30:
        raise SystemExit("run < 30 s, no steady state, refusing to publish (protocol rule 7)")
    if args.require_acceptance and all(c["acceptance"] is None for c in cells):
        raise SystemExit("no acceptance counters found and --require-acceptance set "
                         "(protocol rule 2)")

    artifact = {"lane": args.lane, "suite": args.suite, "model": args.model,
                "trials": args.trials, "elapsed_s": round(elapsed, 1),
                "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "environment": capture_environment(), "cells": cells}

    stamp = time.strftime("%Y-%m-%d", time.gmtime())
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    jf = out / f"{stamp}-{args.lane}-{args.suite}.json"
    jf.write_text(json.dumps(artifact, indent=2))

    md = [f"### [M-here] {args.lane} / {args.suite}, {stamp}",
          "| Prompt tok | Conc | Thinking | Per-stream tok/s | Aggregate tok/s | "
          "TTFT p50 s | TTFT p95 s | Acceptance |",
          "|---:|---:|:---:|---:|---:|---:|---:|---:|"]
    for c in cells:
        pt = c["class"] or c["prompt_tokens_actual"] or c["prompt_tokens_nominal"]
        md.append(f"| {pt} | {c['concurrency']} | {'on' if c['thinking'] else 'off'} | "
                  f"{c['per_stream_tok_s']} | {c['aggregate_tok_s']} | "
                  f"{c['ttft_p50_s'] if c['ttft_p50_s'] is not None else '-'} | "
                  f"{c['ttft_p95_s'] if c['ttft_p95_s'] is not None else '-'} | "
                  f"{c['acceptance'] if c['acceptance'] is not None else '-'} |")
    mf = out / f"{stamp}-{args.lane}-{args.suite}.md"
    mf.write_text("\n".join(md) + "\n")
    print(f"wrote {jf} and {mf}")

if __name__ == "__main__":
    asyncio.run(amain())
