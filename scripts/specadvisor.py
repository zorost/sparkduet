#!/usr/bin/env python3
"""
specadvisor.py, acceptance-adaptive DSpark draft depth, done honestly.

The ecosystem ships static num_speculative_tokens (k=5). Measured acceptance spans
0.58 (creative prose, where speculation is a net LOSS) to 0.91 (factual QA).
DeepSeek's own datacenter recipe uses k=7. A fixed k is provably suboptimal.

What this sidecar actually does (and does not) do:

  DOES   scrape the engine's real speculative-decoding counters
         (accepted/drafted token totals) and maintain a windowed acceptance rate;
  DOES   compute the throughput-optimal k from the acceptance curve, bounded by
         the captured cuda-graph set (seqs*(k+1) <= 48);
  DOES   log every decision and POST the recommendation to the router's
         /admin/spec-k, where operators and dashboards can read it;
  DOES NOT hot-swap k on a live engine. Draft depth is an engine-boot parameter:
         applying a recommendation = restart the lane with the new
         D_MTP_NUM_TOKENS at a quiet moment (sparkduetctl.sh restart). Any tool
         that claims to retune it live on stock vLLM is overselling.

Per-workload-class acceptance (code/math/prose/tool) comes from controlled runs:
`bench.py --suite spec` measures it with counter deltas per class. This watcher
measures the LIVE MIX you actually serve.

Usage:
  python3 scripts/specadvisor.py --once          # one reading + recommendation
  python3 scripts/specadvisor.py                 # watch loop (SPEC_WINDOW_S)
"""
from __future__ import annotations
import argparse, json, os, re, time, urllib.request
from pathlib import Path

WINDOW_S  = int(os.environ.get("SPEC_WINDOW_S", "300"))
K_SET     = sorted(int(x) for x in os.environ.get("SPEC_K_SET", "3,5,7").split(","))
MIN_TOK   = int(os.environ.get("SPEC_MIN_DRAFT_TOKENS", "2000"))
METRICS   = os.environ.get("SPEC_METRICS_URL", "http://127.0.0.1:30000/metrics")
ROUTER    = os.environ.get("SPEC_ROUTER_URL", "")
LOG       = Path(os.environ.get("SPEC_LOG", str(Path(__file__).parent.parent / "results" / "specadvisor-log.jsonl")))
MAX_SEQS  = int(os.environ.get("D_MAX_NUM_SEQS", "6"))
CURRENT_K = int(os.environ.get("D_MTP_NUM_TOKENS", "5"))

# Totals only: the *_created gauges and *_per_pos_total series that live next
# to these counters double the ratio if matched (see bench.py, same fix).
ACCEPT_PAT = re.compile(r'^vllm:(?:spec_decode_)?num_accepted_tokens_total\{[^}]*\}\s+([0-9.e+]+)', re.M)
DRAFT_PAT  = re.compile(r'^vllm:(?:spec_decode_)?num_draft_tokens_total\{[^}]*\}\s+([0-9.e+]+)', re.M)


def accepted_len(a: float, k: int) -> float:
    """Mean tokens emitted per verify step at acceptance a and draft depth k."""
    return (1.0 - a ** (k + 1)) / (1.0 - a) if a < 1.0 else float(k + 1)

# Each drafted token costs a verify-step fraction (compute + graph memory); deep
# drafts on low-acceptance traffic are net-negative. Objective: tokens per unit cost.
STEP_COST = float(os.environ.get("SPEC_STEP_COST", "0.08"))

def throughput_score(a: float, k: int) -> float:
    return accepted_len(a, k) / (1.0 + STEP_COST * k)

def allowed_k() -> list[int]:
    """cuda-graph capture size is seqs*(k+1); k must keep capture <= captured max (48)."""
    return [k for k in K_SET if MAX_SEQS * (k + 1) <= 48]

def read_counters() -> tuple[float, float] | None:
    try:
        with urllib.request.urlopen(METRICS, timeout=10) as r:
            text = r.read().decode()
    except Exception:
        return None
    acc = [float(v) for v in ACCEPT_PAT.findall(text)]
    dra = [float(v) for v in DRAFT_PAT.findall(text)]
    if not acc or not dra:
        return None
    return sum(acc), sum(dra)

def recommend(acceptance: float) -> dict:
    ks = allowed_k()
    best = max(ks, key=lambda k: throughput_score(acceptance, k)) if ks else CURRENT_K
    return {"acceptance": round(acceptance, 3), "current_k": CURRENT_K,
            "recommended_k": best,
            "expected_gain": round(throughput_score(acceptance, best) /
                                   max(1e-9, throughput_score(acceptance, CURRENT_K)) - 1, 3),
            "apply": f"set D_MTP_NUM_TOKENS={best} and `sparkduetctl.sh restart` at a quiet moment"
                     if best != CURRENT_K else "keep current k"}

def post_router(rec: dict) -> bool:
    if not ROUTER:
        return False
    try:
        req = urllib.request.Request(ROUTER.rstrip("/") + "/admin/spec-k",
                                     data=json.dumps(rec).encode(), method="POST",
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status == 200
    except Exception:
        return False

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true", help="single window, print and exit")
    args = ap.parse_args()
    LOG.parent.mkdir(parents=True, exist_ok=True)

    prev = read_counters()
    if prev is None:
        raise SystemExit("specadvisor: no speculative-decoding counters at "
                         f"{METRICS}, is DSpark speculation enabled? (fail-static: nothing to do)")
    while True:
        time.sleep(WINDOW_S if not args.once else min(WINDOW_S, 60))
        cur = read_counters()
        rec = {"ts": int(time.time()), "window_s": WINDOW_S}
        if cur is None:
            rec["status"] = "metrics-unavailable (fail-static)"
        else:
            d_acc, d_dra = cur[0] - prev[0], cur[1] - prev[1]
            prev = cur
            if d_dra < MIN_TOK:
                rec["status"] = f"insufficient traffic ({int(d_dra)} draft tokens < {MIN_TOK})"
            else:
                rec.update(recommend(d_acc / d_dra))
                rec["posted_to_router"] = post_router(rec)
        with LOG.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)
        if args.once:
            break

if __name__ == "__main__":
    main()
