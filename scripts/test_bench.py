#!/usr/bin/env python3
"""CPU-only unit tests for bench.py helpers (no GPU, no server)."""
import os, sys, importlib.util

spec = importlib.util.spec_from_file_location(
    "bench", os.path.join(os.path.dirname(__file__), "bench.py"))
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)

fails = 0
def check(name, cond):
    global fails
    print(("PASS" if cond else "FAIL"), name)
    fails += 0 if cond else 1

# synth prompts: unique per salt (prefix-cache poison) and roughly sized
a = bench.synth_prompt(2048, "salt-a")
b = bench.synth_prompt(2048, "salt-b")
check("prompts unique per salt", a != b)
check("prompt sized ~4 chars/token", 0.5 < len(a) / (2048 * 4) <= 1.0)
check("prompt ends with a task", a.rstrip().endswith("three sentences."))

# percentile helper: exact anchors
vals = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
check("p50 of 1..10", abs(bench.pctl(vals, 0.50) - 0.5) < 0.11)
check("p95 of 1..10", bench.pctl(vals, 0.95) >= 0.9)
check("pctl empty is 0", bench.pctl([], 0.5) == 0.0)

# Prometheus counter parsing (same patterns as specadvisor). The sample
# includes the three trap series a real engine exposes alongside the totals:
# *_created unix-timestamp gauges and *_per_pos_total breakdowns. Matching
# any of them double-counts acceptance (the exact bug shipped on 2026-08-25).
sample = (
    'vllm:spec_decode_num_accepted_tokens_total{e="0"} 900\n'
    'vllm:spec_decode_num_accepted_tokens_created{e="0"} 1.787625626e+09\n'
    'vllm:spec_decode_num_accepted_tokens_per_pos_total{e="0",position="0"} 600\n'
    'vllm:spec_decode_num_accepted_tokens_per_pos_total{e="0",position="1"} 300\n'
    'vllm:spec_decode_num_draft_tokens_total{e="0"} 1500\n'
    'vllm:spec_decode_num_draft_tokens_created{e="0"} 1.787625626e+09\n'
    'vllm:spec_decode_num_drafts_total{e="0"} 300\n'
)
acc = [float(v) for v in bench.ACCEPT_PAT.findall(sample)]
dra = [float(v) for v in bench.DRAFT_PAT.findall(sample)]
check("acceptance totals only, traps excluded", acc == [900.0] and dra == [1500.0])

# suites are structurally sound
for name, s in bench.SUITES.items():
    ok = ("conc" in s and "gen" in s and ("prompts" in s or s.get("classes")))
    check(f"suite {name} well-formed", ok)

# class prompts exist for the spec suite
check("spec classes defined", set(bench.CLASS_PROMPTS) == {"code", "math", "prose", "tool"})

sys.exit(1 if fails else 0)
