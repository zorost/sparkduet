#!/usr/bin/env python3
"""CPU-only unit tests for specadvisor.py control math."""
import os, sys, importlib.util

os.environ["SPEC_K_SET"] = "3,5,7,8"
os.environ["D_MAX_NUM_SEQS"] = "6"
os.environ["D_MTP_NUM_TOKENS"] = "5"
spec = importlib.util.spec_from_file_location(
    "specadvisor", os.path.join(os.path.dirname(__file__), "specadvisor.py"))
sa = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sa)

fails = 0
def check(name, cond):
    global fails
    print(("PASS" if cond else "FAIL"), name)
    fails += 0 if cond else 1

# accepted length is monotonic in acceptance and in k (for a>0)
check("accepted_len monotone in a", sa.accepted_len(0.9, 5) > sa.accepted_len(0.6, 5))
check("accepted_len monotone in k", sa.accepted_len(0.8, 7) > sa.accepted_len(0.8, 3))
check("accepted_len a=0 is 1.0", abs(sa.accepted_len(0.0, 5) - 1.0) < 1e-9)
check("accepted_len a=1 is k+1", abs(sa.accepted_len(1.0, 5) - 6.0) < 1e-9)

# known anchor from the ecosystem: a=0.6,k=5 ≈ 2.38 (the measured 2.4x band)
check("a=0.6 k=5 ≈ 2.38", abs(sa.accepted_len(0.6, 5) - 2.38) < 0.02)

# allowed_k respects the cuda-graph capture bound seqs*(k+1) <= 48
check("allowed_k bounded", sa.allowed_k() == [3, 5, 7])   # 6*(8+1)=54 > 48 → 8 excluded

# recommendation direction: high acceptance -> deepest allowed k, low -> shallowest
check("a=0.90 -> k=7", sa.recommend(0.90)["recommended_k"] == 7)
check("a=0.40 -> k=3", sa.recommend(0.40)["recommended_k"] == 3)

# a recommendation equal to the current k says "keep", a different one says "restart"
rec_same = sa.recommend(0.72)          # mid acceptance
if rec_same["recommended_k"] == sa.CURRENT_K:
    check("keep-current wording", rec_same["apply"] == "keep current k")
else:
    check("restart wording", "restart" in rec_same["apply"])

# expected_gain is 0 when recommended == current
rec = sa.recommend(0.9)
if rec["recommended_k"] == sa.CURRENT_K:
    check("gain zero at same k", abs(rec["expected_gain"]) < 1e-9)
else:
    check("gain positive at better k", rec["expected_gain"] > 0)

# counter regexes: totals only, and the neighbor series must NOT match.
# (*_created timestamp gauges and *_per_pos_total breakdowns double the ratio;
# that bug shipped once, on 2026-08-25, and this test is its tombstone.)
sample = ('vllm:spec_decode_num_accepted_tokens_total{engine="0"} 1200\n'
          'vllm:spec_decode_num_accepted_tokens_created{engine="0"} 1.7876e+09\n'
          'vllm:spec_decode_num_accepted_tokens_per_pos_total{engine="0",position="0"} 800\n'
          'vllm:spec_decode_num_draft_tokens_total{engine="0"} 2000\n'
          'vllm:spec_decode_num_draft_tokens_created{engine="0"} 1.7876e+09\n'
          'vllm:spec_decode_num_drafts_total{engine="0"} 400\n')
acc = [float(v) for v in sa.ACCEPT_PAT.findall(sample)]
dra = [float(v) for v in sa.DRAFT_PAT.findall(sample)]
check("accept counter parsed", acc == [1200.0])
check("draft counter parsed", dra == [2000.0])

sys.exit(1 if fails else 0)
