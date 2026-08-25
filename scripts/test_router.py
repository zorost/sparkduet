#!/usr/bin/env python3
"""CPU-only unit tests for router.py lane selection and validation."""
import os, sys, importlib.util

os.environ.update({"ROUTER_PORT": "18888", "P_PROMPT_THRESHOLD": "8192",
                   "D_HIGH_WATER_SEQS": "5", "MASTER_ADDR": "127.0.0.1",
                   "WORKER_HOST": "127.0.0.2",
                   "LANE_SPLIT_ENABLE": "1", "LANE_FLEET_ENABLE": "1"})
spec = importlib.util.spec_from_file_location("router", os.path.join(os.path.dirname(__file__), "router.py"))
router = importlib.util.module_from_spec(spec)
spec.loader.exec_module(router)

fails = 0
def check(name, got, want):
    global fails
    ok = got == want
    print(("PASS" if ok else "FAIL"), name, f"(got={got!r} want={want!r})")
    fails += 0 if ok else 1

class H(dict):
    def get(self, k, d=None): return super().get(k, d)

# 1. explicit header pins the lane
p = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
check("header pin fleet", router.choose_lane(dict(p), H({"X-SparkDuet-Lane": "fleet"}))[0], "fleet")

# 2. model suffix pins and rewrites
p2 = {"model": "m@split", "messages": [{"role": "user", "content": "hi"}]}
lane, _ = router.choose_lane(p2, H())
check("model suffix pin", lane, "split")
check("model suffix rewrite", p2["model"], "m")

# 3. long prompt routes to split (only because LANE_SPLIT_ENABLE=1)
big = {"model": "m", "messages": [{"role": "user", "content": "x" * 40000}]}
check("long prompt -> split", router.choose_lane(big, H())[0], "split")

# 4. default short prompt -> depth
check("short prompt -> depth", router.choose_lane(dict(p), H())[0], "depth")

# 5. spill: saturate depth inflight beyond high water
for _ in range(6):
    router.STATE.pick("depth")
check("spill to fleet at high water", router.choose_lane(dict(p), H())[0], "fleet")
for _ in range(6):
    router.STATE.release("depth")
check("back to depth after release", router.choose_lane(dict(p), H())[0], "depth")

# 6. disabled lanes never receive implicit traffic
os.environ["LANE_SPLIT_ENABLE"] = "0"
os.environ["LANE_FLEET_ENABLE"] = "0"
check("split disabled -> depth", router.choose_lane(dict(big), H())[0], "depth")
for _ in range(6):
    router.STATE.pick("depth")
check("fleet disabled -> stay depth", router.choose_lane(dict(p), H())[0], "depth")
for _ in range(6):
    router.STATE.release("depth")
os.environ["LANE_SPLIT_ENABLE"] = "1"
os.environ["LANE_FLEET_ENABLE"] = "1"

# 7. saturation returns None (mapped to a typed 429 by the handler)
router.STATE.inflight["depth"] = router.LANE_MAX
check("lane saturation refuses", router.STATE.pick("depth"), None)
router.STATE.inflight["depth"] = 0

# 8. fleet balances to the least-loaded replica
router.STATE.inflight["fleet-A"] = 3
router.STATE.inflight["fleet-B"] = 1
host, port, label = router.STATE.pick("fleet")
check("fleet least-loaded", label, "fleet-B")
router.STATE.release(label)
router.STATE.inflight["fleet-A"] = 0
router.STATE.inflight["fleet-B"] = 0

# 9. token estimate monotonicity
check("token estimate", router.count_prompt_tokens(big) > router.count_prompt_tokens(p), True)

# 10. non-numeric env rejected at import time (e.g. a float string)
os.environ["ROUTER_PORT"] = "8.5"
try:
    spec2 = importlib.util.spec_from_file_location("router2", os.path.join(os.path.dirname(__file__), "router.py"))
    r2 = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(r2)
    check("non-numeric port rejected", "no error", "SystemExit")
except SystemExit:
    check("non-numeric port rejected", "SystemExit", "SystemExit")

sys.exit(1 if fails else 0)
