#!/usr/bin/env python3
"""CPU-only check that Lane N is wired in env, compose, and ctl."""
from pathlib import Path

root = Path(__file__).resolve().parent.parent
fails = 0


def check(name, cond):
    global fails
    print(("PASS" if cond else "FAIL"), name)
    fails += 0 if cond else 1


env = (root / "configs/sparkduet.env.example").read_text()
ctl = (root / "scripts/sparkduetctl.sh").read_text()
prep = (root / "scripts/prepare-models.sh").read_text()
compose = (root / "configs/lane-next.compose.yml").read_text()

check("compose exists", (root / "configs/lane-next.compose.yml").is_file())
check("N_MODEL in env example", "N_MODEL=" in env)
check("N_MAX_NUM_BATCHED_TOKENS=1024", "N_MAX_NUM_BATCHED_TOKENS=1024" in env)
check("ctl starts next", "start_next()" in ctl)
check("ctl switch accepts next", "[depth|fleet|next|glm|split]" in ctl)
check("prepare-models flash-next", "flash-next" in prep)
commands = "\n".join(line for line in compose.splitlines() if not line.lstrip().startswith("#"))
check("no dummy load format", "--load-format dummy" not in commands)
check("qwen parsers", "--tool-call-parser qwen3_coder" in compose and "--reasoning-parser qwen3" in compose)
check("no dspark spec on Lane N", "dspark" not in compose)
check("next-entry", "next-entry.sh" in compose)
check("PLE patch", (root / "patches/next-ple-fp8.py").is_file())
check("modelopt_fp4", "--quantization modelopt_fp4" in commands)
check("eager GB10", "--enforce-eager" in commands)
check("no PLE CPU offload", "VLLM_PLE_CPU_OFFLOAD" not in commands)

raise SystemExit(fails)
