#!/usr/bin/env python3
"""CPU-only check that Lane G is wired in env, compose, and ctl."""
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
compose = (root / "configs/lane-glm.compose.yml").read_text()

check("compose exists", (root / "configs/lane-glm.compose.yml").is_file())
check("G_MODEL in env example", "G_MODEL=" in env)
check("dedicated glm53 image", "glm53-flash-arm64-cu130" in env)
check("ctl starts glm", "start_glm()" in ctl)
check("glm image guard", "head missing image $G_VLLM_IMAGE" in ctl)
check("ctl switch accepts glm", "[depth|fleet|next|glm|split]" in ctl)
check("prepare-models glm-flash", "glm-flash" in prep)
commands = "\n".join(line for line in compose.splitlines() if not line.lstrip().startswith("#"))
check("no dummy load format", "--load-format dummy" not in commands)
check("glm parsers", "--tool-call-parser glm47" in compose and "--reasoning-parser glm45" in compose)
check("uses G_VLLM_IMAGE", "${G_VLLM_IMAGE}" in compose)
check("no house DeepSeek image", "${VLLM_IMAGE}" not in compose)
entry = (root / "patches/glm-entry.sh").read_text()
check("no Mia credit", "Mia" not in compose and "Maya" not in compose)
check("entry pins FlashInfer 0.6.18", "0.6.18.dev20260819" in entry)
check("entry pins NCCL 2.30.7", "nvidia-nccl-cu13==2.30.7" in entry)

raise SystemExit(fails)
