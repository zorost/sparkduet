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
check("ctl unloads library before glm", "api/models/unload" in ctl)
check("glm image guard", "head missing image $gimg" in ctl)
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

# ---- Lane G on EXL3 (G_ENGINE=exl3) ----------------------------------------
x_path = root / "configs/lane-glm-exl3.compose.yml"
x = x_path.read_text()
x_entry_path = root / "patches/glm-exl3-entry.sh"
x_entry = x_entry_path.read_text()
patch_dir = root / "patches/glm-exl3-sm121"
x_cmd = "\n".join(l for l in x.splitlines() if not l.lstrip().startswith("#"))

check("exl3 compose exists", x_path.is_file())
check("ctl selects engine", 'case "${G_ENGINE:-vllm}"' in ctl)
check("ctl maps exl3 to its lane file", "lane-glm-exl3.compose.yml" in ctl)
check("ctl rejects a bad engine", "G_ENGINE must be vllm or exl3" in ctl)
check("stop_all covers exl3", "lane-glm lane-glm-exl3" in ctl)
check("warmup asks the exl3 id", "G_EXL3_SERVED_NAME" in ctl)
check("prepare-models glm-exl3", "stage_glm_exl3" in prep and "glm-exl3|exl3" in prep)
check("staging pins the scored revision", "5ab363a8dcf6405955fd5f99671e01a1c9fb124b" in prep)
check("staging verifies 120 shards", 'found $shards' in prep)
check("G_ENGINE in env example", "G_ENGINE=vllm" in env)
check("exl3 env block", "G_EXL3_MODEL=" in env and "G_EXL3_IMAGE=" in env)

# The two lanes must stay swappable: same port, same container names, so
# 9Router, sparkduetctl and zorost-lane-guard need no special case.
check("exl3 serves the same port", "--port ${G_PORT}" in x)
check("exl3 keeps house container names",
      "container_name: sparkduet-glm-head" in x
      and "container_name: sparkduet-glm-worker" in x)
check("exl3 ranks never self-restart", x.count('restart: "no"') >= 1)

# MiaAI-Lab's contract for this checkpoint.
check("exl3 quantization declared", "--quantization exl3" in x_cmd)
check("no marlin MoE backend", "marlin" not in x_cmd)
check("graphs stay on", "--enforce-eager" not in x_cmd)
check("MTP capture set includes 3", 'G_EXL3_CUDAGRAPH_SIZES="1 2 3 4 6 8 12"' in env)
# sparkduet.env is sourced by bash, so any value with spaces must be quoted or
# the shell runs the second word as a command
for line in env.splitlines():
    if line.startswith("G_EXL3_") and " " in line.split("=", 1)[1].split("#")[0].strip():
        check(f"quoted multiword value: {line.split('=')[0]}",
              line.split("=", 1)[1].lstrip().startswith(('"', "'")))
check("speculator is MTP", '"method":"mtp"' in x_cmd)
check("no DFlash2 anywhere", "dflash" not in x_cmd.lower() and "dflash" not in x_entry.lower())
check("packed fp8 KV, not NVFP4", "G_EXL3_KV_DTYPE=fp8" in env and "nvfp4" not in x_cmd.lower())
check("glm parsers on exl3", "--tool-call-parser glm47" in x and "--reasoning-parser glm45" in x)

# The published image predates these patches; the entry script must mount and
# apply them, and must refuse to serve without the prefix-cache fix.
for p in ("patch_hybrid_prefix_hit", "patch_scheduler_decode_floor",
          "patch_suppress_stops_in_reasoning", "patch_clamp_max_tokens"):
    check(f"{p} vendored", (patch_dir / f"{p}.py").is_file())
    check(f"{p} applied by entry", p in x_entry)
check("entry hard-fails on a missing patch", "refusing to serve" in x_entry)
check("entry execs vllm serve", 'exec vllm serve "$@"' in x_entry)
check("entry is executable", x_entry_path.stat().st_mode & 0o111 != 0)
check("patches mounted into the container", "../patches:/sparkduet-patches:ro" in x)

# Attribution is a condition of the ShapleyMcg grant, not a courtesy.
readme = (patch_dir / "README.md").read_text()
# the notice is quoted prose, so it wraps; compare on collapsed whitespace
readme_flat = " ".join(readme.replace(">", " ").split())
check("shapleymcg license vendored", (patch_dir / "LICENSE.shapleymcg-1.0").is_file())
check("MiaAI-Lab MIT license vendored", (patch_dir / "LICENSE.MiaAI-Lab").is_file())
check("required attribution notice present",
      "ShapleyMcg, created by Brandon M. Music" in readme_flat
      and "Use of ShapleyMcg without this attribution is unlicensed." in readme_flat)
check("canonical repository linked", "github.com/brandonmmusic-max/shapleymcg" in readme)
check("MiaAI-Lab credited for the overlay", "MiaAI-Lab" in readme)
check("compose points at the license", "ShapleyMcg" in x)

raise SystemExit(fails)
