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
check("compose is house-owned", "${G_VLLM_IMAGE}" in compose)
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
check("ctl rejects a bad engine", "G_ENGINE must be vllm, exl3 or tensorfold" in ctl)
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

# House contract for this checkpoint.
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
ignore = (root / ".gitignore").read_text()
check("restricted overlay is gitignored", "local/restricted/" in ignore)
ctl_maps = "compose_rel" in ctl and "local/restricted/" in ctl
check("ctl prefers a local/restricted compose when present", ctl_maps)
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
check("required attribution notice present",
      "ShapleyMcg, created by Brandon M. Music" in readme_flat
      and "Use of ShapleyMcg without this attribution is unlicensed." in readme_flat)
check("canonical repository linked", "github.com/brandonmmusic-max/shapleymcg" in readme)
check("EXL3 readme stays first-party", "ShapleyMcg" in readme)
check("compose points at the license", "ShapleyMcg" in x)

# ---- Lane G on TensorFold (G_ENGINE=tensorfold) ------------------------------
import json, subprocess, sys, tempfile

t_path = root / "configs/lane-glm-tensorfold.compose.yml"
t = t_path.read_text()
t_cmd = "\n".join(l for l in t.splitlines() if not l.lstrip().startswith("#"))
t_entry_path = root / "patches/glm-tensorfold-entry.sh"
t_entry = t_entry_path.read_text()
t_docker = (root / "patches/glm-tensorfold/Dockerfile").read_text()
t_build = root / "scripts/build-tensorfold-image.sh"

check("tensorfold compose exists", t_path.is_file())
check("ctl maps tensorfold to its lane file", "tensorfold) gfile=lane-glm-tensorfold.compose.yml" in ctl)
check("stop_all covers tensorfold", "lane-glm lane-glm-exl3 lane-glm-tensorfold" in ctl)
check("tensorfold warmup asks the exl3 id", 'G_ENGINE:-vllm}" == tensorfold' in ctl and "D_MAX_NUM_SEQS=\"$G_TF_PARALLEL\"" in ctl)
check("ctl writes the end-token overlay on both nodes",
      ctl.count("glm_eos_overlay.py") >= 2 and "different end-token overlays" in ctl)
check("ctl requires one image id on both ranks", "different $gimg images" in ctl)
check("G_TF block in env example", all(k in env for k in (
      "G_TF_IMAGE=", "G_TF_CONTEXT=", "G_TF_PARALLEL=", "G_TF_KV=", "G_TF_PROMPT_CACHE_GIB=",
      "G_TF_MAX_TOKENS=", "G_TF_MEMORY_RESERVE_GIB=", "G_TF_CLEAR_THINKING=", "G_TF_EXTRA_EOS=", "G_TF_STATE_DIR=")))
check("default engine unchanged", "\nG_ENGINE=vllm\n" in env)
eos_line = next(l for l in env.splitlines() if l.startswith("G_TF_EXTRA_EOS="))
check("extra-EOS value is quoted for bash", eos_line.split("=", 1)[1].startswith("'"))

# Same hop and names as the other GLM engines.
check("tensorfold serves the same port", "--port ${G_PORT}" in t_cmd)
check("tensorfold keeps house container names",
      "container_name: sparkduet-glm-head" in t and "container_name: sparkduet-glm-worker" in t)
check("tensorfold ranks never self-restart", 'restart: "no"' in t)
check("tensorfold serves the EXL3 checkpoint and id",
      "${G_EXL3_MODEL}:/model:ro" in t and "--name ${G_EXL3_SERVED_NAME}" in t_cmd)
check("tensorfold ranks are placed by env", 'TF_GLM_RANK: "0"' in t and 'TF_GLM_RANK: "1"' in t
      and 'TF_GLM_WORLD: "2"' in t and "NCCL_COMM_ID: ${MASTER_ADDR}:${MASTER_PORT}" in t)
check("end-token overlay mounted over the checkpoint's",
      "/generation_config.json:/model/generation_config.json:ro" in t)

# License-clean drafting: MTP head only.
check("tensorfold drafts with MTP only", "--drafter none" in t_cmd and 'TF_GLM_MTP: "1"' in t)
check("no drafter directory or bits on tensorfold",
      "--drafter-bits" not in t_cmd and "--drafter /" not in t_cmd and "dflash" not in t.lower())
check("tensorfold entry refuses another drafter",
      "--drafter must be none" in t_entry and "exactly one '--drafter none'" in t_entry)
check("tensorfold entry has no speculator names", "dflash" not in t_entry.lower())
check("tensorfold entry execs serve", 'exec "$BIN" serve /model "$@"' in t_entry)
check("tensorfold entry is executable", t_entry_path.stat().st_mode & 0o111 != 0)
check("thinking off at the door, no default effort", "--no-thinking" in t_cmd and "--reasoning-effort" not in t_cmd)

# Image: the pinned upstream release, verified, no engine patches.
check("image pins the release and its sha256",
      "ARG TF_VERSION=1.3.0" in t_docker
      and "718467e07007c566153bcddbd60b311709dfa8e6e98fab3c67330e0e3052987c" in t_docker
      and 'sha256sum -c -' in t_docker and "sha256sum -c SHA256SUMS" in t_docker)
check("image carries the house NCCL", "ARG NCCL_VERSION=2.30.7" in t_docker)
check("image carries verbs for RoCE", "libibverbs1" in t_docker and "ibverbs-providers" in t_docker)
check("image applies no engine patches", "git apply" not in t_docker and ".patch" not in t_docker)
check("image builds upstream at the pinned commit by default",
      "ARG TF_SOURCE=git" in t_docker and "ARG TF_REF=81de26f649677aa72ae0034d4af6e010cd947038" in t_docker
      and 'test "$(git rev-parse HEAD)" = "${TF_REF}"' in t_docker and "FROM engine-${TF_SOURCE}" in t_docker)
check("toolchain pins are verified",
      "--require-hashes" in t_docker
      and "d2804f94f54f7fcf8169f4b212a619ad688a4e2e6592e976f0cf2e58e2cd8af3" in t_docker
      and "ARG XGRAMMAR_COMMIT=97787376faee5ed8466cfad57c99855e4ce2f6aa" in t_docker)
check("image builds the grammar library for tool calls", "zig build grammar" in t_docker)
check("env names the source", "G_TF_SOURCE=git" in env)
check("image build helper exists and is executable", t_build.is_file() and t_build.stat().st_mode & 0o111 != 0)

# The overlay script on a fake checkpoint.
overlay = root / "scripts/glm_eos_overlay.py"
with tempfile.TemporaryDirectory() as d:
    m = Path(d) / "model"; m.mkdir()
    (m / "tokenizer.json").write_text(json.dumps({"model": {"vocab": {"a": 0}}, "added_tokens": [
        {"id": 154820, "content": "<|endoftext|>"}, {"id": 154824, "content": "<|assistant|>"},
        {"id": 154827, "content": "<|user|>"}]}))
    (m / "generation_config.json").write_text(json.dumps({"eos_token_id": [154820, 154827], "temperature": 1.0}))
    out = Path(d) / "state" / "generation_config.json"
    run = lambda *a: subprocess.run([sys.executable, str(overlay), *a], capture_output=True, text=True)
    r = run(str(m), str(out))
    g = json.loads(out.read_text()) if out.is_file() else {}
    check("overlay adds <|assistant|> to the end tokens", r.returncode == 0 and g.get("eos_token_id") == [154820, 154827, 154824])
    check("overlay keeps the other generation fields", g.get("temperature") == 1.0)
    check("overlay never writes the checkpoint",
          json.loads((m / "generation_config.json").read_text())["eos_token_id"] == [154820, 154827])
    r = run(str(m), str(out), "")
    check("empty token list writes an unchanged copy",
          r.returncode == 0 and json.loads(out.read_text())["eos_token_id"] == [154820, 154827])
    r = run(str(m), str(out), "<|nope|>")
    check("an unknown token is refused", r.returncode == 1)
    (m / "generation_config.json").write_text("{}")
    (m / "config.json").write_text(json.dumps({"eos_token_id": 154820}))
    r = run(str(m), str(out))
    check("overlay falls back to config.json's end token",
          r.returncode == 0 and json.loads(out.read_text())["eos_token_id"] == [154820, 154824])

raise SystemExit(fails)
