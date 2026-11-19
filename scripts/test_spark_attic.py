#!/usr/bin/env python3
"""CPU-only check of the spark-attic wiring (docs/SPARK-ATTIC.md).

Proves that SPARK_ATTIC unset or 0 leaves every compose command exactly as it
was, that SPARK_ATTIC=1 only adds the overlay file, and that each overlay adds
mounts and env to services its lane file already has. It runs the ctl
functions against stub docker and ssh binaries. Nothing touches a node.
"""
import os
import re
import subprocess
import tempfile
from pathlib import Path

root = Path(__file__).resolve().parent.parent
fails = 0


def check(name, cond):
    global fails
    print(("PASS" if cond else "FAIL"), name)
    fails += 0 if cond else 1


def services(text):
    body = text.split("\nservices:\n", 1)[1]
    return re.findall(r"^  ([A-Za-z0-9_-]+):\s*$", body, re.M)


# ---- overlays -------------------------------------------------------------------
overlay_dir = root / "configs/spark-attic"
overlays = sorted(overlay_dir.glob("*.attic.yml"))
check("overlays exist", len(overlays) >= 6)
for ov in overlays:
    text = ov.read_text()
    lane = root / "configs" / ov.name.replace(".attic.yml", ".compose.yml")
    check(f"{ov.name}: lane file exists", lane.is_file())
    if not lane.is_file():
        continue
    osvc, lsvc = services(text), services(lane.read_text())
    check(f"{ov.name}: only services of {lane.name}", osvc and set(osvc) <= set(lsvc))
    code = "\n".join(l for l in text.splitlines() if not l.lstrip().startswith("#"))
    check(f"{ov.name}: no image, build, command or new port",
          not re.search(r"^\s+(image|build|command|entrypoint|ports|container_name):", code, re.M))
    keys = set(re.findall(r"^    ([a-z_]+):", code, re.M))
    check(f"{ov.name}: adds only volumes and environment", keys <= {"volumes", "environment"})
    check(f"{ov.name}: key mounted read-only",
          re.search(r"target: /etc/spark-attic/key\n\s+read_only: true", code) is not None)
    check(f"{ov.name}: no host path created", code.count("create_host_path: false") == 3)
    for var in ("PYTHONPATH: /opt/spark-attic/python", "SPARK_ATTIC_KEY_FILE: /etc/spark-attic/key",
                "SPARK_ATTIC_PLACE: ${SPARK_ATTIC_PLACE:-kv}"):
        check(f"{ov.name}: {var.split(':')[0]}", var in code)
    check(f"{ov.name}: no relative host path", "../" not in code)
for sg in ("lane-next-sglang", "lane-glm-sglang"):
    check(f"no overlay for SGLang {sg}", not (overlay_dir / f"{sg}.attic.yml").exists())

# ---- env, docs, preflight -----------------------------------------------------
env = (root / "configs/sparkduet.env.example").read_text()
check("env example never sets SPARK_ATTIC", not re.search(r"^\s*SPARK_ATTIC\w*=", env, re.M))
check("env example documents SPARK_ATTIC=0", "# SPARK_ATTIC=0" in env)
doc = (root / "docs/SPARK-ATTIC.md").read_text()
check("doc has no em dash", "—" not in doc)
blocks = "\n".join(doc.split("```")[1::2])
check("no doc command passes --with-sysctl or --drm-dir", "--with-sysctl" not in blocks and "--drm-dir" not in blocks)
harness = (root / "docs/HARNESS.md").read_text()
check("HARNESS points at SPARK-ATTIC.md", "docs/SPARK-ATTIC.md" in harness)
pre = (root / "scripts/spark-attic-preflight.sh").read_text()
pre_code = "\n".join(l for l in pre.splitlines() if not l.lstrip().startswith("#"))
check("preflight runs --check", "args=(--check)" in pre_code)
check("preflight pins v0.1.6", "VERSION=0.1.6" in pre_code)
check("preflight refuses --with-sysctl", "--with-sysctl|--drm-dir" in pre_code)
check("preflight never reads the key", "/etc/spark-attic/key" not in pre_code)
check("preflight exits with the installer's code", 'exit "$rc"' in pre_code)

# ---- ctl: byte-for-byte off path, overlay on path ----------------------------------
with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    (tmp / "bin").mkdir()
    for tool in ("docker", "ssh"):
        p = tmp / "bin" / tool
        p.write_text('#!/bin/bash\n{ printf "%s" "' + tool + '"; printf "\\x1f%s" "$@"; printf "\\n"; } >> "$LOG"\n')
        p.chmod(0o755)
    envf = tmp / "test.env"
    envf.write_text((root / "configs/sparkduet.env.example").read_text())
    lanes = ["lane-depth", "lane-next", "lane-glm", "lane-glm-exl3", "lane-pd"]
    script = f"""
set -euo pipefail
source <(sed '/^cmd=/,$d' '{root}/scripts/sparkduetctl.sh')
ROOT='{root}'; ENV_FILE='{envf}'; load_env
start_fleet
for l in {' '.join(lanes)}; do compose_worker $l.compose.yml up -d w; compose_head $l.compose.yml up -d h; done
"""

    def run(extra):
        log = tmp / "log"
        log.write_text("")
        env = {"PATH": f"{tmp}/bin:/usr/bin:/bin", "HOME": str(tmp), "LOG": str(log), **extra}
        r = subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True)
        return r.returncode, [l.split("\x1f") for l in log.read_text().splitlines()]

    d = "/srv/ai/sparkduet"
    expect_off = [
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", "spark@10.0.0.2",
         f"cd '{d}' && FLEET_MODEL_ID='Qwen/Qwen3.8-27B' FLEET_SERVED_NAME='sparkduet-qwen38-27b' "
         "FLEET_PORT='30010'     docker compose --env-file sparkduet.env -f 'configs/lane-fleet.compose.yml' "
         "up -d fleet-replica"],
        ["docker", "compose", "--env-file", str(envf), "-f", f"{root}/configs/lane-fleet.compose.yml",
         "up", "-d", "fleet-replica"],
    ]
    for l in lanes:
        expect_off.append(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", "spark@10.0.0.2",
                           f"cd '{d}' && docker compose --env-file sparkduet.env -f 'configs/{l}.compose.yml' up -d w"])
        expect_off.append(["docker", "compose", "--env-file", str(envf), "-f",
                           f"{root}/configs/{l}.compose.yml", "up", "-d", "h"])
    rc, got = run({})
    have_fixture = rc == 0 and got and "spark@10.0.0.2" in got[0]
    check("ctl fixture ran (env example wiring)", have_fixture)
    if have_fixture:
        check("SPARK_ATTIC unset: compose commands unchanged", got == expect_off)
        rc0, got0 = run({"SPARK_ATTIC": "0"})
        check("SPARK_ATTIC=0: compose commands unchanged", rc0 == 0 and got0 == expect_off)
        rc1, got1 = run({"SPARK_ATTIC": "1"})
        check("SPARK_ATTIC=1: runs", rc1 == 0 and len(got1) == len(expect_off))
        def overlay_of(lane_path):
            name = Path(lane_path).name.replace(".compose.yml", ".attic.yml")
            return f"{root}/configs/spark-attic/{name}"
        head_ok = all(g == e[:6] + ["-f", overlay_of(e[5])] + e[6:]
                      for g, e in zip(got1, expect_off) if e[0] == "docker")
        check("SPARK_ATTIC=1: head adds only the lane's overlay", head_ok)
        worker_ok = all(
            "SPARK_ATTIC_PLACE='kv' docker compose" in g[-1] and "-f 'configs/spark-attic/" in g[-1]
            and g[-1].replace("SPARK_ATTIC_PLACE='kv' ", "").replace(
                re.search(r" -f 'configs/spark-attic/[^']+'", g[-1]).group(0), "") == e[-1]
            for g, e in zip(got1, expect_off) if e[0] == "ssh")
        check("SPARK_ATTIC=1: worker adds only the overlay and SPARK_ATTIC_PLACE", worker_ok)
        sg = subprocess.run(["bash", "-c", f"""
set -euo pipefail
source <(sed '/^cmd=/,$d' '{root}/scripts/sparkduetctl.sh')
ROOT='{root}'; ENV_FILE='{envf}'; load_env
compose_head lane-next-sglang.compose.yml up -d next-head"""],
            env={"PATH": f"{tmp}/bin:/usr/bin:/bin", "HOME": str(tmp), "LOG": str(tmp / "sg"), "SPARK_ATTIC": "1"},
            capture_output=True, text=True)
        check("SPARK_ATTIC=1 refuses a lane without an overlay", sg.returncode != 0)

raise SystemExit(fails)
