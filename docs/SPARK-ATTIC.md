# SPARK-ATTIC.md, lending the display carveout to the engine

State on 2026-11-19: **not installed on either Spark, and off in this repo.**
`SPARK_ATTIC` defaults to off. Nothing here was installed, deployed or
restarted on the pair from this repo. The resident GLM EXL3 lane runs exactly
as before.

## What it is

[spark-attic](https://github.com/bertholomus/spark-attic) (MIT OR Apache-2.0)
is an upstream project. This repo does not vendor its daemon, shims, client or
keys. It only carries an opt-in compose overlay, a read-only preflight wrapper
and this page.

- `atticd` is a root daemon (`attic.service`). On a headless GB10 it lends the
  display carveout that the GPU reserves and nobody uses, about 2,044 MiB, to
  CUDA as ordinary device memory. It serves a socket at
  `/run/spark-attic/attic.sock` and checks a key at `/etc/spark-attic/key`.
- A per-driver shim (`rmshim.so`) is what lets it do that. A shim is trusted
  only once `attic-validate.sh` has passed on a node running that exact
  driver.
- A vLLM plugin (`spark_attic.vllm_plugin`, entry point
  `vllm.general_plugins`) can add the borrowed memory to the KV pool. It knows
  two internal vLLM shapes, called 0.24 and 0.32 here. It picks one by looking
  at the running modules, not at the version string.
- The client fails open. If atticd is absent, nothing is free, or the shape is
  unknown, the engine boots as if the plugin were not there.
- Once the plugin has raised the KV budget, the engine depends on that memory.
  A placement failure after that point raises, and the server stops. The
  exception is `SPARK_ATTIC_PLACE=off`, which turns the placement off.

## Why OpenCode and Hermes do not get a larger context window

OpenCode, Hermes and Open WebUI hold no CUDA memory. They talk HTTP to the
engine on `:30000`. They benefit only if that engine really grows its KV pool.

A larger KV pool means more tokens can be cached at once: more concurrent
sequences, or fewer preemptions on long ones. It does not change
`max_model_len`. The window an agent may advertise is still the one the engine
reports on `/v1/models` (see `docs/HARNESS.md`). Therefore this change leaves
three things alone:

- the house catalog context
- the 9Router rows
- the picker windows

About 2 GiB per node is small next to the GLM worker's roughly 101 GiB.

## The three gates

All three must hold before `SPARK_ATTIC=1` is worth trying on a lane.

1. **A validated shim for the running driver, on both nodes.** The release
   ships shims for some driver versions. One is trusted on a node only after
   `attic-validate.sh` passes there.
2. **The display is idle, on both nodes.** atticd lends nothing while a
   display is in use. It counts three signals:
   - a process other than `nvidia-persistenced` holding `/dev/nvidia-modeset`
   - a graphics (type G) process
   - a DRM connector that reports `connected`
3. **The KV shape is still one the plugin knows.** This repo cannot check that
   ahead of time. The fail-open start is the check: read the engine log on the
   first boot with the flag on.

The lanes are TP=2 across both nodes. The scheduler sizes the KV pool by the
smallest rank. **A one-node lender cannot grow a TP=2 KV pool**, because the
rank that cannot lend sets the limit. That is why both nodes must pass, and why
`sparkduetctl.sh` checks both before it adds the overlay. Lane F (fleet) runs
one independent replica per node, so there each node gains or not on its own.

## Preflight on 2026-11-19

The measurements below were taken on the nodes by an operator, not from this
VM, which cannot reach the Sparks.

Both nodes:
- aarch64
- NVIDIA driver 580.173.02
- kernel 6.17.0-1031-nvidia
- no spark-attic and no dispram installed

Official `install.sh` v0.1.6, run as `--check --from` the release tarball:

| Gate | dspark1 (head) | dspark2 (worker) |
|---|---|---|
| shim | **FAIL**: no shim for 580.173.02 | **FAIL**: no shim for 580.173.02 |
| display | **FAIL**: "a display is in use: atticd lends nothing while one is". Xorg and gnome-shell run as type G processes. | PASS: only `nvidia-persistenced` holds `/dev/nvidia-modeset` |
| KV shape | see below | see below |

The v0.1.6 artifact carries shims for:
- 580.178.04
- 580.196.07
- 595.112.07
- 615.71.09
- 615.71.10
- 615.78.08
- 620.70.06

It has none for 580.173.02. The pair therefore fails gate 1 on both nodes and
gate 2 on dspark1.

**KV shape, read inside the live GLM image** (`glm53-flash-sm121:e3-pipeline-f1s8`,
vLLM `0.1.dev20051+g487ecf187`):

- `KVCacheTensor` has the fields `size`, `shared_by`, `offset` and
  `block_stride`. That is the 0.24 field set, with no unknown fields.
- `GPUModelRunner._allocate_kv_cache_tensors(self, kv_cache_config)` exists,
  and `initialize_kv_cache_tensors` calls it.
- `vllm.v1.worker.utils.allocate_kv_cache` does not exist, so this is not the
  0.32 shape.
- `Worker.load_model` on the class takes only `self`, so the weights placement
  would turn itself off.

So `SPARK_ATTIC_PLACE=kv` is the placement that could work on a later boot of
this image. Nobody restarted the lane to prove it.

Resident lane on that day: GLM EXL3 TP=2. `sparkduet-glm-head` ran on dspark1
and `sparkduet-glm-worker` on dspark2, both up about 42 hours, the worker at
about 103,559 MiB. Leave them running.

## What to do about the gates

- **Shim.** NVIDIA's open-gpu-kernel-modules repo has a tag for 580.173.02.
  The supported path is the upstream one:
  1. Run `scripts/attic-build.sh 580.173.02` on a build host. It builds
     `rmshim.so` and `BUILD-INFO` from that tag's headers and never writes
     `VALIDATED`.
  2. Run `attic-validate.sh` on each node. That is the only step that marks a
     shim validated.

  Both scripts are upstream's. This repo does not ship them or the result.
- **Display on dspark1.** Leave Xorg, gnome-shell and the desktop running. Do
  not log the session out, and do not pass `--drm-dir`, which only points the
  check at a test tree. While dspark1 drives a display it lends nothing, and
  by the TP=2 rule above the GLM lane gains nothing.
- **Until both nodes pass, leave `SPARK_ATTIC` off.**

## Running the preflight

On each node, from the repo:

```
scripts/spark-attic-preflight.sh --from /path/to/spark-attic-0.1.6-aarch64.tar.gz
scripts/spark-attic-preflight.sh            # no artifact: shim and display checks only warn
```

The wrapper runs the official v0.1.6 `install.sh --check` and nothing else:

- It checks the installer against a pinned sha256 and against the version
  stamped inside it.
- It runs the check as root through `sudo`.
- It exits with the installer's code: 0 clean, 1 problems, 2 usage.
- It does not install anything and does not pass `--with-sysctl` or
  `--drm-dir`.
- It never reads or prints `/etc/spark-attic/key`.

The installer documents that `--check` writes nothing anywhere.

## Attaching an engine later

Do this only after the installer has run on both nodes and the upstream
`spark-attic status` reports `"state":"ready"` and `"validated":true` on both.
Installing is a separate, explicit operator step. It is not part of this repo.
No `--with-sysctl`.

These are the exact additions an engine container needs. They match
upstream's `docker run` example:

```
volumes:
  /run/spark-attic                      -> /run/spark-attic
  /etc/spark-attic/key                  -> /etc/spark-attic/key            (read-only)
  /usr/local/share/spark-attic/python   -> /opt/spark-attic/python         (read-only)
environment:
  PYTHONPATH=/opt/spark-attic/python
  SPARK_ATTIC_KEY_FILE=/etc/spark-attic/key
  SPARK_ATTIC_PLACE=kv
```

The overlay also sets `SPARK_ATTIC_SOCK=/run/spark-attic/attic.sock`, which is
upstream's default, to make the path explicit.

In this repo those additions live in `configs/spark-attic/<lane>.attic.yml`,
one file per vLLM lane file. Each overlay:

- adds the three bind mounts and the env vars to the services its lane file
  already defines
- defines no image, no command and no new service
- uses absolute host paths, so it also merges on top of a `local/restricted/`
  lane file
- sets `create_host_path: false`, so a node without the install fails the
  start instead of docker creating an empty `/etc/spark-attic/key` directory

The SGLang lanes have no overlay, and `SPARK_ATTIC=1` refuses to start them.

The flag is read from the shell for one start:

```
SPARK_ATTIC=1 scripts/sparkduetctl.sh restart glm        # not today: the gates fail
```

With `SPARK_ATTIC` unset or `0`, `sparkduetctl.sh` sends byte-for-byte the
same compose commands as before. `scripts/test_spark_attic.py` checks this.

With `SPARK_ATTIC=1`, it does three things:

1. It checks on both nodes that the socket, the key path and the client
   package exist (`test -e` only, the key is not read).
2. It prints each node's keyless `status.json`.
3. It adds `-f configs/spark-attic/<lane>.attic.yml` to the head and worker
   compose commands, and `SPARK_ATTIC_PLACE` inline to the worker's.

Do not put `SPARK_ATTIC` in `sparkduet.env`. That file is sourced after the
shell env and would pin the flag for every start.

Before the first boot with the flag on, check two things on the image:

- **`PYTHONPATH`.** The overlay sets it to the client package only. Check that
  the image does not already set one:
  `docker image inspect <image> --format '{{.Config.Env}}'`.
  The tracked lane files and entry scripts set none.
- **`VLLM_PLUGINS`.** If it is set, it must include `spark_attic`.

On that first boot, read the engine log for the plugin's line. It should say
either that it raised the KV budget, or that it turned itself off and why. If
it raised the budget and a later placement fails, the server stops. To boot
without the placement, restart with `SPARK_ATTIC_PLACE=off`, or with
`SPARK_ATTIC` unset.

## The key

`/etc/spark-attic/key` is created by the installer, root-only (0600). An
engine receives it only as a read-only file mount, through
`SPARK_ATTIC_KEY_FILE`. It is never printed, copied into chat, put in an env
var or committed. Nothing in this repo reads it: the ctl only tests that the
path exists. `.gitignore` already excludes `*.key`.

## Files

| File | What |
|---|---|
| `configs/spark-attic/*.attic.yml` | Opt-in compose overlays, one per vLLM lane file |
| `scripts/sparkduetctl.sh` | `SPARK_ATTIC=1` adds the overlay. Unset or 0 changes nothing. |
| `scripts/spark-attic-preflight.sh` | Read-only `install.sh --check` wrapper |
| `scripts/test_spark_attic.py` | CPU-only check of the above |
| `configs/sparkduet.env.example` | Documents `SPARK_ATTIC=0`, commented out |

## Sources

These upstream files were read on 2026-11-19, from
github.com/bertholomus/spark-attic:
- `README.md`
- `INSTALL.md`
- `AGENTS.md`
- `install.sh` (v0.1.6 release)
- `python/pyproject.toml`
- `python/spark_attic/vllm_plugin.py`
- `scripts/attic-build.sh`
- `scripts/attic-validate.sh`
