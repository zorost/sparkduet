# Lane G on TensorFold (G_ENGINE=tensorfold)

The EXL3/TR3 4bpw checkpoint the vLLM EXL3 lane serves, on TensorFold's native
CUDA server instead of vLLM. Same weights, same served id, same `:30000`, same
container names. Opt-in: `G_ENGINE=vllm` stays the default until an A/B on the
pair says otherwise (`docs/MODELS.md`).

| File | What it is |
|---|---|
| `Dockerfile` | Builds TensorFold from upstream main at a pinned commit (default), or unpacks the `v1.3.0` `linux-aarch64` release archive checked against its published sha256 and its own `SHA256SUMS`. Adds NCCL 2.30.7, libibverbs with the mlx5 provider, and the CUDA 13 runtime (cuBLASLt for the vision tower). Applies no engine patch. |
| `../glm-tensorfold-entry.sh` | Checks the rank's placement and that the lane drafts with the checkpoint's MTP head only, then runs `tensorfold-native serve /model`. |
| `../../configs/lane-glm-tensorfold.compose.yml` | The two ranks, one per Spark. |
| `../../scripts/glm_eos_overlay.py` | Writes the end-token overlay (below). |
| `../../scripts/build-tensorfold-image.sh` | Builds the image on the head and ships the same image id to the worker. |

## Why this engine, and which build

A public two-GB10 recipe announced on 2026-11-20 that its GLM speed-ups now
live in upstream TensorFold, at main commit `81de26f6`. That commit reports
1.3.0 but comes after the `v1.3.0` tag (`d3d0bf11`). Its changelog lists, as
unreleased, programmatic dependent launches along GLM's CUDA decode chain,
next-turn cache guesses that carry the reply's reasoning (#3264), and KV
page-retirement bookkeeping (#3852). The release archive has none of those.
So the default build (`G_TF_SOURCE=git`) compiles upstream at `81de26f6`, with
Zig 0.17.0 from PyPI's hash-pinned `ziglang` wheel and xgrammar `v0.2.8`
(`97787376`) for structured output and required tool calls.
`G_TF_SOURCE=release` takes the release archive instead: no compile, without
those changes. Neither applies a patch.
An agent loop on this pair sends a median of about 98k tokens a step
(`docs/HARNESS.md`), so how much of each step resumes from a kept prompt state,
and how fast the rest is read, matters as much as decode. Upstream lists this
exact checkpoint for two CUDA ranks and marks its EXL3 speed qualification
"TBD", so nothing here claims a number for this pair.

## License and attribution

- **TensorFold**, Apache-2.0 (https://github.com/ashhart/TensorFold). The binary
  is downloaded at image build, not stored here. Its `LICENSE`, `NOTICE`,
  `THIRD_PARTY_NOTICES.md` and `LICENSES/` stay in `/opt/tensorfold` inside the
  image.
- **Weights**, `brandonmusic/GLM-5.3-Flash-tr3-4bpw`, ShapleyMcg License v1.0.
  Attribution is a condition of that grant: the required notice is in
  `patches/glm-exl3-sm121/README.md` and applies to this engine unchanged.
- **NCCL**, NVIDIA's license, from the `nvidia-nccl-cu13` wheel.

Drafts come from the checkpoint's own MTP head. No external drafter is
downloaded, mounted or accepted by the entry script.

## End tokens

GLM-5.3-Flash can end a reply with `<|assistant|>`, which the checkpoint's
`generation_config.json` does not list among its end tokens. An engine that
stops only at the listed ids then writes a second turn until the reply limit,
which in an agent loop is a 16384-token stall. Before each start,
`sparkduetctl.sh` runs `scripts/glm_eos_overlay.py` on both Sparks. It writes a
copy of `generation_config.json` with that token's id added into
`G_TF_STATE_DIR`, checks the two copies match, and the lane mounts it read-only
over the original. The checkpoint is never written. `G_TF_EXTRA_EOS=''` serves
the checkpoint's own list.

## Build

On the head Spark, with `sparkduet.env` in place:

```bash
scripts/build-tensorfold-image.sh --dry-run   # what it would run
scripts/build-tensorfold-image.sh             # build, then docker save | ssh worker docker load
```

## What was checked, and where

On the build VM (x86_64, no GPU), 2026-11-20:

- The release archive's sha256 matches the published `.sha256`, its own
  `SHA256SUMS` passes, `VERSION` reads `1.3.0`, and the binary is an aarch64
  executable whose fatbins hold sm_121 images (upstream `RUNBOOK.md`).
- The pins the source build checks: `v1.3.0` is `d3d0bf11` and `81de26f6` is
  later; `build.zig.zon` at `81de26f6` asks for Zig 0.17.0; xgrammar `v0.2.8`
  peels to `97787376`; the `ziglang` 0.17.0 aarch64 wheel's sha256 is PyPI's.
- `tensorfold-native` needs `libnccl.so.2` for the GLM pair, `libibverbs.so.1`
  for the RoCE gathers and `libcublasLt.so.13` for `--vision`; the image adds
  each.
- The entry script's refusals, the compose wiring and the overlay script are
  covered by `scripts/test_lane_glm.py`.

**[unverified on chassis]** The image has not been built on a Spark, and the
lane has not booted on the pair. The base image tag was not pulled from the
build VM. Pin it by digest after the first pull.
