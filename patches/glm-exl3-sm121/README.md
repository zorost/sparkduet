# Lane G EXL3 on SM121 (GB10)

Runtime patches for the EXL3/TR3 4bpw serve of GLM-5.3-Flash on two DGX Sparks.

## Why these files are here

The published overlay image
`ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks:exl3` predates three patches that
are now in MiaAI-Lab's Dockerfile, plus a house clamp for OpenCode's 9999999
`max_tokens` sentinel. Its `/opt/glm53/` carries ten files and none of these:

| File | What it fixes | In the image? |
|---|---|---|
| `patch_hybrid_prefix_hit.py` | prefix-cache block accounting on the hybrid mamba path | no |
| `patch_scheduler_decode_floor.py` | mixed prefill stealing a running decode step (upstream issue #6) | no |
| `patch_suppress_stops_in_reasoning.py` | client stop strings firing inside `<think>` | no |
| `patch_clamp_max_tokens.py` | OpenCode `max_tokens=9999999` 400s against `max_model_len` | no |

The head script applies whatever exists under `/opt/glm53/`, so absent files fail
open and the engine boots without the fix. `patch_hybrid_prefix_hit` is the one
that matters for correctness: the lane runs `--enable-prefix-caching`, which is
the code path it repairs. `glm-exl3-entry.sh` bind-mounts this directory over
`/opt/glm53/` and applies all three before `vllm serve`.

Rebuilding the image from MiaAI-Lab's current Dockerfile would bake these in and
make the mount redundant. That build is long and the mount is equivalent, so the
house lane mounts.

The other patches the head path needs (`patch_model_overrides`,
`patch_glm_video_placeholders`, `patch_glm5_drafter_group`, the EXL3 quantization
method, the aarch64 CPU allreduce stubs) are already baked and applied in the
image. Do not vendor them; the image copies are authoritative.

## Not vendored: DFlash2

MiaAI-Lab's published numbers (62.9 tok/s single stream [M-else]) use DFlash2 k=7 as the
speculator. `incoai/GLM-5.3-Flash-DFlash2` is licensed CC BY-NC-ND 4.0, which is
non-commercial and forbids derivatives. The house lane cannot use it. Lane G EXL3
runs the license-clean rollback MiaAI-Lab documents, `SPEC_METHOD=mtp` with
`MTP_TOKENS=2`, which is the same MTP speculation the NVFP4 lane already used.
The `dflash2_*` and `patch_dflash2` files baked into the image stay inert because
the lane never passes `--speculative-config method=dflash`.

## Why EXL3 at all

Quality per byte on this checkpoint. An independent teacher-logit panel
(KLD, five cold runs, 25 sealed windows, 51,175 positions) puts EXL3/TR3 4bpw at
0.024555 nats against NVFP4's 0.060535 at effectively the same footprint, level
with official FP8 at 54 percent of the bytes. Speed is not the reason: without
DFlash2, MTP k=2 measures near the NVFP4 lane.

## Attribution

### Weights, EXL3/TR3 quantization

`brandonmusic/GLM-5.3-Flash-tr3-4bpw`, pinned at revision
`5ab363a8dcf6405955fd5f99671e01a1c9fb124b`, licensed ShapleyMcg License v1.0
(`LICENSE.shapleymcg-1.0`). Section 3 makes attribution a condition of the
grant, so the notice below travels with any copy, derivative, or published
result of this lane.

> This work includes or was produced using ShapleyMcg, created by Brandon M.
> Music (https://github.com/brandonmmusic-max/shapleymcg). ShapleyMcg is
> licensed under the ShapleyMcg License v1.0, an attribution-required license
> that grants no rights to the person known as "0xSero." Use of ShapleyMcg
> without this attribution is unlicensed.

```bibtex
@misc{music2026shapleymcg,
  author = {Music, Brandon M.},
  title  = {ShapleyMCG: An Auditable Calibration-to-Encoding Pipeline for
            Low-Bit Mixture-of-Experts Models},
  year   = {2026},
  url    = {https://github.com/brandonmmusic-max/shapleymcg},
  note   = {Licensed under the ShapleyMcg License v1.0}
}
```

The license is permissive for commercial use and excludes one named third party,
unrelated to this deployment.

### Overlay, GB10 serve recipe

`patch_*.py` in this directory are from
https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks, copyright 2026
Mia's AI Lab, MIT (`LICENSE.MiaAI-Lab`). The EXL3 vLLM quantization method, the
FLASHINFER_MLA_SPARSE_SM120 NoPE padding, and the aarch64 exllamav3 stubs are
that project's work.

### Base model

`zai-org/GLM-5.3-Flash`, MIT. ExLlamaV3 is MIT.
