# Lane G EXL3 on SM121 (GB10)

Runtime patches for the EXL3/TR3 4bpw serve of GLM-5.3-Flash on two DGX Sparks.

## Why these files are here

The EXL3 vLLM image you pin as `G_EXL3_IMAGE` may predate four house
patches. `glm-exl3-entry.sh` bind-mounts this directory over `/opt/glm53/`
and applies them before `vllm serve`. A missing file is a hard failure.

| File | What it fixes | In a stock overlay? |
|---|---|---|
| `patch_hybrid_prefix_hit.py` | prefix-cache block accounting on the hybrid mamba path | often no |
| `patch_scheduler_decode_floor.py` | mixed prefill stealing a running decode step | often no |
| `patch_suppress_stops_in_reasoning.py` | client stop strings firing inside `<think>` | often no |
| `patch_clamp_max_tokens.py` | OpenCode `max_tokens=9999999` 400s against `max_model_len` | no |
| `patch_adaptive_verify_k.py` | Optional verify-length trim (`GLM53_ADAPTIVE_K=ema`). Default off. Public MTP k=2 is unchanged. | no |

`patch_hybrid_prefix_hit` is the correctness fix: the lane runs
`--enable-prefix-caching`. The other patches the image already applies
(`patch_model_overrides`, `patch_glm_video_placeholders`,
`patch_glm5_drafter_group`, the EXL3 quantization method, the aarch64 CPU
allreduce stubs) stay in the image. Do not vendor them here.

## Not shipped: DFlash2

A 62.9 tok/s headline [M-else] needs DFlash2 k=7. Those weights
(`incoai/GLM-5.3-Flash-DFlash2`) are CC BY-NC-ND 4.0: non-commercial, no
derivatives. This repo will not ship them. Lane G EXL3 runs license-clean
MTP (`SPEC_METHOD=mtp`, `MTP_TOKENS=2`), the same speculator family the
NVFP4 lane already used. Any `dflash2_*` files baked into an image stay
inert because the lane never passes `--speculative-config method=dflash`.

Public SparkDuet stays on MTP-2. A gitignored `local/restricted/` tree can
override the same-named compose on a private pair. Tracked files stay MTP.
See `docs/LOCAL-OVERLAYS.md`.

## Why EXL3 at all

Quality per byte on this checkpoint. An independent teacher-logit panel
(KLD, five cold runs, 25 sealed windows, 51,175 positions) puts EXL3/TR3
4bpw at 0.024555 nats against NVFP4's 0.060535 at effectively the same
footprint, level with official FP8 at 54 percent of the bytes. Speed is
not the reason: MTP k=2 measures near the NVFP4 lane.

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

### Base model

`zai-org/GLM-5.3-Flash`, MIT. ExLlamaV3 is MIT.
