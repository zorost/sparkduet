# SM121 kernel patches for Flash-Next on SGLang

Build context for `N_SGLANG_IMAGE` (`qwen38-flashnext-dspark:local`), the image
`configs/lane-next-sglang.compose.yml` runs. It is a derivative of
`lmsysorg/sglang:qwen38flashnext` and adds nothing but Python source patches, so
the build is seconds once the base image is local.

## Why the stock image cannot serve this lane

Qwen4Exp routes attention through Qwen Sparse Attention. On SM121 (GB10) both
stock paths are broken, and one of them fails silently:

- FlashInfer TRT-LLM paged sparse decode is numerically correct on exact SM120
  but corrupts long-context decode on SM121. A 120k prompt comes back as 32
  tokens of id 0 (`!`) while the server still answers HTTP 200. Nothing in the
  logs marks it.
- Excluding TRT-LLM falls through to packed FA4 CuTe varlen, which does not
  compile on GB10 at all (MLIR layout-congruence error).

So the choice on an unpatched image is a server that lies or a server that will
not start.

## What the three files do

| File | Origin | Effect |
|---|---|---|
| `sm121_varlen.py` | sglang#36845, then MiaAI-Lab 0f95001 | Triton packed one-query varlen kernel matching the QSA call contract. Reads `cu_seqlens` on-device so CUDA-graph replay stays valid. Zeros non-finite attention so an empty selected-KV does not become token id 0. |
| `apply_nvfp4_patches.py` | MiaAI-Lab | Applied at image build. Forces `_resolve_trtllm_sparse_decode` to `None` on SM121 (sglang#36806) even if a newer base re-enables it, returns the Triton fallback from `_resolve_flash_attn_varlen_func` on SM121, wires NVFP4 KV for the QSA pools, and installs the leftover token-id-0 abort (16 consecutive zeros, no radix insert, prefix-cache reset). SM100/SM120 keep their native paths. |
| `qsa_nvfp4_kv.py` | MiaAI-Lab | NVFP4 KV cache method declaring plain BF16 dequant reads for every backend and phase, so the pool allocates packed FP4 plus per-block FP8 scales with no FP8 dequant workspace. |

Every patch asserts on its anchor. A base-image layout change fails the build
loudly instead of producing an image that quietly serves the broken path.

## Provenance

Vendored from [MiaAI-Lab/Qwen3.8-Flash-Next-Dual-DGX-Sparks](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Dual-DGX-Sparks)
at commit `0f950012c8d8323acac9a08846a32ef7953f5f62` (29 Aug 2026 leftover
token-id-0 abort on top of `344f9d0`), MIT. Kernel and apply scripts are
ported from the `.patch/` context that `start.sh` generates. The house
does not run `start.sh`. See `LICENSE.MiaAI-Lab`.

The upstream repo is a self-contained launcher that manages its own containers.
The house does not run it: lane N is started as a pair by `sparkduetctl.sh` and
recovered by `zorost-lane-guard`, and a second orchestrator competing for
:30000 is the exact failure that took the lane down on 2026-08-28. Only the
build context is vendored; the launch contract lives in the lane file.
`sparkduetctl.sh` will LD_PRELOAD host `libnccl.so.2.30.7` when that file
exists on both nodes (`~/nccl-2.30.7/`). The image ships NCCL 2.29.7.

## Rebuild

Run on both nodes. The head and the worker must run the identical image.

```bash
docker pull lmsysorg/sglang:qwen38flashnext
docker build -t qwen38-flashnext-dspark:local patches/next-sglang-sm121
```
