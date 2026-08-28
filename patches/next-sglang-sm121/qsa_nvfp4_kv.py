"""NVFP4 KV cache for the QSA (Qwen4-Exp sparse attention) path on DGX Spark.

Upstream SGLang's NVFP4 KV recipe assumes two consumers that do not exist on
the QSA path: FlashInfer prefill reading an FP8 *dequant workspace* covering
the whole pool, and TRT-LLM decode consuming native packed FP4.  The QSA
backend instead gathers a sparse subset of KV rows with Triton kernels and
wants dense BF16 rows.  On top of that, the full-size FP8 workspace would eat
most of the FP4 savings (fp4 data + scales + fp8 workspace is ~1.56 B/elem
against bf16's 2 B/elem; without the workspace it is 0.5625 B/elem — a 3.6x
cut, verified CUDA-graph safe on SM121 via flashinfer's nvfp4_kv_* kernels).

This module provides:

* ``QSANVFP4KVCacheMethod`` — an NVFP4 method whose attention-access rules
  declare PLAIN BF16 reads only (no DEQUANT_WORKSPACE, no NATIVE_FP4), so the
  pool allocates packed FP4 + per-block FP8 scales and nothing else.  Plain
  readers (``get_key_buffer``) dequantize on demand through
  ``dequantize_kv_tensor``.  Quantization, storage layout, per-layer global
  scales and slot moves are inherited unchanged from the upstream method.
* ``try_fp4_view`` / ``compact_and_dequant`` / ``gather_history_fp4`` — the
  gather-dequant helpers the patched ``QwenSparseAttnBackend`` calls on its
  decode/verify and chunked-prefill paths.  The stock Triton compaction
  kernel runs over the packed FP4 buffers and (a second time) over the
  per-block scale buffers — same leading dims, dim/16 — and the compacted
  rows are dequantized with flashinfer's ``nvfp4_kv_dequantize``.

Wiring (applied by the sibling ``apply_nvfp4_patches.py`` build step):

* ``get_kv_cache_quant_method("nvfp4")`` routes here (this image serves QSA
  models; non-QSA models should use the stock image for nvfp4 KV).
* ``_handle_kv4_compatibility`` allows nvfp4 KV for QSA hybrids whose
  ``--attention-backend`` flag only selects the GDN linear-attn kernels.
* The pool-configurator cell size skips the FP8 workspace share.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from sglang.srt.layers.quantization.fp4_kv_cache_quant_method import (
    KVCacheAttentionAccess,
    KVCacheAttentionAccessKind,
    KVCacheAttentionPhase,
    KVCacheBackendMatcher,
    NVFP4KVCacheMethod,
)
from sglang.srt.layers.quantization.kvfp4_tensor import NVFP4KVQuantizeUtil

_BF16 = torch.bfloat16
_U8 = torch.uint8
_FP8 = torch.float8_e4m3fn


def _plain_access(phase: KVCacheAttentionPhase) -> KVCacheAttentionAccess:
    return KVCacheAttentionAccess(
        phase,
        KVCacheAttentionAccessKind.PLAIN,
        KVCacheBackendMatcher(any_backend=True),
        storage_dtype=_U8,
        attention_kv_dtype=_BF16,
        scale_recipe="nvfp4",
    )


class QSANVFP4KVCacheMethod(NVFP4KVCacheMethod):
    """NVFP4 KV cache as consumed by the QSA backend: plain BF16 dequant reads.

    Differs from the upstream ``NVFP4KVCacheMethod`` only in its declared
    attention accesses: PLAIN for both phases and every backend.  As a
    consequence the pool allocates no FP8 dequant workspace
    (``needs_dequant_workspace()`` is False) and plain readers dequantize
    packed FP4 + scales on demand.
    """

    def __init__(self, num_layers: int, device: str):
        super().__init__(num_layers, device)
        # Pre-allocate so the out-of-range fallback never does torch.ones
        # during CUDA-graph capture.
        self._ones_scale = torch.ones(1, dtype=torch.float32, device=device)

    def attention_accesses(self) -> tuple[KVCacheAttentionAccess, ...]:
        return (
            _plain_access(KVCacheAttentionPhase.PREFILL),
            _plain_access(KVCacheAttentionPhase.DECODE),
        )

    def _layer_global_scale(
        self, scales_gpu: torch.Tensor, layer_id: int
    ) -> torch.Tensor:
        # The write path indexes k_scales_gpu by the GLOBAL layer id (with
        # load_scales_from_model resizing the vector to cover global ids);
        # guard against shorter vectors (all-ones scales) anyway.
        if 0 <= layer_id < scales_gpu.numel():
            return scales_gpu[layer_id : layer_id + 1]
        return self._ones_scale

    def dequantize_kv_tensor(
        self,
        fp4_tensor: torch.Tensor,
        scales: torch.Tensor,
        layer_id: int,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """Dequantize one packed FP4 KV tensor (whole-pool view) for plain reads."""
        if scales.dtype != _FP8:
            scales = scales.view(_FP8)
        return NVFP4KVQuantizeUtil.dequantize(
            fp4_tensor.view(_U8),
            scales,
            self._layer_global_scale(self.k_scales_gpu, layer_id),
            dtype=dtype or _BF16,
        )


class QSAFP4KVView:
    """Packed FP4 + per-block-scale buffers of one QSA full-attention layer."""

    def __init__(self, k_fp4, v_fp4, k_sf, v_sf, k_gs, v_gs):
        self.k_fp4 = k_fp4  # uint8 [rows, head_num, head_dim // 2]
        self.v_fp4 = v_fp4
        self.k_sf = k_sf  # float8_e4m3 view [rows, head_num, head_dim // 16]
        self.v_sf = v_sf
        self.k_gs = k_gs  # 1-element fp32 global scale (on device)
        self.v_gs = v_gs
        self.head_num = k_fp4.shape[1]
        self.head_dim = k_fp4.shape[2] * 2
        self.device = k_fp4.device

    @property
    def k_sf_u8(self) -> torch.Tensor:
        return self.k_sf.view(_U8)

    @property
    def v_sf_u8(self) -> torch.Tensor:
        return self.v_sf.view(_U8)

    def dequant_rows(self, k_rows, k_sf_rows, v_rows, v_sf_rows):
        """Dequantize already-gathered packed rows -> (k_bf16, v_bf16)."""
        k = NVFP4KVQuantizeUtil.dequantize(
            k_rows, k_sf_rows.view(_FP8), self.k_gs
        )
        v = NVFP4KVQuantizeUtil.dequantize(
            v_rows, v_sf_rows.view(_FP8), self.v_gs
        )
        return k, v


def try_fp4_view(pool, layer_id: int) -> Optional[QSAFP4KVView]:
    """Return the layer's FP4 buffers, or None for an unquantized pool.

    On SM100 the stock native-FP4 consumers stay in charge, so the QSA
    gather-dequant path is SM120/SM121 (DGX Spark) only.
    """
    from sglang.srt.utils import is_sm100_supported

    if is_sm100_supported():
        return None
    full_pool = getattr(pool, "full_kv_pool", pool)
    quant_method = getattr(full_pool, "quant_method", None)
    if quant_method is None or getattr(quant_method, "name", None) != "nvfp4":
        return None
    k_fp4, v_fp4, k_sf, v_sf = pool.get_raw_kv_buffer(layer_id)
    k_gs = quant_method._layer_global_scale(quant_method.k_scales_gpu, layer_id)
    v_gs = quant_method._layer_global_scale(quant_method.v_scales_gpu, layer_id)
    return QSAFP4KVView(k_fp4, v_fp4, k_sf, v_sf, k_gs, v_gs)


def compact_and_dequant(
    backend,
    fp4_kv: QSAFP4KVView,
    scratch_capacity: int,
    req_indices,
    topk_indices,
    sequence_lens,
    cu_seqlens_k,
    batch: int,
    topk: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sparse gather + dequant for the QSA decode/verify path.

    Runs the stock compaction kernel twice — once over the packed FP4 data,
    once over the per-block scale buffers — then dequantizes the compacted
    rows to BF16.  Kernel-only, so CUDA-graph capture/replay is safe (the
    dequant scratch rows past the packed prefix hold garbage that the varlen
    attention kernel never reads, exactly like the BF16 path).
    """
    from sglang.srt.layers.attention.qsa.sparse_attn import (
        qwen_sparse_kv_extraction_compact_triton,
    )

    req_to_token = backend.req_to_token_pool.req_to_token
    n, d = fp4_kv.head_num, fp4_kv.head_dim
    # The scratch cache is keyed by (heads, dim, dtype, device), so the FP4
    # data (dim/2), scale (dim/16) and BF16 (dim) buffers never collide.
    pk_fp4, pv_fp4 = backend._get_fa2_scratch(
        scratch_capacity, n, d // 2, _U8, fp4_kv.device
    )
    pk_sf, pv_sf = backend._get_fa2_scratch(
        scratch_capacity, n, d // 16, _U8, fp4_kv.device
    )
    qwen_sparse_kv_extraction_compact_triton(
        fp4_kv.k_fp4,
        fp4_kv.v_fp4,
        req_to_token,
        req_indices,
        topk_indices,
        sequence_lens,
        cu_seqlens_k,
        pk_fp4,
        pv_fp4,
        batch,
        topk,
    )
    qwen_sparse_kv_extraction_compact_triton(
        fp4_kv.k_sf_u8,
        fp4_kv.v_sf_u8,
        req_to_token,
        req_indices,
        topk_indices,
        sequence_lens,
        cu_seqlens_k,
        pk_sf,
        pv_sf,
        batch,
        topk,
    )
    return fp4_kv.dequant_rows(pk_fp4, pk_sf, pv_fp4, pv_sf)


def gather_history_fp4(
    fp4_kv: QSAFP4KVView, req_to_token, req_indices, sequence_lens
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Chunked-prefill history gather: index_select + dequant per request.

    Mirrors the BF16 path's per-request ``index_select`` + ``cat`` (the
    validated chunk-prefill kernel consumes tightly packed full-context
    K/V); the only difference is that the gathered packed rows are
    dequantized to BF16 on the way out.
    """
    k_parts = []
    v_parts = []
    k_fp4, v_fp4 = fp4_kv.k_fp4, fp4_kv.v_fp4
    k_sf_u8, v_sf_u8 = fp4_kv.k_sf_u8, fp4_kv.v_sf_u8
    for i, seq_len in enumerate(sequence_lens):
        slots = req_to_token[req_indices[i], :seq_len].long()
        k_parts.append(
            NVFP4KVQuantizeUtil.dequantize(
                k_fp4[slots], k_sf_u8[slots].view(_FP8), fp4_kv.k_gs
            )
        )
        v_parts.append(
            NVFP4KVQuantizeUtil.dequantize(
                v_fp4[slots], v_sf_u8[slots].view(_FP8), fp4_kv.v_gs
            )
        )
    return torch.cat(k_parts), torch.cat(v_parts)
