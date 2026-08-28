#!/usr/bin/env python3
"""Apply the DSpark NVFP4-KV source patches inside the qwen38flashnext image.

All patches are QSA-scoped and inert unless ``--kv-cache-dtype nvfp4`` is set:

1. ``qwen_sparse_attn_backend.py`` — the QSA gather paths (decode/verify and
   chunked prefill) read the packed FP4 + per-block-scale buffers and
   dequantize the gathered rows, instead of pulling whole BF16 pool views.
2. ``fp4_kv_cache_quant_method.py`` — route ``nvfp4`` to the QSA plain-dequant
   method (no FP8 dequant workspace, no native-FP4 decode path).
3. ``server_args.py`` — allow nvfp4 KV for QSA hybrid models whose
   ``--attention-backend`` only selects the GDN linear-attn kernels.
4. ``pool_configurator.py`` — don't reserve the FP8 workspace share of the
   FP4 cell size for QSA models (the QSA method allocates none).
5. ``memory_pool.py`` — NVFP4 write path: ignore hybrid pool's Python
   ``k_scale=1.0`` default and use on-device ``k_scales_gpu`` (host→CUDA
   ``torch.tensor`` is illegal during decode CUDA-graph capture).
6. ``sparse_attn.py`` — SM121 Triton cannot ``tl.dot`` fp8e4nv. The
   chunk-prefill / prefill GQA kernels upcast K/V (and Q) to fp32 before
   the dots so ``--kv-cache-dtype fp8_e4m3`` survives extend, not only
   the paged-varlen decode fallback.
"""

import pathlib

SRT = pathlib.Path("/sgl-workspace/sglang/python/sglang/srt")

MARKER = "qsa_nvfp4_kv"


def patch(path, replacements):
    s = path.read_text()
    if MARKER in s:
        print(f"{path.name}: already patched")
        return
    for anchor, replacement in replacements:
        count = s.count(anchor)
        assert count == 1, f"{path.name}: anchor matched {count} times (want 1):\n{anchor}"
        s = s.replace(anchor, replacement, 1)
    path.write_text(s)
    print(f"{path.name}: patched")


# ---------------------------------------------------------------------------
# 1. QSA attention backend: FP4-aware gather paths
# ---------------------------------------------------------------------------
BACKEND = SRT / "layers" / "attention" / "qwen_sparse_attn_backend.py"

IMPORT_ANCHOR = """from sglang.srt.layers.attention.qsa.sparse_attn import (
    qwen_sparse_fa2_cu_seqlens_triton,
    qwen_sparse_kv_extraction_compact_triton,
    qwen_sparse_valid_counts_triton,
    sparse_gqa_fwd_interface_triton,
    sparse_gqa_fwd_interface_triton_ck,
)
"""
IMPORT_REPLACEMENT = IMPORT_ANCHOR + (
    "from sglang.srt.layers.attention import qsa_nvfp4_kv  # dspark: NVFP4 KV cache\n"
)

PAGED_HEAD_ANCHOR = """        pool = self.token_to_kv_pool
        k_buffer = pool.get_key_buffer(layer.layer_id)
        v_buffer = pool.get_value_buffer(layer.layer_id)
        if not q.is_cuda:
            metadata = self._resolve_metadata(forward_batch)
            slots = self._logical_to_physical(topk_indices, metadata)
            output = qsa_sparse_attention(q, k_buffer, v_buffer, slots, layer.scaling)
            return output.reshape(q.shape[0], -1)
"""
PAGED_HEAD_REPLACEMENT = """        pool = self.token_to_kv_pool
        fp4_kv = qsa_nvfp4_kv.try_fp4_view(pool, layer.layer_id)
        if fp4_kv is None:
            k_buffer = pool.get_key_buffer(layer.layer_id)
            v_buffer = pool.get_value_buffer(layer.layer_id)
        else:
            k_buffer = v_buffer = None
        if not q.is_cuda:
            metadata = self._resolve_metadata(forward_batch)
            if fp4_kv is not None:
                # Whole-pool plain dequant (slow; CPU fallback only).
                k_buffer = pool.get_key_buffer(layer.layer_id)
                v_buffer = pool.get_value_buffer(layer.layer_id)
            slots = self._logical_to_physical(topk_indices, metadata)
            output = qsa_sparse_attention(q, k_buffer, v_buffer, slots, layer.scaling)
            return output.reshape(q.shape[0], -1)
"""

EXTRACTION_ANCHOR = """        packed_k, packed_v = self._get_fa2_scratch(
            scratch_capacity,
            k_buffer.shape[1],
            k_buffer.shape[2],
            k_buffer.dtype,
            k_buffer.device,
        )
        qwen_sparse_kv_extraction_compact_triton(
            k_buffer,
            v_buffer,
            self.req_to_token_pool.req_to_token,
            (
                metadata.row_req_pool_indices
                if metadata.row_req_pool_indices is not None
                else forward_batch.req_pool_indices
            ),
            topk_indices,
            sequence_lens,
            cu_seqlens_k,
            packed_k,
            packed_v,
            batch,
            topk,
        )
"""
EXTRACTION_REPLACEMENT = """        if fp4_kv is not None:
            # dspark NVFP4: compact the packed FP4 rows and the per-block
            # scale rows, then dequantize the gathered rows to BF16.
            packed_k, packed_v = qsa_nvfp4_kv.compact_and_dequant(
                self,
                fp4_kv,
                scratch_capacity,
                (
                    metadata.row_req_pool_indices
                    if metadata.row_req_pool_indices is not None
                    else forward_batch.req_pool_indices
                ),
                topk_indices,
                sequence_lens,
                cu_seqlens_k,
                batch,
                topk,
            )
        else:
            packed_k, packed_v = self._get_fa2_scratch(
                scratch_capacity,
                k_buffer.shape[1],
                k_buffer.shape[2],
                k_buffer.dtype,
                k_buffer.device,
            )
            qwen_sparse_kv_extraction_compact_triton(
                k_buffer,
                v_buffer,
                self.req_to_token_pool.req_to_token,
                (
                    metadata.row_req_pool_indices
                    if metadata.row_req_pool_indices is not None
                    else forward_batch.req_pool_indices
                ),
                topk_indices,
                sequence_lens,
                cu_seqlens_k,
                packed_k,
                packed_v,
                batch,
                topk,
            )
"""

EXTEND_CHUNK_ANCHOR = """        pool = self.token_to_kv_pool
        k_buffer = pool.get_key_buffer(layer.layer_id)
        v_buffer = pool.get_value_buffer(layer.layer_id)
        req_to_token = self.req_to_token_pool.req_to_token
        req_indices = forward_batch.req_pool_indices.tolist()
        k_parts = [
            k_buffer.index_select(
                0, req_to_token[req_indices[i], : sequence_lens[i]].long()
            )
            for i in range(len(sequence_lens))
        ]
        v_parts = [
            v_buffer.index_select(
                0, req_to_token[req_indices[i], : sequence_lens[i]].long()
            )
            for i in range(len(sequence_lens))
        ]
"""
EXTEND_CHUNK_REPLACEMENT = """        pool = self.token_to_kv_pool
        req_to_token = self.req_to_token_pool.req_to_token
        req_indices = forward_batch.req_pool_indices.tolist()
        fp4_kv = qsa_nvfp4_kv.try_fp4_view(pool, layer.layer_id)
        if fp4_kv is not None:
            # dspark NVFP4: gather each request's history from the packed
            # FP4 + scale buffers and dequantize to BF16.
            k_cat, v_cat = qsa_nvfp4_kv.gather_history_fp4(
                fp4_kv, req_to_token, req_indices, sequence_lens
            )
        else:
            k_buffer = pool.get_key_buffer(layer.layer_id)
            v_buffer = pool.get_value_buffer(layer.layer_id)
            k_parts = [
                k_buffer.index_select(
                    0, req_to_token[req_indices[i], : sequence_lens[i]].long()
                )
                for i in range(len(sequence_lens))
            ]
            v_parts = [
                v_buffer.index_select(
                    0, req_to_token[req_indices[i], : sequence_lens[i]].long()
                )
                for i in range(len(sequence_lens))
            ]
            k_cat = torch.cat(k_parts)
            v_cat = torch.cat(v_parts)
"""

EXTEND_CK_ANCHOR = """        output = sparse_gqa_fwd_interface_triton_ck(
            q.contiguous(),
            torch.cat(k_parts),
            torch.cat(v_parts),
"""
EXTEND_CK_REPLACEMENT = """        output = sparse_gqa_fwd_interface_triton_ck(
            q.contiguous(),
            k_cat,
            v_cat,
"""

# ---------------------------------------------------------------------------
# 2. NVFP4 recipe routing: QSA plain-dequant method
# ---------------------------------------------------------------------------
FP4_METHOD = SRT / "layers" / "quantization" / "fp4_kv_cache_quant_method.py"

FP4_METHOD_ANCHOR = """    return KV_CACHE_QUANT_REGISTRY[name](**kwargs)
"""
FP4_METHOD_REPLACEMENT = """    if name == "nvfp4":
        # dspark (DGX Spark / SM121): QSA models consume the NVFP4 pool via
        # plain BF16 dequant reads — no FP8 dequant workspace, no
        # native-FP4 decode. See attention/qsa_nvfp4_kv.py.
        from sglang.srt.layers.attention.qsa_nvfp4_kv import QSANVFP4KVCacheMethod

        return QSANVFP4KVCacheMethod(**kwargs)
    return KV_CACHE_QUANT_REGISTRY[name](**kwargs)
"""

# ---------------------------------------------------------------------------
# 3. server_args: allow nvfp4 KV for QSA hybrid models
# ---------------------------------------------------------------------------
SERVER_ARGS = SRT / "server_args.py"

SERVER_ARGS_ANCHOR = """        if is_cuda():
            if self.kv_cache_dtype == "nvfp4" and not (
                is_sm100_supported() or is_sm120_supported()
            ):
                raise RuntimeError(
                    "--kv-cache-dtype=nvfp4 requires Blackwell SM100 or SM120. "
                    "Use --kv-cache-dtype=fp4_mx_block16 for the block-size-16 FP4 recipe."
                )
"""
SERVER_ARGS_REPLACEMENT = SERVER_ARGS_ANCHOR + """            # dspark (DGX Spark / SM121, see qsa_nvfp4_kv): on QSA hybrid
            # models the full-attention layers read the FP4 pool through the
            # QSA backend's Triton plain-dequant path; --attention-backend
            # only selects the GDN linear-attention kernels, so the MHA
            # allow-list below does not apply.
            if self.kv_cache_dtype == "nvfp4":
                try:
                    from sglang.srt.layers.attention.qsa.config import is_qwen_qsa

                    if is_qwen_qsa(self.get_model_config().hf_config):
                        return
                except Exception:
                    pass
"""

# ---------------------------------------------------------------------------
# 4. pool_configurator: no FP8 workspace share for QSA FP4 cell size
# ---------------------------------------------------------------------------
POOL_CFG = SRT / "model_executor" / "pool_configurator.py"

POOL_CFG_ANCHOR = """                # FP4 prefill uses one shared FP8 dequant workspace across layers.
                cell_size += n * k * 2 * kv_size
"""
POOL_CFG_REPLACEMENT = """                # FP4 prefill uses one shared FP8 dequant workspace across
                # layers — except on the QSA path (dspark qsa_nvfp4_kv),
                # whose method allocates no FP8 workspace.
                _is_qsa_kv4 = False
                try:
                    from sglang.srt.layers.attention.qsa.config import is_qwen_qsa

                    _is_qsa_kv4 = is_qwen_qsa(model_config.hf_config)
                except Exception:
                    pass
                if not _is_qsa_kv4:
                    cell_size += n * k * 2 * kv_size
"""

# ---------------------------------------------------------------------------
# 5. memory_pool: NVFP4 store must use on-device global scales
# ---------------------------------------------------------------------------
POOL = SRT / "mem_cache" / "memory_pool.py"

QUANT_SCALES_ANCHOR = """    def _quantized_scales(self, global_layer_id: int, k_scale, v_scale):
        if k_scale is None and hasattr(self.quant_method, "k_scales_gpu"):
            k_scale = self.quant_method.k_scales_gpu[
                global_layer_id : global_layer_id + 1
            ]
            v_scale = self.quant_method.v_scales_gpu[
                global_layer_id : global_layer_id + 1
            ]
        return k_scale, v_scale
"""
QUANT_SCALES_REPLACEMENT = """    def _quantized_scales(self, global_layer_id: int, k_scale, v_scale):
        # dspark qsa_nvfp4_kv: HybridTokenToKVPool.set_kv_buffer defaults
        # k_scale=v_scale=1.0 (Python floats). That skips the on-device
        # per-layer NVFP4 global scales, and NVFP4KVQuantizeUtil.quantize
        # then does torch.tensor(..., device=cuda) — illegal during CUDA
        # graph capture. Host scalars are treated as unset on nvfp4 only.
        use_gpu = hasattr(self.quant_method, "k_scales_gpu")
        nvfp4 = getattr(self.quant_method, "name", None) == "nvfp4"
        host_scalar = nvfp4 and not (
            torch.is_tensor(k_scale) and k_scale.is_cuda
        )
        if use_gpu and (k_scale is None or host_scalar):
            k_scale = self.quant_method.k_scales_gpu[
                global_layer_id : global_layer_id + 1
            ]
            v_scale = self.quant_method.v_scales_gpu[
                global_layer_id : global_layer_id + 1
            ]
        return k_scale, v_scale
"""

# ---------------------------------------------------------------------------
# 6. sparse_attn.py: Triton GQA cannot tl.dot fp8e4nv on SM121
# ---------------------------------------------------------------------------
SPARSE_ATTN = SRT / "layers" / "attention" / "qsa" / "sparse_attn.py"
FP8_DOT_MARKER = "dspark: SM121 Triton cannot tl.dot fp8e4nv"

GQA_DOT_ANCHOR = """        keys = tl.load(
            k_base + token[None, :] * sk_n + offs_d[:, None] * sk_d,
            mask=valid[None, :],
            other=0.0,
        )
        values = tl.load(
            v_base + token[:, None] * sv_n + offs_d[None, :] * sv_d,
            mask=valid[:, None],
            other=0.0,
        )
        scores = tl.where(valid[None, :], tl.dot(q_values, keys), -float("inf"))
        next_max = tl.maximum(max_value, tl.max(scores, 1))
        alpha = tl.math.exp2(max_value - next_max)
        probabilities = tl.math.exp2(scores - next_max[:, None])
        accumulator = tl.dot(
            probabilities.to(values.dtype), values, accumulator * alpha[:, None]
        )
"""
GQA_DOT_REPLACEMENT = """        keys = tl.load(
            k_base + token[None, :] * sk_n + offs_d[:, None] * sk_d,
            mask=valid[None, :],
            other=0.0,
        )
        values = tl.load(
            v_base + token[:, None] * sv_n + offs_d[None, :] * sv_d,
            mask=valid[:, None],
            other=0.0,
        )
        # dspark: SM121 Triton cannot tl.dot fp8e4nv (fp8 KV cache).
        keys = keys.to(tl.float32)
        values = values.to(tl.float32)
        q_dot = q_values.to(tl.float32)
        scores = tl.where(valid[None, :], tl.dot(q_dot, keys), -float("inf"))
        next_max = tl.maximum(max_value, tl.max(scores, 1))
        alpha = tl.math.exp2(max_value - next_max)
        probabilities = tl.math.exp2(scores - next_max[:, None])
        accumulator = tl.dot(probabilities, values, accumulator * alpha[:, None])
"""


def patch_count(path, anchor, replacement, expected, marker):
    s = path.read_text()
    if marker in s:
        print(f"{path.name}: already patched")
        return
    count = s.count(anchor)
    assert count == expected, (
        f"{path.name}: anchor matched {count} times (want {expected}):\n{anchor}"
    )
    path.write_text(s.replace(anchor, replacement))
    print(f"{path.name}: patched ({expected} sites)")


def main() -> None:
    patch(
        BACKEND,
        [
            (IMPORT_ANCHOR, IMPORT_REPLACEMENT),
            (PAGED_HEAD_ANCHOR, PAGED_HEAD_REPLACEMENT),
            (EXTRACTION_ANCHOR, EXTRACTION_REPLACEMENT),
            (EXTEND_CHUNK_ANCHOR, EXTEND_CHUNK_REPLACEMENT),
            (EXTEND_CK_ANCHOR, EXTEND_CK_REPLACEMENT),
        ],
    )
    patch(FP4_METHOD, [(FP4_METHOD_ANCHOR, FP4_METHOD_REPLACEMENT)])
    patch(SERVER_ARGS, [(SERVER_ARGS_ANCHOR, SERVER_ARGS_REPLACEMENT)])
    patch(POOL_CFG, [(POOL_CFG_ANCHOR, POOL_CFG_REPLACEMENT)])
    patch(POOL, [(QUANT_SCALES_ANCHOR, QUANT_SCALES_REPLACEMENT)])
    patch_count(
        SPARSE_ATTN,
        GQA_DOT_ANCHOR,
        GQA_DOT_REPLACEMENT,
        expected=2,
        marker=FP8_DOT_MARKER,
    )
    print("NVFP4 KV patches applied")


if __name__ == "__main__":
    main()
