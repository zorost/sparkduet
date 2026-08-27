#!/usr/bin/env python3
"""GB10 GLM-5.3-Flash: use SM90 NoPE MLA, not the packed SM120 path.

Stock glm53-flash-arm64-cu130 lists FLASHINFER_MLA_SPARSE_SM120 first on
capability 12. This LibertAIDAI checkpoint is NoPE (qk_rope_head_dim=0),
so concat_and_cache_mla dies with pe_dim must be 64 for fp8_ds_mla.

Each edit is anchored to one exact stock string. Missing anchors skip.
"""
from __future__ import annotations

from pathlib import Path

VLLM = Path("/usr/local/lib/python3.12/dist-packages/vllm")
FI = Path("/usr/local/lib/python3.12/dist-packages/flashinfer")
LOG = "[glm53-sm90]"


def apply_once(path: Path, old: str, new: str, label: str) -> str:
    if not path.is_file():
        print("%s skip %s (missing %s)" % (LOG, label, path.name))
        return "missing"
    text = path.read_text()
    n_old = text.count(old)
    n_new = text.count(new)
    if n_old == 0 and n_new == 1:
        print("%s skip %s (already)" % (LOG, label))
        return "skipped"
    if n_old != 1:
        print("%s skip %s (old=%d new=%d)" % (LOG, label, n_old, n_new))
        return "skip"
    path.write_text(text.replace(old, new, 1))
    print("%s applied %s" % (LOG, label))
    return "applied"


def main() -> int:
    apply_once(
        VLLM / "platforms/cuda.py",
        """        elif device_capability.major == 12:
            return [
                AttentionBackendEnum.TRITON_MLA,
                AttentionBackendEnum.FLASHINFER_MLA_SPARSE_SM120,
            ]""",
        """        elif device_capability.major == 12:
            return [
                AttentionBackendEnum.TRITON_MLA,
                AttentionBackendEnum.FLASHINFER_MLA_SPARSE_SM90,
                AttentionBackendEnum.FLASHINFER_MLA_SPARSE_SM120,
            ]""",
        "cuda.py SM90 on cap 12",
    )
    sm90 = VLLM / "v1/attention/backends/mla/flashinfer_mla_sparse_sm90.py"
    apply_once(
        sm90,
        "    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:\n        return capability.major == 9\n",
        "    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:\n        return capability.major in (9, 12)\n",
        "sm90 capability 9+12",
    )
    apply_once(sm90, '            backend="fa3",\n', '            backend=("fa3" if torch.cuda.get_device_capability()[0] == 9 else "fa2"),\n', "sm90 FA2 on GB10")
    apply_once(
        sm90,
        """        if not has_flashinfer_sm90_nope_mla():
            return (
                "FLASHINFER_MLA_SPARSE_SM90 requires FlashInfer with SM90 "
                "MLA support (ckv_scale_arr in "
                "BatchMLAPagedAttentionWrapper.run, FlashInfer >= 0.6.18)"
            )""",
        """        if kv_cache_dtype in ("fp8", "fp8_e4m3") and not has_flashinfer_sm90_nope_mla():
            return (
                "FLASHINFER_MLA_SPARSE_SM90 fp8 KV requires FlashInfer with "
                "SM90 MLA support (ckv_scale_arr in "
                "BatchMLAPagedAttentionWrapper.run, FlashInfer >= 0.6.18)"
            )""",
        "sm90 fp8 gate",
    )
    apply_once(
        VLLM / "platforms/cuda.py",
        """    @classmethod
    def is_arch_support_pdl(cls) -> bool:
        try:
            device = torch.cuda.current_device()
            major, _ = torch.cuda.get_device_capability(device)
        except Exception:
            return False
        return major >= 9
""",
        """    @classmethod
    def is_arch_support_pdl(cls) -> bool:
        try:
            device = torch.cuda.current_device()
            major, _ = torch.cuda.get_device_capability(device)
        except Exception:
            return False
        return major in (9, 10)
""",
        "PDL off on SM12x",
    )
    idx = VLLM / "model_executor/layers/sparse_attn_indexer_kpool.py"
    apply_once(
        idx,
        "                pool_topk = torch.empty(\n                    (num_rows, select_k), dtype=torch.int32, device=logits.device\n                )\n",
        "                pool_topk = torch.full(\n                    (num_rows, select_k), -1, dtype=torch.int32, device=logits.device\n                )\n",
        "indexer topk -1 prefill",
    )
    apply_once(
        idx,
        "            pool_topk = torch.empty(\n                (num_rows, select_k), dtype=torch.int32, device=logits.device\n            )\n",
        "            pool_topk = torch.full(\n                (num_rows, select_k), -1, dtype=torch.int32, device=logits.device\n            )\n",
        "indexer topk -1 decode",
    )
    apply_once(
        VLLM / "models/glm5next/nvidia/ops/kpool_compress.py",
        "    hist_out = tl.where(pid >= 0, hist_val, -1)\n",
        "    hist_out = tl.where((pid >= 0) & (pid < pool_len), hist_val, -1)\n",
        "kpool pid clamp",
    )
    apply_once(
        FI / "data/include/flashinfer/attention/mla.cuh",
        "    constexpr uint32_t CTA_TILE_KV = 16;                                                \\\n"
        "    constexpr bool QK_SHARD = true;                                                     \\\n",
        "    constexpr uint32_t CTA_TILE_KV = 32;                                                \\\n"
        "    constexpr bool QK_SHARD = true;                                                     \\\n",
        "FA2 CTA tile restore 32 (QK_SHARD)",
    )
    # ponytail: CTA_TILE_KV=16 + QK_SHARD makes NUM_MMA_KV/2=0; nvcc rejects
    # zero-sized s_frag/p_f16. Stock 32 is the working tile. Upgrade: drop
    # this revert once the image no longer carries the 16/true pair.
    apply_once(
        FI / "mla/_core.py",
        "            major, minor = get_compute_capability(self.device)\n            if major != 9:\n",
        "            major, minor = get_compute_capability(self.device)\n            if major not in (9, 12):\n",
        "FA2 fp8 SM12 gate",
    )
    # 0.6.18 FA2: cap fp8 tile so GB10 ~101KB smem does not overflow.
    # Absent on 0.6.17 (skip). Apply after glm-entry upgrades FI.
    apply_once(
        FI / "data/include/flashinfer/attention/mla.cuh",
        "    constexpr uint32_t EFF_CTA_TILE_KV = std::is_same_v<DTypeKV, __nv_fp8_e4m3> ? 32 : CTA_TILE_KV;\n",
        "    constexpr uint32_t EFF_CTA_TILE_KV = std::is_same_v<DTypeKV, __nv_fp8_e4m3> ? (CTA_TILE_KV < 32u ? CTA_TILE_KV : 32u) : CTA_TILE_KV;\n",
        "FA2 fp8 CTA tile cap",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
