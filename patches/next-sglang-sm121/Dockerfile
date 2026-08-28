# DSpark kernel-work image for Qwen3.8-Flash-Next-NVFP4 on DGX Spark (SM121/GB10).
# sglang#36806 (exclude SM121 from TRT-LLM sparse decode) + #36845 (Triton
# packed-varlen fallback) + NVFP4 KV for the QSA path. See start.sh header.
FROM lmsysorg/sglang:qwen38flashnext

COPY sm121_varlen.py /sgl-workspace/sglang/python/sglang/srt/layers/attention/qsa/sm121_varlen.py
COPY qsa_nvfp4_kv.py /sgl-workspace/sglang/python/sglang/srt/layers/attention/qsa_nvfp4_kv.py
COPY apply_nvfp4_patches.py /tmp/apply_nvfp4_patches.py
RUN python3 - <<'PYEOF'
p = "/sgl-workspace/sglang/python/sglang/srt/layers/attention/qwen_sparse_attn_backend.py"
s = open(p).read()

# sglang#36845: SM121 packed-varlen Triton fallback (not FA4 CuTe).
if "qsa.sm121_varlen" not in s:
    anchor = "    try:\n        from flash_attn import flash_attn_varlen_func"
    assert anchor in s, "flash_attn varlen anchor not found — upstream image layout changed"
    insert = (
        "    from sglang.srt.utils import is_sm121\n"
        "\n"
        "    if is_sm121():\n"
        "        from sglang.srt.layers.attention.qsa.sm121_varlen import (\n"
        "            qsa_sm121_varlen_attention,\n"
        "        )\n"
        "\n"
        "        return qsa_sm121_varlen_attention\n"
    )
    s = s.replace(anchor, insert + anchor, 1)

# sglang#36806: never take FlashInfer TRT-LLM sparse decode on SM121.
# A newer base image may re-enable it (token-id-0 at long context).
marker = "dspark: SM121 must not use TRT-LLM sparse decode"
if marker not in s:
    fn = "def _resolve_trtllm_sparse_decode():"
    i = s.find(fn)
    assert i >= 0, "trtllm resolver not found — upstream image layout changed"
    ds = s.find('"""', i)
    assert ds > 0, "trtllm resolver docstring not found"
    ds_end = s.find('"""', ds + 3)
    assert ds_end > 0, "trtllm resolver docstring unterminated"
    ds_end += 3
    s = s[:ds_end] + (
        "\n    from sglang.srt.utils import is_sm121\n"
        "\n"
        "    # dspark: SM121 must not use TRT-LLM sparse decode\n"
        "    # (sglang#36806 / #36845). That path silently emits token id 0\n"
        "    # at long context on GB10.\n"
        "    if is_sm121():\n"
        "        return None\n"
    ) + s[ds_end:]

open(p, "w").write(s)
print("qwen_sparse_attn_backend.py patched for SM121 (sglang#36806 + #36845)")
PYEOF
RUN python3 /tmp/apply_nvfp4_patches.py && rm -f /tmp/apply_nvfp4_patches.py
