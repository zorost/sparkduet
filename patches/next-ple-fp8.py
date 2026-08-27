#!/usr/bin/env python3
"""RadixArk NVFP4 + FP8 PLE on ModelOpt hybrid (GB10 / qwen38-flash-next).

Stock _get_ple_embedding_quant_method only accepts Fp8Config, so a ModelOpt
NVFP4 expert checkpoint builds an unquantized ngram table and then dies on
ngram_embedding.weight_scale. Registering that scale as a Parameter also
collides with load_weights. Detect FP8 PLE via ple_embedding_dtype and keep
the scale as a buffer.
"""
from pathlib import Path

ROOT = Path("/usr/local/lib/python3.12/dist-packages/vllm/models/qwen3_8_flash_next/nvidia/ple_layer.py")
MARK = "ple_embedding_dtype == \"float8_e4m3fn\""


def main() -> None:
    if not ROOT.is_file():
        print("[next-ple] ple_layer.py missing; skip")
        return
    src = ROOT.read_text()
    if MARK in src and "register_buffer(\"weight_scale\"" in src:
        print("[next-ple] already applied")
        return

    old_get = '''def _get_ple_embedding_quant_method(
    quant_config: QuantizationConfig | None,
    prefix: str,
) -> QuantizeMethodBase | None:
    """Select global-scale FP8 only for quantized PLE checkpoint shards."""

    if not isinstance(quant_config, Fp8Config):
        return None
'''
    new_get = '''def _get_ple_embedding_quant_method(
    quant_config: QuantizationConfig | None,
    prefix: str,
    ple_embedding_dtype: str | None = None,
) -> QuantizeMethodBase | None:
    """Select global-scale FP8 only for quantized PLE checkpoint shards."""

    if not isinstance(quant_config, Fp8Config) and ple_embedding_dtype != "float8_e4m3fn":
        return None
    if not isinstance(quant_config, Fp8Config):
        return Qwen3_8FlashNextPLEFp8EmbeddingMethod()
'''
    if old_get not in src:
        raise SystemExit("[next-ple] _get_ple_embedding_quant_method shape changed")
    src = src.replace(old_get, new_get, 1)

    old_call = '''            quant_method=_get_ple_embedding_quant_method(
                quant_config, f"{prefix}.ngram_embedding"
            ),'''
    new_call = '''            quant_method=_get_ple_embedding_quant_method(
                quant_config, f"{prefix}.ngram_embedding",
                getattr(config, "ple_embedding_dtype", None),
            ),'''
    if old_call not in src:
        raise SystemExit("[next-ple] ngram_embedding call site changed")
    src = src.replace(old_call, new_call, 1)

    old_scale = '''        layer.register_parameter("weight_scale", weight_scale)'''
    new_scale = '''        layer.register_buffer("weight_scale", weight_scale.detach(), persistent=True)'''
    if old_scale not in src:
        raise SystemExit("[next-ple] weight_scale register site changed")
    src = src.replace(old_scale, new_scale, 1)

    old_reg = '''            if name.startswith(shard_prefix) and name.endswith(".weight"):'''
    new_reg = '''            if name == "ngram_embedding.weight_scale":
                scale = loaded_weight.detach().to(
                    device=self.ngram_embedding.weight.device, dtype=torch.bfloat16
                )
                if hasattr(self.ngram_embedding, "weight_scale"):
                    self.ngram_embedding.weight_scale.copy_(
                        scale.reshape(self.ngram_embedding.weight_scale.shape)
                    )
                else:
                    self.ngram_embedding.register_buffer(
                        "weight_scale", scale, persistent=True
                    )
                loaded.add(name)
                continue
            if name.startswith(shard_prefix) and name.endswith(".weight"):'''
    if old_reg not in src:
        raise SystemExit("[next-ple] load_weights shard branch changed")
    src = src.replace(old_reg, new_reg, 1)

    ROOT.write_text(src)
    print("[next-ple] patched", ROOT)


if __name__ == "__main__":
    main()
