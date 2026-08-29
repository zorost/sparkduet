### [M-here] Lane D 1M ceiling boot, 2026-08-29

| Knob | Value |
|---|---|
| D_MAX_MODEL_LEN | 1048576 |
| D_GPU_MEM_UTIL | 0.78 |
| D_MTP_NUM_TOKENS | 5 |
| D_KV_DTYPE | nvfp4_ds_mla |
| Engine /v1/models max_model_len | 1048576 |
| kv_cache_size_tokens | 1014644 |
| kv_cache_max_concurrency | 0.968 of 1048576 |
| Picker (85% of pool) | 862447 |

A full 1,048,576 request does not fit. Pickers advertise 862447, not 1M.
Mia's 2.49M pool is util 0.835 on a dedicated pair. Not this boot.

1+1 thinking off: `2` in 0.267 s, 0 reasoning tokens.
