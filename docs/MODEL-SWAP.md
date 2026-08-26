# Swapping models: the recipe outlives the model

DeepSeek-V4-Flash is the flagship *today*. The design assumption is that a
better checkpoint appears every few weeks and you should be able to adopt it
in an afternoon without touching the architecture. Everything model-specific
lives in `configs/sparkduet.env`; everything structural (lanes, router,
doctor, bench, revert) is model-agnostic.

## The branch convention

- **`main`** always serves the current recommended default, with its measured
  artifacts in `results/`.
- **`model/<name>`** branches hold one complete, tested recipe per model:
  env values, any lane-compose deltas, and the benchmark artifacts produced
  on that model. Examples: `model/deepseek-v4-flash-0731`,
  `model/qwen3.8-27b`, `lane-n-flash-next`.
- When a new flagship lands, cut `model/<new-name>` from `main`, tune it
  there, commit its artifacts, and fast-forward `main` once it beats the
  incumbent on your workloads. The old branch stays checkable forever.

```bash
git checkout model/deepseek-v4-flash-0731   # today's default, pinned
git checkout -b model/next-big-thing        # start the next recipe
```

## The 30-minute swap checklist

1. **Fit first.** One node has ~121 GiB GPU-usable. Weights fit one node ->
   Lane F (two replicas). Weights need 122-242 GiB -> Lane D (TP=2). Bigger ->
   it does not fit this pair; stop here. `docs/MODELS.md` has the arithmetic.
2. **Stage weights on both nodes** (Lane D) or the serving node (Lane F):

   ```bash
   huggingface-cli download <org>/<model> --local-dir /srv/ai/models/llm/<model>
   rsync -a /srv/ai/models/llm/<model> worker:/srv/ai/models/llm/
   ```

3. **Edit `configs/sparkduet.env`.** For Lane D that is `DS_MODEL`,
   `DS_SERVED_NAME`, `DS_REVISION`, and the `D_*` shape knobs
   (`D_MAX_MODEL_LEN`, `D_MAX_NUM_SEQS`, `D_GPU_MEM_UTIL`, `D_KV_DTYPE`,
   `D_MTP_NUM_TOKENS`). For Lane F: `F_MODEL` and the `F_*` set. Speculation
   only helps if the checkpoint ships a draft head (MTP/DSpark); set
   `*_MTP_NUM_TOKENS=0` otherwise.
4. **Gate the fabric, then start:**

   ```bash
   ./scripts/sparkduetctl.sh doctor && ./scripts/nccl-check.sh   # TP=2 only
   ./scripts/sparkduetctl.sh switch depth                        # drains, swaps, warms
   ./scripts/sparkduetctl.sh switch next                         # Flash-Next, lane-n-flash-next
   ```

   `switch` captures the incumbent state first; `revert` restores it if the
   new model disappoints.
5. **Read the boot line, not the README.** `Available KV cache memory` and
   `GPU KV cache size ... tokens` tell you what your settings actually
   bought. Record them.
6. **Benchmark before you brag:**

   ```bash
   python3 scripts/bench.py --suite standard
   ```

   The artifact lands in `results/` dated and labeled. Commit it on the
   `model/*` branch. A recipe without artifacts is a rumor.

## What usually needs retuning per model

| Knob | Why it moves |
|---|---|
| `D_GPU_MEM_UTIL` | bigger weights leave less KV; shared nodes need host headroom (we run 0.78 shared, 0.835 works dedicated) |
| `D_MTP_NUM_TOKENS` | acceptance is model- and workload-specific; run `specadvisor.py`, apply its recommended k at a quiet restart |
| `D_KV_DTYPE` | DeepSeek-family MLA supports `nvfp4_ds_mla`; dense models usually want `auto` |
| `D_MAX_MODEL_LEN` | a ceiling, not free: longer ceilings admit requests that can monopolize the KV pool |
| chat template flags | thinking modes differ per family; verify tool-calling against your harness before rollout |

## Serving your own fine-tunes

A merged fine-tune is just another model: merge the adapter
(`finetune/README.md`), drop the folder under `/srv/ai/models/llm/`, point
`F_MODEL` at it (one-node models fleet better than they TP), and follow the
same checklist. GGUF exports can instead go to the llama-swap library for
on-demand loading without touching the resident lanes.
