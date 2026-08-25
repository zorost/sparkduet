# Fine-tuning on the pair

Serving is only half the point of owning two Sparks. The other half: the node that
is not serving is a 128 GB unified-memory training box, and unified memory is a
genuinely different fine-tuning proposition than a 24 GB discrete card.

## The lane

One container per node, from `FINETUNE_IMAGE` (NVIDIA's Unsloth build for DGX
Spark). It idles until you exec a job into it, so nothing dies with your SSH
session:

```bash
cd finetune
docker compose --env-file ../sparkduet.env -f finetune.compose.yml up -d
docker exec sparkduet-finetune python3 /work/finetune/train-smoke.py   # ~3 min gate
```

The smoke test LoRA-tunes a 0.6B Qwen on synthetic rows and **asserts the loss
falls**. Green means the whole chain works on your silicon: CUDA 13 on sm_121a,
quantized kernels, gradient checkpointing, adapter save. Only then spend hours
on a real run.

## What fits in 128 GB unified memory

| Recipe | Model size | Notes |
|---|---|---|
| QLoRA 4-bit | up to ~70B | rank 16-32, seq 4-8K; the headline capability |
| LoRA 16-bit | up to ~27B | comfortable, fast iteration |
| Full fine-tune | up to ~7B | gradient-checkpointed |

The serving stack reads these tables too: a LoRA adapter trained here can be
merged and served by Lane F on the same node that trained it.

## Scheduling against serving

Training and Lane D serving both want the whole GPU. The clean patterns:

- **Serve on the pair, train never**, Lane D owns both nodes.
- **Serve on one, train on one**, Lane F (Qwen) serves on the head while the
  worker trains. The most productive daily setup.
- **Train on both**, pause serving; two independent jobs (sweeps) or one
  distributed job over the 200G link.

`sparkduetctl.sh status` shows what is running where before you commit a node.

## Two-node distributed training

The 200G RoCE link that carries Lane D's NCCL traffic serves torchrun equally
well. For models that need more than one node's memory to train, or to halve
epoch time on large corpora:

```bash
# head (rank 0)
docker exec sparkduet-finetune torchrun --nnodes 2 --node-rank 0 \
  --master-addr $MASTER_ADDR --master-port 29500 your_train.py
# worker (rank 1)
docker exec sparkduet-finetune torchrun --nnodes 2 --node-rank 1 \
  --master-addr $MASTER_ADDR --master-port 29500 your_train.py
```

Export the same `NCCL_*` values from `sparkduet.env` inside both containers so
traffic rides the fabric, not the management LAN. Expect near-linear scaling on
compute-bound fine-tunes; gradient-sync-bound jobs see the ~9-10 GB/s effective
link ceiling first (measure with the 30-step smoke before committing a weekend).

## From adapter to serving

```bash
# merge LoRA into the base, then serve the merged dir with Lane F
docker exec sparkduet-finetune python3 - <<'PY'
from peft import AutoPeftModelForCausalLM
m = AutoPeftModelForCausalLM.from_pretrained("/outputs/smoke-lora")
m.merge_and_unload().save_pretrained("/outputs/smoke-merged")
PY
```

Point `F_MODEL` at the merged directory and `sparkduetctl.sh start fleet`. Your
fine-tune is now behind the same OpenAI-compatible endpoint as everything else.

---

## Your first real fine-tune, start to finish

The smoke test proved the chain. This section walks one real run on a small
model (Qwen3-4B), because a 20-minute 4B iteration teaches you everything you
need before you spend a night on a 27B. Every command is copy-pasteable.

### 1. Shape your data

One JSONL file, one training example per line, chat format:

```bash
mkdir -p finetune/data
cat > finetune/data/style.jsonl <<'EOF'
{"messages": [{"role": "user", "content": "Summarize: the fabric link dropped to TCP."}, {"role": "assistant", "content": "NCCL fell back to sockets; RDMA was not visible in the container. Fix: map /dev/infiniband and add IPC_LOCK."}]}
{"messages": [{"role": "user", "content": "Summarize: KV pool shrank after the vision sidecar."}, {"role": "assistant", "content": "The sidecar lowered gpu_memory_utilization, so fewer KV blocks were allocated. Fix: disable the sidecar or accept the smaller pool."}]}
EOF
```

Aim for 200-2,000 rows of exactly the behavior you want. Quality beats
volume: fifty precise examples of your ticket-triage format outperform five
thousand scraped ones. Pre-rendered `{"text": "..."}` rows also work if you
already have a template.

### 2. Train

```bash
docker exec sparkduet-finetune python3 /work/finetune/train-lora.py \
  --model Qwen/Qwen3-4B \
  --data /work/finetune/data/style.jsonl \
  --out /outputs/style-lora-v1 \
  --seq-len 4096 --rank 16 --epochs 2
```

The trainer prints a JSON summary at the end: first loss, last loss, wall
time, adapter path. Loss should fall clearly (the smoke run went 2.12 to
0.21). If it does not, your data format or learning rate is wrong; fix that
before scaling up.

Scaling the same command:

| Target | Flags | Expect |
|---|---|---|
| 4B iteration | as above | minutes per epoch |
| 27B LoRA | `--model Qwen/Qwen3.8-27B --full-precision --batch 1 --accum 8` | comfortable in 128 GB |
| 70B QLoRA | `--model <70B> --batch 1 --accum 16` (4-bit is the default) | the headline capability |

### 3. Watch it without babysitting it

```bash
docker exec sparkduet-finetune nvidia-smi --query-gpu=utilization.gpu,temperature.gpu,power.draw --format=csv -l 30
```

Training pins the GPU near 100% and the package will run hot; that is the
one workload where sustained heat is the deal you signed. Schedule long runs
for hours you are not serving (see "Scheduling against serving" above), and
stop the lane afterward so the box idles cool.

### 4. Merge and serve it

Merge as shown at the top of this section, then either:

- **Resident:** set `F_MODEL=/outputs/style-merged-v1` in
  `configs/sparkduet.env` and `sparkduetctl.sh start fleet`, or
- **On-demand:** export GGUF for the llama-swap library so it loads only when
  called and unloads after:

```bash
docker exec sparkduet-finetune python3 - <<'PY'
from unsloth import FastLanguageModel
m, tok = FastLanguageModel.from_pretrained("/outputs/style-lora-v1", max_seq_length=4096)
m.save_pretrained_gguf("/outputs/style-v1-gguf", tok, quantization_method="q5_k_m")
PY
```

### 5. Prove it changed something

Ask the base and the fine-tune the same held-out questions through the same
endpoint and diff the answers. Two curls beat a leaderboard for a
task-specific tune. Keep the adapter (`/outputs/style-lora-v1`) even after
merging; adapters are 100x smaller than merges and re-merge onto updated
bases.
