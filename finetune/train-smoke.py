#!/usr/bin/env python3
"""
train-smoke.py, proves the fine-tuning lane end to end in ~3 minutes.

LoRA-tunes a small Qwen on 60 synthetic instruction rows and asserts the loss
actually falls. If this passes, the whole chain works on your Spark: CUDA 13 on
sm_121a, bitsandbytes/triton kernels, gradient checkpointing, adapter save.
Then scale the same recipe: swap MODEL_ID for a 27B (LoRA) or 70B (QLoRA).

Run inside the finetune container:
  docker exec sparkduet-finetune python3 /work/finetune/train-smoke.py
"""
import json, os, time

MODEL_ID = os.environ.get("SMOKE_MODEL", "Qwen/Qwen3-0.6B")
MAX_STEPS = int(os.environ.get("SMOKE_STEPS", "30"))
OUT = os.environ.get("SMOKE_OUT", "/outputs/smoke-lora")

t0 = time.time()
from unsloth import FastLanguageModel  # noqa: E402 (unsloth patches transformers on import)
from datasets import Dataset            # noqa: E402
from trl import SFTConfig, SFTTrainer   # noqa: E402

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_ID, max_seq_length=1024, load_in_4bit=True)
model = FastLanguageModel.get_peft_model(
    model, r=16, lora_alpha=16, lora_dropout=0.0,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    use_gradient_checkpointing="unsloth")

rows = [{"text": f"### Instruction:\nState the square of {i}.\n\n### Response:\n"
                 f"The square of {i} is {i*i}."} for i in range(2, 62)]
ds = Dataset.from_list(rows)

trainer = SFTTrainer(
    model=model, tokenizer=tokenizer, train_dataset=ds,
    args=SFTConfig(per_device_train_batch_size=2, gradient_accumulation_steps=2,
                   max_steps=MAX_STEPS, learning_rate=2e-4, logging_steps=5,
                   output_dir=OUT, report_to="none", seed=7))
result = trainer.train()

hist = [h["loss"] for h in trainer.state.log_history if "loss" in h]
first, last = hist[0], hist[-1]
model.save_pretrained(OUT)
tokenizer.save_pretrained(OUT)

summary = {"model": MODEL_ID, "steps": MAX_STEPS,
           "loss_first": round(first, 4), "loss_last": round(last, 4),
           "wall_s": round(time.time() - t0, 1), "adapter": OUT}
print(json.dumps(summary, indent=2))

assert last < first * 0.9, f"loss did not fall ({first} -> {last}); lane is NOT healthy"
print("FINETUNE LANE OK, loss fell, adapter saved")
