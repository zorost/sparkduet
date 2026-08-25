#!/usr/bin/env python3
"""
train-lora.py, the real trainer behind the walkthrough in finetune/README.md.

Same working chain as train-smoke.py (which gates it), parameterized for your
model and your data. Dataset is a JSONL file where each line is either
  {"messages": [{"role": "user", "content": ...}, {"role": "assistant", ...}]}
or a pre-rendered {"text": "..."} row.

Run inside the finetune container:
  docker exec sparkduet-finetune python3 /work/finetune/train-lora.py \
    --model Qwen/Qwen3-4B --data /work/finetune/data/my.jsonl --out /outputs/my-lora
"""
import argparse, json, time

p = argparse.ArgumentParser()
p.add_argument("--model", required=True, help="HF id or local path")
p.add_argument("--data", required=True, help="JSONL: messages[] or text rows")
p.add_argument("--out", required=True, help="adapter output dir (under /outputs)")
p.add_argument("--seq-len", type=int, default=4096)
p.add_argument("--rank", type=int, default=16)
p.add_argument("--epochs", type=float, default=2.0)
p.add_argument("--max-steps", type=int, default=0, help=">0 overrides epochs")
p.add_argument("--lr", type=float, default=2e-4)
p.add_argument("--batch", type=int, default=2)
p.add_argument("--accum", type=int, default=4)
p.add_argument("--full-precision", action="store_true",
               help="16-bit LoRA (<=27B) instead of 4-bit QLoRA (<=70B)")
args = p.parse_args()

t0 = time.time()
from unsloth import FastLanguageModel  # noqa: E402 (patches transformers on import)
from datasets import Dataset            # noqa: E402
from trl import SFTConfig, SFTTrainer   # noqa: E402

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=args.model, max_seq_length=args.seq_len,
    load_in_4bit=not args.full_precision)
model = FastLanguageModel.get_peft_model(
    model, r=args.rank, lora_alpha=args.rank, lora_dropout=0.0,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    use_gradient_checkpointing="unsloth")

rows = []
with open(args.data) as fh:
    for line in fh:
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if "text" in row:
            rows.append({"text": row["text"]})
        elif "messages" in row:
            rows.append({"text": tokenizer.apply_chat_template(
                row["messages"], tokenize=False, add_generation_prompt=False)})
if not rows:
    raise SystemExit(f"no usable rows in {args.data}")
ds = Dataset.from_list(rows)

cfg = dict(per_device_train_batch_size=args.batch,
           gradient_accumulation_steps=args.accum,
           learning_rate=args.lr, logging_steps=5,
           output_dir=args.out, report_to="none", seed=7)
if args.max_steps > 0:
    cfg["max_steps"] = args.max_steps
else:
    cfg["num_train_epochs"] = args.epochs

trainer = SFTTrainer(model=model, tokenizer=tokenizer, train_dataset=ds,
                     args=SFTConfig(**cfg))
trainer.train()

hist = [h["loss"] for h in trainer.state.log_history if "loss" in h]
model.save_pretrained(args.out)
tokenizer.save_pretrained(args.out)
print(json.dumps({"model": args.model, "rows": len(rows),
                  "loss_first": round(hist[0], 4) if hist else None,
                  "loss_last": round(hist[-1], 4) if hist else None,
                  "wall_s": round(time.time() - t0, 1),
                  "adapter": args.out}, indent=2))
if len(hist) >= 2 and hist[-1] >= hist[0]:
    print("WARNING: loss did not fall; check data format and learning rate")
