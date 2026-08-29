#!/usr/bin/env python3
"""CPU-only check that Lane N is wired in env, compose, and ctl."""
from pathlib import Path

root = Path(__file__).resolve().parent.parent
fails = 0


def check(name, cond):
    global fails
    print(("PASS" if cond else "FAIL"), name)
    fails += 0 if cond else 1


env = (root / "configs/sparkduet.env.example").read_text()
ctl = (root / "scripts/sparkduetctl.sh").read_text()
prep = (root / "scripts/prepare-models.sh").read_text()
compose = (root / "configs/lane-next.compose.yml").read_text()

check("compose exists", (root / "configs/lane-next.compose.yml").is_file())
check("N_MODEL in env example", "N_MODEL=" in env)
check("N_MAX_NUM_BATCHED_TOKENS=1024", "N_MAX_NUM_BATCHED_TOKENS=1024" in env)
check("ctl starts next", "start_next()" in ctl)
check("ctl switch accepts next", "[depth|fleet|next|glm|split]" in ctl)
check("prepare-models flash-next", "flash-next" in prep)
commands = "\n".join(line for line in compose.splitlines() if not line.lstrip().startswith("#"))
check("no dummy load format", "--load-format dummy" not in commands)
check("qwen parsers", "--tool-call-parser qwen3_coder" in compose and "--reasoning-parser qwen3" in compose)
check("no dspark spec on Lane N", "dspark" not in compose)
check("next-entry", "next-entry.sh" in compose)
check("PLE patch", (root / "patches/next-ple-fp8.py").is_file())
check("modelopt_fp4", "--quantization modelopt_fp4" in commands)
check("eager GB10", "--enforce-eager" in commands)
check("no PLE CPU offload", "VLLM_PLE_CPU_OFFLOAD" not in commands)

# ---- Lane N on SGLang (N_ENGINE=sglang) -------------------------------------
sg_path = root / "configs/lane-next-sglang.compose.yml"
check("sglang compose exists", sg_path.is_file())
sg = sg_path.read_text()
sg_cmd = "\n".join(line for line in sg.splitlines() if not line.lstrip().startswith("#"))

check("ctl selects engine", "N_ENGINE:-vllm" in ctl and "lane-next-sglang.compose.yml" in ctl)
check("ctl tears down both engine files", "lane-next lane-next-sglang" in ctl)
check("engine knob in env example", "N_ENGINE=vllm" in env)
check("sglang image knob", "N_SGLANG_IMAGE=" in env)
# The stock upstream image either fails to compile FA4 CuTe on GB10 or silently
# decodes token id 0. Only the patched derivative may serve this lane.
check("patched image, not stock", "lmsysorg/sglang" not in sg_cmd)
check("sm121 patch context vendored",
      (root / "patches/next-sglang-sm121/Dockerfile").is_file()
      and (root / "patches/next-sglang-sm121/sm121_varlen.py").is_file()
      and (root / "patches/next-sglang-sm121/qsa_nvfp4_kv.py").is_file())
check("sglang no dummy load format", "--load-format dummy" not in sg_cmd)
check("NEXTN speculation on", "--speculative-algorithm NEXTN" in sg_cmd)
# NEXTN drafts from the in-checkpoint MTP layer: topk must be 1 and the draft
# count must be steps + 1, or the chain is misconfigured rather than merely slow.
check("spec chain is steps+1 at topk 1",
      "N_SGLANG_SPEC_STEPS=3" in env and "N_SGLANG_SPEC_TOPK=1" in env
      and "N_SGLANG_SPEC_DRAFT=4" in env)
check("flashinfer attention", "N_SGLANG_ATTENTION=flashinfer" in env)
# The QSA indexer allocates fp32 [chunk x history] per sparse layer per chunk.
# 4096 against a long history froze both boxes upstream.
check("chunked prefill at or below 1024", "N_SGLANG_CHUNK=1024" in env)
check("prefill graphs off on GB10", "--disable-prefill-cuda-graph" in sg_cmd)
check("PLE placement left to the auto-rule",
      "ple-offload-embedding" not in sg_cmd)
# Batches above the top of the pinned decode-graph list run ungraphed.
graph_top = max(int(n) for n in env.split('N_SGLANG_GRAPH_BS="')[1].split('"')[0].split())
max_reqs = int(env.split("N_SGLANG_MAX_REQS=")[1].split()[0].split("#")[0])
check(f"decode graphs cover max requests ({graph_top} >= {max_reqs})", graph_top >= max_reqs)
# Both ranks must get a byte-identical recipe or the TP group will not form.
head, worker = sg.split("next-worker:")[0], sg.split("next-worker:")[1]
flags = lambda t: sorted(w for w in t.split() if w.startswith("--") and w != "--node-rank")
check("both ranks carry the same flags", flags(head.split("command:")[1]) == flags(worker.split("command:")[1]))
check("ranks differ only by node-rank", "--node-rank 0" in head and "--node-rank 1" in worker)
# Same served id and same container names on both engines: swapping the engine
# must not move the 9Router id or break the guard's pair check.
check("served id unchanged by engine", "--served-model-name ${N_SERVED_NAME}" in sg_cmd)
check("house container names", "sparkduet-next-head" in sg and "sparkduet-next-worker" in sg)
check("no per-rank docker restart", 'restart: "no"' in sg)
check("ctl resolves host NCCL", "resolve_next_nccl" in ctl)
check("compose mounts host NCCL dir", "N_SGLANG_NCCL_DIR" in sg)
check("env example documents NCCL dir", "N_SGLANG_NCCL_DIR=" in env)
# Mia's agent workaround and the GLM lanes: thinking off unless the client
# asks. Without this, Cursor Flash-Next thinks until max_tokens.
check("thinking off by default",
      "--default-chat-template-kwargs" in sg_cmd
      and "enable_thinking" in sg_cmd)
check("qwen3_coder on sglang", "--tool-call-parser ${N_SGLANG_TOOL_PARSER:-qwen3_coder}" in sg)
check("sanitize NaN logits", "SGLANG_SANITIZE_NAN_LOGITS" in sg)
varlen = (root / "patches/next-sglang-sm121/sm121_varlen.py").read_text()
apply = (root / "patches/next-sglang-sm121/apply_nvfp4_patches.py").read_text()
check("varlen zeros non-finite QSA",
      "finite = output == output" in varlen and "kv_end > kv_start" in varlen)
check("token0 abort in apply", "def patch_token0_guard" in apply and "TOKEN0_RUN = 16" in apply)
check("token0 skips radix insert", 'getattr(req, "token0_loop", False)' in apply)
check("token0 resets prefix cache", "self.tree_cache.reset()" in apply)
check("native context default", "N_SGLANG_CONTEXT=262144" in env)
check("Tony mamba pin 97", "N_SGLANG_MAMBA_CACHE=97" in env and "--max-mamba-cache-size" in sg_cmd)
check("ReplaySSM spec on", "--enable-linear-replayssm-spec" in sg_cmd)
check("spec attention decode", "--speculative-attention-mode" in sg_cmd)
check("no cuda-graph padding", "--disable-cuda-graph-padding" in sg_cmd)

raise SystemExit(fails)
