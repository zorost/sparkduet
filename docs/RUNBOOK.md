# RUNBOOK.md, day-2 operations and harness integration

Everything here assumes `sparkduet.env` is filled in and `doctor` passes. The
examples write `HEAD` for the head node's LAN address and `:PORT` for whatever
you configured; substitute your own. Nothing in this file requires a specific
network layout beyond "clients can reach the head node".

---

## 1. The operating loop

```bash
./scripts/sparkduetctl.sh doctor      # before anything, always
./scripts/sparkduetctl.sh status      # what is running where, with health
./scripts/sparkduetctl.sh start depth # or: fleet | split | router
./scripts/warmup.sh                   # after every engine start
./scripts/sparkduetctl.sh switch fleet# drain, stop, start the other lane
./scripts/sparkduetctl.sh stop        # stop sparkduet containers (never rm)
./scripts/sparkduetctl.sh revert      # put back what start displaced
```

Rules the tooling enforces so you do not have to remember them:

- `start` refuses on a failing `doctor`. `doctor` checks SSH to the worker,
  image digests equal on both nodes, weights present, fabric link up, RDMA
  port active, disk and memory headroom.
- Lane D starts worker-first (the headless rank must be listening before the
  head rank dials); stop is head-first. `sparkduetctl` owns that ordering.
- Every `start` writes a capture of what was running before to
  `results/incumbent-<date>/`. `revert` replays it. If you serve something you
  care about, additionally run `capture-incumbent` before your first
  experiment and copy the output off-box.
- `switch` drains via the router when the router is up (`/admin/drain`, wait
  for inflight zero, then swap, then `/admin/undrain`), in-flight requests
  finish, new ones get a typed `503 draining` with a retry hint.

## 2. First traffic

```bash
curl -s http://HEAD:30000/v1/models | python3 -m json.tool   # lane D direct
curl -s http://HEAD:30000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "deepseek-ai/DeepSeek-V4-Flash-0731",
  "messages": [{"role": "user", "content": "Say OK."}],
  "max_tokens": 16
}'
```

Model IDs are whatever you set in `sparkduet.env` (`DS_SERVED_NAME`,
`QWEN_SERVED_NAME`); the examples below use the defaults.

With the router up, the same calls go to `:30008` and you gain lane pinning:
model `sparkduet-qwen38-27b@fleet` or header `X-SparkDuet-Lane: fleet`.

Watch the router's decisions (one JSON line per request: lane, rule, backend,
status, latency):

```bash
docker logs -f sparkduet-router      # or however you run it; it prints to stdout
curl -s http://HEAD:30008/admin/health | python3 -m json.tool
```

## 3. Pointing coding harnesses at the pair

All four harnesses below are verified daily drivers on a live pair. Each needs
exactly two facts: the base URL and a model ID. Use the router port for lane
routing, or a lane port to hard-wire one lane.

### OpenCode (terminal + web harness)

OpenCode has no built-in generic OpenAI provider; it loads
`@ai-sdk/openai-compatible` on demand. That one `npm` key is the whole
integration. `~/.config/opencode/opencode.json` (mode 600 if you add keys):

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "sparkduet": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "SparkDuet pair",
      "options": { "baseURL": "http://HEAD:30008/v1" },
      "models": {
        "deepseek-ai/DeepSeek-V4-Flash-0731": { "name": "DeepSeek V4 Flash (Lane D)" },
        "sparkduet-qwen38-27b@fleet": { "name": "Qwen3.8-27B (Lane F)" }
      }
    }
  },
  "model": "sparkduet/deepseek-ai/DeepSeek-V4-Flash-0731"
}
```

`opencode` for the interactive TUI; `opencode run "..."` for one-shot scripted
use (right form inside tmux and CI).

### Cursor (Mac/desktop IDE)

Settings → Models → OpenAI API: set the override base URL to
`http://HEAD:30008/v1`, add the served model names verbatim
(`deepseek-ai/DeepSeek-V4-Flash-0731`, `sparkduet-qwen38-27b`). Cursor sends standard
chat-completions; thinking models stream `reasoning_content`, which Cursor
renders as thinking tokens.

### DeepSeek CLI / DeepSeek Harness (`dsh`)

In the web UI: Settings → Models → Add custom provider → protocol
`openai-completions`, base URL `http://HEAD:30008/v1`, then "Fetch available
models" and pick your default. Two operational notes that look like bugs and
are not: (a) `dsh web` rejects API calls whose `Host` header it does not
trust, pass `--trusted-host` for every name you will reach it under;
(b) a fresh install defaults its first chat to DeepSeek's cloud route, which
fails without their key, switch the model chip to your provider once and it
persists.

Headless form for scripts: `dsh --profile headless "task..."`.

### Hermes (unattended agent)

`~/.hermes/config.yaml` (mode 600):

```yaml
model:
  default: deepseek-ai/DeepSeek-V4-Flash-0731
  provider: custom
  base_url: http://HEAD:30008/v1
  api_key: none        # SparkDuet ships no auth; your gateway key if you added one
  context_length: 131072
  max_tokens: 8192
```

Hermes refuses agent mode below 64K context, so `context_length` matters.
Keep `max_tokens` well under the lane's `max_model_len`; Hermes counts it
against the window before sending.

### Anything else

If it speaks the OpenAI API, it works: base URL + model name. `usage` frames
are always present on streams (the lanes run with `stream_options`
compatibility), so token accounting in agent frameworks is accurate.

## 4. Serving + fine-tuning together

The productive daily pattern on a pair: Lane F replica on the head serves
harness traffic while the worker trains (`finetune/README.md`). Before
committing the worker to a training run:

```bash
./scripts/sparkduetctl.sh status     # confirm nothing you need runs there
docker compose --env-file sparkduet.env -f finetune/finetune.compose.yml up -d
docker exec sparkduet-finetune python3 /work/finetune/train-smoke.py
```

For Lane D serving (both nodes), training waits, or runs in gaps: Lane D
tolerates a stopped worker-side trainer, not a live one competing for memory.

## 5. Benchmarks you can publish

```bash
python3 scripts/bench.py --suite standard --lane depth --output results/
python3 scripts/bench.py --suite spec --lane depth        # per-class acceptance
python3 scripts/bench.py --suite fleet --base-url http://HEAD:30008/v1 --lane fleet
```

Artifacts land in `results/` as dated JSON + markdown with `[M-here]` labels
pre-filled. The protocol in `docs/BENCHMARK-PROTOCOL.md` is binding; the
harness enforces the parts a harness can enforce (usage-frame token counting,
TTFT percentiles, acceptance deltas, minimum run length).

## 6. Troubleshooting, fastest-first

| Symptom | First check | Usual cause |
|---|---|---|
| Lane D boots then dies in minutes | `docker logs sparkduet-depth-head` for NCCL timeouts | fabric down or NCCL env not pinned to the QSFP interface; run `nccl-check.sh --full` |
| Decode is single-digit tok/s on Lane D | same | NCCL silently fell back to the management LAN |
| First request after boot takes ~minutes | - | you skipped `warmup.sh`; Triton JIT compiles mid-request |
| `429 lane_saturated` under light load | `curl :30008/admin/health` inflight counts | a crashed backend left inflight counters high; restart the router |
| Harness sees the model but responses garble | test the lane port directly with curl | harness template mismatch; the lane is fine, fix the client template |
| Worker unreachable over SSH, serving fine | management plane (WiFi flap, see FIELD-NOTES §1) | move operator SSH to wired/tailnet |
| OOM at engine boot | `doctor` memory line | KV pool too big for what else runs on the node; lower `*_GPU_MEM_UTIL` |
| Two nodes benchmark differently | `nvidia-smi -q -d CLOCK` both | one node is clock-capped by policy; artifacts record clocks |
| Pair wedges mid-serve, containers alive but no tokens | flight-recorder dumps under `$VLLM_CACHE_DIR/nccl-flight/` on each node | a collective outlived the NCCL watchdog; `torchfrtrace trace_rank_*` names the stalled rank and collective, then restart the lane |

## 7. Updating anything

Images and checkpoints are pinned in `sparkduet.env`. To move a pin: change
the env, `doctor` (it checks digest equality across nodes after you pull),
re-run the lane's benchmark suite, commit the new `results/` artifact next to
the pin change. A pin change without an artifact is a revert waiting to
happen.
