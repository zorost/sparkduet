# Inkling-2-Mini: what a second Spark buys

Review of 26 Nov 2026. Desk review only: nothing on the pair was touched
or measured for this note. The live-lane facts are the operator's.

## Verdict

**A second Spark can only add a second copy today. It cannot raise the
context window.** The live lane stays as it is.

## Live lane (do not replace)

| Fact | Value |
|---|---|
| Engine | vLLM, Spark 1 only, TP=1 |
| Weights | OpenUncensored Inkling-2-Mini, NVFP4 |
| `max_model_len` | 262144 |
| Speed | about 35 tok/s, short reply, 25 Nov 2026 |
| Spark 2 | about 17 GiB free, an image service holds the rest |

## Why a second copy does not raise context

A request lives in one engine. Its KV cache sits in that engine's pool
and cannot spill to another box. Two copies are two separate 262144
windows, not one 524288 window. What a copy adds is concurrency,
aggregate tok/s, and a spare if one node dies. Single-stream speed stays
at about 35 tok/s.

## The only route to a bigger window: TP=2

Tensor parallel splits weights and KV across both nodes, so one request
sees one larger pool. It is closed today for three separate reasons:

1. Published Hyperflare and SGLang builds refuse `tensor-parallel-size`
   above 1 for this NVFP4 checkpoint.
2. vLLM TP=2 on this checkpoint is untested here. Trying it means
   restarting the head as a two-node lane, which replaces the live lane.
3. Spark 2 has about 17 GiB free. A TP rank needs half the weights plus
   its KV share.

Two more ceilings apply even if TP=2 opens up:

- The checkpoint's own trained limit. If its config tops out at 262144,
  no amount of memory raises it. Not checked for this note.
- Spark 1's boot line (`GPU KV cache size ... tokens`). If that pool is
  already larger than 262144, Spark 1 could serve more on its own, but
  that is also a live-lane restart and out of scope.

## Adding a second copy

Not possible right now unless the checkpoint is small: weights plus KV
must fit in Spark 2's free memory, and about 17 GiB will not hold a copy
with a 262144 pool. The weight size was not checked for this note. In
practice the image service has to stop or shrink first.

If it goes ahead:

- Run it on Spark 2 under its own container name and its own port,
  `--restart no`, same image and weights revision as Spark 1.
- Do not use `sparkduetctl.sh start`, `switch`, `restart`, `stop`, or
  `revert` for this. All of them call `stop_all`, and `switch` drains
  and refreshes the house pickers.
- Before starting, save the live spec:
  `./scripts/sparkduetctl.sh capture-incumbent <live-container>` on
  Spark 1. The file is a `docker inspect` dump with env vars. It lands in
  `results/incumbent-<name>-<date>.json`, which `.gitignore` does not
  cover (only `results/incumbent-*/` directories are ignored). Do not
  commit it.
- Clients keep pointing at Spark 1. Add Spark 2 to a gateway only after
  `/v1/models` and a short completion pass on it.

## Rollback

The trial is additive, so rollback never touches Spark 1.

1. Remove the Spark 2 backend from any gateway or picker that lists it.
2. On Spark 2: `docker stop <trial-container> && docker rm <trial-container>`.
3. Start the image service again if it was stopped, and confirm it
   answers.
4. On Spark 1: `GET /v1/models` still shows `max_model_len` 262144, and
   a short completion still runs at about 35 tok/s.

If step 4 fails, the live lane was disturbed by something outside this
trial. Restore it from the captured spec with `docker start`, not with
`sparkduetctl.sh revert`.

## Open items before any TP=2 attempt

- Weight size on disk and the checkpoint's configured context limit.
- Spark 1 boot-line KV pool in tokens.
- A vLLM build that accepts TP=2 for this NVFP4 checkpoint, proven on a
  maintenance window with the capture above in hand.
