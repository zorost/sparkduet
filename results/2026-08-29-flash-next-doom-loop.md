# Flash-Next leftover `!` loop after sglang#36845, 2026-08-29

## Verdict

The house already had the kernel half (sglang#36806 + #36845). That closed
the SM121 TRT-LLM silent token-0 path. It did not close the leftover
long-thinking loop: a 1,600-token thinking decode can still emit `!`, stay
HTTP 200, insert into radix, and poison the next request until both ranks
restart.

The leftover abort landed in `patches/next-sglang-sm121/`. The radix-insert
anchor in `lmsysorg/sglang:qwen38flashnext` is indented 16 spaces. The house
apply script matches this image. Both Sparks rebuilt
`qwen38-flashnext-dspark:local`. DeepSeek stayed resident. The new image is
used on the next `switch next`.

## What landed in the house image

1. `sm121_varlen.py` zeros non-finite QSA attention (empty selected-KV or a
   0 softmax running sum).
2. `apply_nvfp4_patches.py` installs `patch_token0_guard`: abort after 16
   consecutive token-id-0 samples, skip radix insert, reset the prefix cache
   before the next prefill. Build fails if the SGLang anchors moved.
3. Compose sets `SGLANG_SANITIZE_NAN_LOGITS=1` and pins
   `--tool-call-parser qwen3_coder`.
4. Server thinking stays off by default. Clients can still turn it on; the
   guard is for that path.

## What we did not do

- Did not swap off DeepSeek to run `scripts/doom_loop_repro.py` live.
- Did not take a YaRN 1M default, mem 0.82, or a second start.sh.
- Did not take a TRT-LLM kernel on SM121.

Live check after the next `switch next`:

```bash
python3 scripts/doom_loop_repro.py --base-url http://127.0.0.1:30000/v1
```
