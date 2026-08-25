# Container-start hotfixes (Lane D)

Applied by `entry.sh` before `vllm serve` starts, on both ranks. Both are
idempotent and gated to the pinned engine image's exact source; on a new
image they no-op with a log line instead of breaking the boot.

| File | What it fixes | Origin |
|---|---|---|
| `hotfix-dsv4-issue55-tool-truncation.py` | A tool call truncated by `max_tokens` used to report `finish_reason: "tool_calls"` with unparseable JSON `arguments`, poisoning agent transcripts (HTTP 400 on replay). Truncated calls now report `"length"` and invalid arguments are dropped. | [MiaAI-Lab's recipe](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark), issue #55, MIT. Carried verbatim with the license notice. |
| `hotfix-dsv4-suppress-stops-in-reasoning.py` | Client `stop` strings could fire inside the reasoning block, ending a request mid-think with `content: null`. Stops now stay dormant until reasoning closes. | Same recipe, porting tonyd2wild's Stage-C patch 5 to the Anemll image path. MIT, carried verbatim. |

Status check inside a running head container:

```bash
docker exec sparkduet-depth-head \
  python3 /sparkduet-patches/hotfix-dsv4-issue55-tool-truncation.py --status
```

Opt-outs are the originals' env switches (`DSPARK_SUPPRESS_STOPS_IN_REASONING=0`
to let stops fire inside reasoning). Full credit in `../CREDITS.md`.
