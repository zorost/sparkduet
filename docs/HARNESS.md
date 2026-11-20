# HARNESS.md, how clients use the resident

One page. It covers how OpenCode, Hermes and Open WebUI should use the
resident GLM-5.3-Flash EXL3 lane so a build job ends in minutes, not hours.
Written 2026-11-18 from one measured session on Spark 1 and the sources at the
end. Nothing here was applied or restarted on the pair from this repo: every
chassis step is **[unverified on chassis]** until someone runs it there.

## The stack on the Lab pair (2026-11-18)

| Port (loopback) | What | Note |
|---|---|---|
| `:30000` | Engine | `GET /v1/models` reported `brandonmusic/GLM-5.3-Flash-EXL3-TR3-4bpw`, `max_model_len` 1000000 [M-here]. The tracked recipe door is `G_EXL3_MAX_MODEL_LEN=262144`. Always read the live value. |
| `:20127` | Clamp | If a request leaves the thinking fields out and reserves more than 2048 output tokens, it turns thinking on at effort high. A client that sends an effort is expected to keep it. [unverified on chassis] |
| `:20128` | 9Router | Lists whatever `:30000` serves as `spark1/<served id>`. |
| `:18787` | Headroom | Where OpenCode's `baseURL` points. |

Only the engine is part of this repo. The other three are house services.

## Rules for every client

1. **One model at a time.** Titles, summaries, subagents, the Magic Context
   historian and Hermes compression all land on the same engine. Point no
   helper at a second model on the pair. One long agent run at a time; a chat
   sent during it starts late, because the lane does not mix a peer prefill
   into a running decode step [P].
2. **Advertise 85 percent of the live window, never more than the engine
   reports.** `max_model_len * 85 // 100`, which is 850000 today. Output cap
   16384. Do not shrink a client's output reservation to 2048 or less.
3. **No shell calls to `:30000`, `:20127` or `:20128`.** The agent would be
   polling the model that is running it.
4. **No sleep-polling.** Run the command in the foreground with a timeout.
5. **Effort is the client's choice and stays fixed inside a session.** Build
   low, Plan and Hard high. In one public recipe for this model family,
   changing effort mid-conversation was a full prefix-cache miss [M-else].

## What the slow Build session showed [M-here]

| Measure | Value |
|---|---|
| Wall clock | 273.5 min, 205 steps, 204 ending in another tool call |
| Context per step | median 98052 tokens |
| Bash time | 201 min, 73 percent of the wall clock |
| `sleep` | 51 commands, 10091 s: 168 min, 61 percent of the wall clock |
| Self-polling | 20 `curl` calls to `:30000`, `:20128`, `:20127` |

The model was not the slow part. Waiting was.

## OpenCode

Live file `~/.config/opencode/opencode.json`, rules
`~/.config/opencode/AGENTS.md`. Template and notes: `configs/opencode/`.
Field names below are from the `v1.19.6` docs and source.

| Setting | Value | Why |
|---|---|---|
| `agent.build.steps` | 60 | Unset means no limit. At the limit OpenCode tells the model to stop calling tools and summarize; the user continues with one message. |
| `agent.build.reasoningEffort` | `low` | Unknown agent keys pass to the provider as model options. Thinking tokens stay in the transcript and are sent again every step: 30 OpenCode runs on another local model took a median 179.6 s with thinking on and 106.0 s with it off [M-else, source 6]. Low keeps thinking on, which is safer than off for long structured output [M-else]. |
| `agent.plan`, `agent.hard` | `high` | Few steps, hard decisions. |
| `agent.build.permission.bash` | `*` allow, then deny `sleep *`, `*:30000*`, `*:20127*`, `*:20128*` | Last matching rule wins, so `*` goes first. Each parsed sub-command is checked, including pipes, `&&` and loops. Not caught: `timeout 30 sleep 5`, `bash -c "sleep 30"`. The rules file covers those. |
| `agent.build.tools` | `rea_*` false | Keeps the 255-tool REA catalog off every Build request. A longer tool list is prefill on every step: on another harness, one bug hunt went from 319 s to 509 s when the tool list grew from two to six [M-else, source 5]. The underscore matters; a bare `rea*` also hides the built-in `read`. |
| `compaction` | `auto` false, `prune` false | Magic Context is the only compactor. Its docs require native compaction off, and it stays off while a conflict is unresolved. Do not turn native compaction on. |
| `limit` | context 850000, output 16384 | Rule 2. |

Magic Context: `execute_threshold_tokens.default` is 36000, applied as
`min(36000, max(12000, served * 55 // 100))`. Do not raise the cap. A token
entry wins over `execute_threshold_percentage`, and the percentage is used
when no context limit resolves, so the 35 percent key must never be the only
threshold: on an 850000 window it would wait until 297500 tokens.

## Hermes

This repo ships no Hermes config, so these are limits, not a second product.
`~/.hermes/config.yaml`, `model.provider: custom`.

- `model.context_length` at or under 85 percent of the live window: 850000 or
  less today. Hermes refuses agent use under 64000.
- Compression starts at `compression.threshold` times `context_length`
  (default 0.50). At 850000 that is 425000 tokens. A lower
  `context_length`, or `compression.threshold_tokens`, keeps Hermes steps
  near the OpenCode budget. [P, not measured here]
- Give unattended runs a turn limit (`agent.max_turns` or `--max-turns`);
  current docs give the default as unlimited or 500 depending on the page.
- The same no-shell-polling and no-sleep rules apply.
- **Do not start the messaging gateway.** It is a separate command
  (`hermes gateway`) and nothing in this repo starts, stops or restarts it.

## Open WebUI

Chat, not a 200-step agent. This repo ships no WebUI config.

- Native function calling loops up to `CHAT_RESPONSE_MAX_TOOL_CALL_ITERATIONS`
  rounds per reply, default 1024. Set it low. [P, not measured here]
- Title, tag and follow-up generation are extra calls to the same resident.
  Turn off the ones nobody reads. Do not set a second task model on the pair.
- Per-model `max_tokens` and `reasoning_effort` are forwarded. A chat that
  sends no effort gets what the clamp decides: thinking at effort high once
  more than 2048 output tokens are reserved.
- Legacy function calling rewrites the prompt each turn and breaks the prefix
  cache. Use Native.

## If Lane G runs on TensorFold

`G_ENGINE=tensorfold` is opt-in (`docs/MODELS.md`). The rules above hold.
Three things change for clients:

- Prompt plus `max_tokens` past the window is HTTP 400, not a clamp. Keep
  the 85 percent context and 16384 output limits. Never send `9999999`.
- Effort maps to the template's `low`, `high` and `max`: Hermes' default
  `medium` arrives as `high`. Any effort turns thinking on.
- Each conversation keeps its own prompt state. Keep one session's effort
  fixed, and do not interleave many short chats with a long Build run: kept
  states share the 6 GiB prompt cache, least recently used out first.

## Reviewed and not adopted

Public GB10 recipes for this checkpoint family were read on 2026-11-18.
Their native-MTP settings agree with the tracked lane, so the serving recipe
is unchanged: see `docs/MODELS.md`. `chat_template_kwargs.clear_thinking`
for multi-turn agents is reported to shorten prompts [M-else]; it waits for a
measurement on this pair.

On 2026-11-20 a public two-GB10 TensorFold recipe announced faster decode,
fewer patches and a smarter prompt cache. Its headline speeds use an external
drafter whose license is non-commercial, 8-bit dense and head weights and its
own engine patches; none of those are adopted. What it credits to upstream TensorFold is adopted as the
opt-in `G_ENGINE=tensorfold` engine above.

## Still to verify on Spark 1

- The live Build glob is `rea_*`, not `rea*`, and `read` is in Build's tools.
- `"*": "allow"` is the first key of the live `permission.bash`.
- A Build request reaches the engine with effort low (clamp log).
- OpenCode was restarted after the edit and a Build session ends inside 60
  steps with no `sleep` and no call to the three ports.

## Sources opened (2026-11-18)

`opencode.ai`, the NVIDIA forums and Hugging Face were not reachable from the
build VM; GitHub copies were read instead.

1. OpenCode `v1.19.6`, github.com/anomalyco/opencode: docs `agents.mdx`,
   `permissions.mdx`, `config.mdx`, `mcp-servers.mdx`, `rules.mdx`; source
   `core/src/v1/config/agent.ts`, `core/src/util/wildcard.ts`,
   `opencode/src/permission/index.ts`, `opencode/src/tool/shell.ts`,
   `opencode/src/session/llm/request.ts`,
   `core/src/session/runner/max-steps.ts`.
2. Magic Context, github.com/cortexkit/magic-context: `CONFIGURATION.md`,
   `README.md`, release notes for `v0.21.7`.
3. Headroom, github.com/chopratejas/headroom: `README.md`.
4. Hermes Agent, github.com/NousResearch/hermes-agent: `providers.md`,
   `configuration.md`, `context-compression-and-caching.md`,
   `cli-commands.md`, `cli-config.yaml.example`. Open WebUI,
   github.com/open-webui: docs `server-side-tool-calling.md`; source
   `env.py`, `utils/payload.py`.
5. github.com/DG1001/local-agentic-coding-128gb: `README.md`,
   `docs/harness.md`, `docs/operations.md`, `docs/speed.md`.
6. github.com/evanwtf/local-llm issues 329, 330, 333, 334.
7. Six public GB10 serving recipes for GLM-5.3-Flash EXL3, read for serving
   flags and client advice. This repo does not name other recipe authors.
8. TensorFold at `81de26f6`, github.com/ashhart/TensorFold, read on
   2026-11-20: `README.md`, `RUNBOOK.md`, `CHANGELOG.md`, `build.zig.zon`,
   `docs/run-guide.md`, `docs/api.md`, `docs/supported-models.md`,
   `docs/recipes/glm-5.3-flash.md`, `docs/recipes/glm-5.3-flash-cuda.md`; and
   the `v1.3.0` `linux-aarch64` release archive and its `RUNTIME.md`.
