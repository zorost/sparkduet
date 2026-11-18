# OpenCode on the resident lane

Sanitized copies of the house OpenCode setup for the pair. Why each value is
what it is: [`docs/HARNESS.md`](../../docs/HARNESS.md).

| File here | Live file on the box | Notes |
|---|---|---|
| `opencode.json.example` | `~/.config/opencode/opencode.json` | Mode 600. Global config. |
| `AGENTS.md.example` | `~/.config/opencode/AGENTS.md` | Global rules, loaded into every session. |
| `magic-context.jsonc.example` | `~/.config/cortexkit/magic-context.jsonc` | Older Magic Context builds read `~/.config/opencode/magic-context.jsonc` and move it on upgrade. |

Checked against OpenCode `v1.19.6` docs and source. Nothing here was run on
the chassis from this repo. [unverified on chassis]

## Keys stay on the machine

The example carries no API key. `options.apiKey` is `{env:SPARK_GATEWAY_KEY}`,
which OpenCode reads from the environment of the process that starts it. A
key written inline belongs only in the live mode 600 file, never in git, an
issue, or a paste.

## What is in the template

- **Model row.** `spark/spark1/<served id>` through Headroom on
  `127.0.0.1:18787`. `limit.context` is 85 percent of what the engine
  reported on 2026-11-18 (`max_model_len` 1000000, so 850000) and
  `limit.output` is 16384. The house picker refresh rewrites this row after a
  lane swap. On another pair, read `GET :30000/v1/models` and set
  `context = max_model_len * 85 // 100`. Never advertise more than the engine
  reports.
- **`compaction`.** `auto` and `prune` are both false on purpose. Magic
  Context is the only compactor. Do not turn native compaction on, and do not
  drop the `plugin` row while compaction is off.
- **`agent.build`.** 60 steps, `reasoningEffort` low, bash allowed except
  `sleep` and the three model-stack ports, REA tools off.
- **`agent.plan`, `agent.hard`.** `reasoningEffort` high.

## Two things that are easy to break

1. **Key order in `permission.bash`.** OpenCode applies the last matching
   rule, so `"*": "allow"` must be the first key and the denies come after it.
   A catch-all placed last re-allows everything.
2. **The REA glob needs its underscore.** MCP tools are named
   `<server>_<tool>`, so REA tools are `rea_...` and the glob is `rea_*`. A
   bare `rea*` also matches the built-in `read` tool and hides it from Build.
   `scripts/test_harness_config.py` checks both.

`tools` is the deprecated spelling in the 1.19 line and still works. The
current spelling of the same rule is `"permission": { "rea_*": "deny" }`.

## What is not in the template

The `mcp` block. `context7`, `playwright-app` and `rea` are registered on the
box by their own installers (`re/deploy-spark1.sh --apply-agents` for REA) and
stay enabled, so a session that needs REA can use Plan or Hard, or a project
config that turns it back on. The Build globs refer to those server names.

## Apply by hand

Merge, do not overwrite: the live file also holds the `mcp` block and the
key. Back up, edit, check that it parses, then restart OpenCode.

```bash
cp -p ~/.config/opencode/opencode.json ~/.config/opencode/opencode.json.$(date -u +%Y%m%dT%H%M%SZ).bak
python3 -m json.tool ~/.config/opencode/opencode.json >/dev/null && echo parses
```
