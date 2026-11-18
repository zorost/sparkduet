# Reverse-engineering workbench (RE desk)

A side desk on the SparkDuet pair for inspecting software and model files
the operator is allowed to inspect: our own builds, open source releases, and
files already on disk. It uses [REA](https://github.com/morluto/rea), an MIT
licensed tool published on npm as `rea-agents`. This repo pins
`rea-agents@6.9.0` (see `re/REA_VERSION`). REA is not vendored here.

Status of this document: the commands were run on a Linux x86_64 build VM
against the pinned package (`re/docs/TEST.md`). Nothing has been run on the
Sparks. Every step marked **[unverified on chassis]** still has to be proven
on Spark 1.

## Can this be a full program on the pair?

No. A slice of it can, and that slice is worth having.

REA is a large tool. Much of it depends on engines that do not exist for
Linux arm64, or that upstream has only verified on Linux x64 or macOS. On two
aarch64 GB10 boxes the honest picture is:

| Part of REA | On this pair | Why |
| --- | --- | --- |
| Built-in static readers: GGUF and safetensors headers, CUDA/HIP code objects, ELF notes, binary hardening, Go and Rust build metadata, kernel modules, static .NET, PyInstaller, Deno, and others | **In the program** | Pure Node.js. No engine, no GPU. Expected to run on arm64. [unverified on chassis] |
| Static JavaScript and Electron analysis | **In the program** | Needs Node.js, npm and a procps `ps`. [unverified on chassis] |
| GNU objdump headers and disassembly | **In the program once binutils is present** | REA calls an objdump you supply. [unverified on chassis] |
| Ghidra decompilation | Possible later, not in the first slice | Upstream supports Linux arm64, but the native decompiler may have to be built by hand, and it runs a JVM whose heap competes with the resident lane. |
| Rizin, Kuna, WABT, TShark | Not in the first slice | Operator-supplied tools. Upstream does not state Linux arm64 verification for them. |
| Hopper | **Unavailable** | Official Linux packages are x86_64 only. |
| IDA, JEB | **Unavailable** | Commercial. No installation on the Sparks. |
| JADX (Android decompilation) | **Unavailable** | Upstream requires an x64 host. |
| pwntools ELF layout, recorded Linux crashes, offline EVM | **Unavailable, and not house features** | Upstream is Linux x64 only. The pwntools layout tool refuses Linux arm64 by name. |
| Browser capture, process capture, Frida, debuggers, firmware extraction | Not part of this desk | They run or observe a live target. This desk is static only. |

The full matrix with a citation for each row is in `re/docs/FIT.md`.

So the program is this: a pinned CLI, one folder per investigation, a
read-only host check, and two static first paths that suit this hardware
(model file headers and GPU code objects). It is operational and easy to
follow as a static inspection desk. It is not a general reverse-engineering
lab, and this document does not pretend it is.

## How it stays off the resident lane

- REA here is a CPU process. It does not use CUDA, load weights, or run
  inference. Upstream says so for both first paths: the model reader "does
  not load a model, interpret weights, run inference" and the GPU reader
  "neither executes the artifact nor invokes CUDA/ROCm tools".
- There is no service. Nothing listens on a port. Nothing is started at boot.
  There is nothing to start or stop, so this desk never touches Caddy, the
  Hermes gateway, or `sparkduetctl.sh`.
- No lane swap, no second model. The reasoning model, when an agent is used,
  is whichever agent the operator already runs against the existing OpenAI
  port. REA is a tool server for that agent. It adds no model route and needs
  no API key.
- Page cache is the one real risk. On GB10 the GPU driver allocates from
  `MemFree`, not `MemAvailable` (see `scripts/cache_flusher.sh`). Reading a
  very large file grows `Cached` and takes memory the resident lane may need.
  House rules that follow from this:
  - `re/re.sh model` reads the header only. It never passes `--hash full`.
    Do not run a full-file hash of a weight file while a lane is resident.
  - `re/re.sh gpu` refuses files over 2048 MiB unless you set
    `RE_ALLOW_LARGE=1`, because REA hashes the whole file for that
    inspection.
  - Do not install Ghidra on Spark 1 as part of this desk without deciding
    its JVM heap first.

## What you need on Spark 1

- Node.js 22.x at least 22.23.4, or 24.x at least 24.22.1, or 26.12.1 and
  newer. Node 23 and 25 are not supported. Plus the npm that ships with it.
- A procps `ps` at `/bin/ps` or `/usr/bin/ps`.
- Network access to the npm registry once, for the install.

The check below tells you which of these are missing. If Node.js is missing
or too old, installing it is an operator decision outside this desk.

## Make it available [unverified on chassis]

The product home is `re/` in this repo. On Spark 1 the desk lives in a side
directory, `/home/dspark1/re-desk`, owned by `dspark1`, laid out as
`re-desk/re/...` so every `re/re.sh` command below works from there. It is
never placed in `/srv/ai/sparkduet`, the deployed serving tree.

Two scripts do it, in this order:

1. **Push from the Mac**, from this repo checkout:
   `re/push-from-mac.sh`. It ships the committed `re/` tree over
   `ssh dspark1` with rsync into `/home/dspark1/re-desk/re/` only, leaves
   the box's `re/projects/` alone, then runs step 2 for you.
2. **Deploy on Spark 1**: `re/deploy-spark1.sh` from `/home/dspark1/re-desk`.
   It refuses any host that is not aarch64. It runs `re/re.sh check`,
   `re/re.sh install --apply` (skipped when the pin is already installed),
   `re/re.sh check` again, then `rea setup --client opencode --dry-run` and
   `rea setup --client hermes --dry-run`. Nothing in an agent config changes.
3. **Check** the output: both `check` runs end in `core requirements
   present`, the `rea` row says `matches pin`, and each plan names one config
   file and one skill folder.
4. **Register**: `re/push-from-mac.sh --apply-agents`, or
   `re/deploy-spark1.sh --apply-agents` on the box. It takes timestamped
   backups (`*.re-desk.<UTC time>.bak`) of
   `~/.config/opencode/opencode.json` and `~/.hermes/config.yaml`, runs the
   official `rea setup` for both clients, then shapes only the `rea` rows:
   OpenCode gets `type local`, a list `command`, `enabled true`,
   `timeout 15000`; both get `REA_MCP_OUTPUT_SCHEMA_PROFILE=compact`.
   Skills land where setup puts them: `~/.agents/skills/reverse-engineer-anything`
   for OpenCode and `~/.hermes/skills/reverse-engineer-anything` for Hermes.
5. **Restart OpenCode only**: the script prints the command, or pass
   `--restart-opencode`. It restarts `zorost-opencode.service` only after the
   config parses and holds a valid `mcp.rea` row.

Hermes registration does not start the Hermes messaging gateway, and the
script never starts, stops or restarts any unit whose name contains
`gateway`. It does not restart Hermes or its dashboard. A new `hermes` CLI
session picks up `mcp_servers.rea`; restart the dashboard only if it still
does not show `rea`. cursor-agent on the Sparks stays unregistered.

The resident lane serves every OpenCode and Hermes turn, so the tool list is
trimmed with `REA_MCP_OUTPUT_SCHEMA_PROFILE=compact`. If a normal coding
session still feels heavy, the next knob is `REA_MCP_TOOL_FAMILIES`, set with
`rea setup --client <client> --tool-families <ids>`. The installed 6.9.0
package's README and skill do not list the family ids, so this desk names
none; REA names the valid ids when it rejects a value. Restart the client
after changing it.

## Step 1. Check the host [unverified on chassis]

From `/home/dspark1/re-desk` on Spark 1 (or any checkout of this repo):

```bash
re/re.sh check
```

It prints the architecture, Node.js, npm, `ps`, memory figures from
`/proc/meminfo`, whether REA is installed at the pinned version, and which
optional tools are present. It reads and prints. It installs nothing, edits
nothing, and does not probe the lane. Exit status 0 means the core
requirements are present. Exit status 1 means a `MISSING` row needs fixing.

## Step 2. Install the pinned REA [unverified on chassis]

```bash
re/re.sh install            # prints the plan and stops
re/re.sh install --apply    # runs it
```

The plan is one command:

```bash
npm install --global --prefix "$HOME/.local/share/sparkduet-re" \
  --omit=optional --ignore-scripts rea-agents@6.9.0
```

It writes files under `~/.local/share/sparkduet-re` for the current user. No
sudo. `--omit=optional` skips the Frida binding, which this desk does not
use. `--ignore-scripts` matches how upstream's own updater installs. Then
run `re/re.sh check` again and confirm the `rea` row says `matches pin`.

## Step 3. Make a project

One folder per investigation, created by one command:

```bash
re/re.sh new glm-gguf-header \
  --target /absolute/path/outside/the/repo/model.gguf \
  --question "Which architecture and context length does this file declare?"
```

Rules:

- The slug is lowercase letters, digits and dashes.
- The target is an absolute path **outside this repo**. The command refuses
  a path inside the repo, so a target cannot be committed by accident.
- A slug that starts with `local-` makes a project git never tracks. Use it
  for anything under a restrictive licence.

You get `re/projects/<slug>/` with:

| File | Purpose | Tracked by git |
| --- | --- | --- |
| `QUESTION.md` | The question, the target path, why we may inspect it | yes |
| `EVIDENCE.md` | Findings, unknowns, and a log line per run | yes |
| `STATUS.md` | `Status: open`, `blocked`, `answered` or `dropped` | yes |
| `evidence/` | REA output for this project | no |

`re/re.sh list` prints every project and its status line.

## Step 4. Run the first inspection [unverified on chassis]

Both paths are static and CPU only.

**Model file header** (GGUF v2/v3 or safetensors):

```bash
re/re.sh model glm-gguf-header
```

This runs the upstream command from `docs/model-artifacts.md`:

```bash
rea inspect-model-artifact /absolute/path/model.gguf \
  --detail summary --evidence_output /absolute/path/model-evidence.json --json
```

It reads the metadata and tensor directory only. The result reports format,
version, typed metadata, and each tensor's name, type, shape and offsets.

**GPU code objects** (CUDA fatbin, CUBIN, PTX, HIP bundles inside an ELF):

```bash
re/re.sh new cuda-lib-targets --target /absolute/path/to/library.so
re/re.sh gpu cuda-lib-targets
```

This runs the upstream command from `docs/gpu-code-objects.md` in its
default inventory mode:

```bash
rea inspect-gpu-code-objects /absolute/path/library.so \
  --detail summary --evidence_output /absolute/path/gpu-evidence.json --json
```

It answers which GPU targets a library carries, for example whether a CUDA
library on the Spark ships code for the GB10 target. Kernel-level detail
needs the `--analysis kernels --targets ... --view ...` form quoted in the
upstream doc. Use that form by hand from the doc. This wrapper does not
guess those values.

**Reading the result.** Each run writes two files in the project's
`evidence/` folder and appends a line to `EVIDENCE.md`:

- `<kind>-<time>.summary.json` is the short reply.
- `<kind>-<time>.evidence.json` holds every fact REA retained.

Then write your findings in `EVIDENCE.md` in plain sentences, list what is
still unknown, and set `Status:` in `STATUS.md`.

**Other static commands** that ran against the pinned package in the build
VM and can be run by hand with an absolute path:

```bash
rea inspect-binary-hardening /absolute/path/to/program --json
rea inspect-elf-notes /absolute/path/to/program --json
rea analyze-javascript-application /absolute/path/to/app --json
rea inspect-with-objdump /absolute/path/to/program file_headers
```

On Spark 1, `rea` is `~/.local/share/sparkduet-re/bin/rea`.

## Using it from an agent (optional)

REA is an MCP tool server. The agent calls it, REA returns findings with
evidence, and the agent's own model does the reasoning. No new model route
is added and no cloud key is involved. Do not put API keys in this repo or
in an REA registration.

Preview a registration before applying it. The dry run changes nothing:

```bash
~/.local/share/sparkduet-re/bin/rea setup --client opencode --dry-run
```

| Agent | Where | Guidance |
| --- | --- | --- |
| OpenCode | Spark 1 | The house choice. It already talks to the local OpenAI port, so tool results stay on the pair. Registered by `re/deploy-spark1.sh --apply-agents` (see "Make it available"), then restart OpenCode. [unverified on chassis] |
| Hermes | Spark 1 | Registered by the same script (`mcp_servers.rea` in `~/.hermes/config.yaml`) for the Hermes CLI and dashboard. This desk never starts, stops or restarts the Hermes gateway. [unverified on chassis] |
| Cursor | Operator Mac | `npx rea-agents@6.9.0 setup --client cursor` on the Mac writes `~/.cursor/mcp.json`. That REA runs on the Mac and sees the Mac's files, not the Sparks'. Tool results go to whichever model Cursor is using. |
| cursor-agent | Sparks | Do not register REA there. It uses Cursor cloud models only, so tool results about local files would leave the pair. |

Two caveats. REA advertises a large tool catalog, and whether the resident
lane's model handles it well is untested here. The registration sets
`REA_MCP_OUTPUT_SCHEMA_PROFILE=compact`; `REA_MCP_TOOL_FAMILIES` is the next
knob (upstream `docs/reference/mcp-contracts.md`). The CLI path in steps 1
to 4 needs no agent and no model at all, so start there.

## Update

The pin is deliberate. To move it:

1. Look up the release: `npm view rea-agents dist-tags.latest`.
2. Read the upstream changelog for the versions in between.
3. Put the new version in `re/REA_VERSION` and in the version mentions in
   this folder's docs.
4. `re/re.sh install --apply`, then `re/re.sh check`.
5. If an agent is registered, rerun its `rea setup --client ... --dry-run`
   and apply the reviewed plan, then restart that agent.

Upstream's own `rea update` also works on this install, but it jumps to the
latest release and leaves the pin file stale. Prefer the steps above.

## Remove

1. If an agent was registered, remove REA's registrations and skill files:
   `~/.local/share/sparkduet-re/bin/rea uninstall`. Add `--purge-data` to
   also remove REA's own cache and state. It leaves other MCP servers,
   Node.js and evidence files alone.
   Restart OpenCode afterwards. The `*.re-desk.*.bak` files next to each
   config are the state before each registration run.
2. Delete the install: `rm -rf ~/.local/share/sparkduet-re`.
3. Project folders under `re/projects/` are yours. Keep or delete them. On
   Spark 1 the whole desk is `/home/dspark1/re-desk`.

No service has to be stopped, because none was started.

## What must never be committed

Target binaries, weights, extracted application trees, HAR and SAZ files,
packet captures, core dumps, analysis databases, REA evidence output, `.env`
files, keys and tokens. `re/.gitignore` blocks these under `re/`. Only
`QUESTION.md`, `EVIDENCE.md` and `STATUS.md` are tracked per project. They
are text you wrote, so do not paste secrets or licence-restricted text into
them. Run `git status` before every commit.

## Boundaries

This desk is for lawful inspection of software the operator owns or is
authorised to examine, which is also upstream's stated scope
(`DISCLAIMER.md`). It is not for writing exploits or payloads, capturing
credentials, building malware, or reaching systems without permission. The
pwntools and recorded-crash paths exist upstream, are x64 only, and are not
part of this program.

## Files

| Path | What it is |
| --- | --- |
| `re/re.sh` | The one entry point: `check`, `install`, `new`, `list`, `model`, `gpu` |
| `re/push-from-mac.sh` | Run on the operator Mac: ship `re/` to `/home/dspark1/re-desk/re/` and deploy |
| `re/deploy-spark1.sh` | Run on Spark 1: check, install, check, register OpenCode and Hermes |
| `re/REA_VERSION` | The pinned `rea-agents` version |
| `re/template/` | The three scaffold files copied into each project |
| `re/projects/` | One folder per investigation |
| `re/docs/FIT.md` | What works on the pair and what does not, with upstream citations |
| `re/docs/APPS-CONTRACT.md` | The row to add to the Command Center Apps catalog |
| `re/docs/TEST.md` | What was run in the build VM and what remains for Spark 1 |
