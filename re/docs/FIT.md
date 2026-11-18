# Fit note: REA on two DGX Spark GB10

This is the research behind `re/README.md`. It answers one question: which
parts of REA are real on a pair of Linux aarch64 GB10 boxes.

## Sources

- npm package `rea-agents`, version **6.9.0**, licence MIT, looked up on
  2026-11-18 with `npm view rea-agents`. `dist-tags.latest` was 6.9.0.
  `engines.node` is `^22.23.4 || ^24.22.1 || >=26.12.1`. The package
  installs two commands, `rea` and `rea-agents`.
- Upstream repository <https://github.com/morluto/rea>, branch `main` at
  commit `82e16b4dda6a5e526845a26ad09251f8fa69807d` (2026-11-18). Paths
  below are relative to that repository.
- The site <https://rea.tools/> could not be reached from the build VM, so
  no claim here is cited to it.

Two cautions about these sources:

1. Upstream `main` documents code that may be newer than the npm release
   (`docs/getting-started/installation.md`, "Released package and main").
   One example found while testing: `rea inspect-elf-dependencies` is
   documented on `main` but is not a command in 6.9.0. Every command this
   desk wraps or recommends was run against the 6.9.0 package itself.
2. Those runs were on Linux x86_64. Nothing below has been run on a GB10.

## Install, update, remove

| Claim | Source |
| --- | --- |
| Setup is `npx rea-agents setup`. A regular CLI install is `npm install --global rea-agents`, then `rea --help`. | `README.md`, "Quick start" |
| Node.js 22.x (>=22.23.4), 24.x (>=24.22.1), or 26.12.1+, plus npm. Node 23, 25, earlier 26.x and prereleases are unsupported. | `README.md`, "What you can analyze"; `docs/getting-started/installation.md`, "Package runners and global installation" |
| On Linux a procps-compatible `ps` is required for Ghidra, JADX, firmware tools and static JavaScript/Electron analysis. REA tries `/bin/ps`, then `/usr/bin/ps`, before its `PATH`. | `README.md`, "Quick start"; `docs/getting-started/installation.md`, "Linux owned-process prerequisite" |
| `rea doctor --json` is read-only. Without options it audits everything, and `healthy` can be false while the chosen workflow works. | `docs/getting-started/installation.md`, "Check readiness for your task" |
| `rea update` updates the npm global installation that owns the running CLI, with dependency lifecycle scripts disabled. | `docs/getting-started/installation.md`, "Update the owning installation" |
| `rea uninstall` removes REA-owned agent registrations and skill files. `--purge-data` also removes REA cache and state. It preserves Node.js, evidence files, captures, and other MCP servers. | `docs/getting-started/installation.md`, "Diagnose, update, and remove" |
| A pinned package request is supported, for example `npm exec --yes --package=rea-agents@<version> -- rea setup`. | `docs/getting-started/installation.md`, "Package runners and global installation" |
| Licence is MIT for REA-authored code. Commercial tools (IDA, Hopper, JEB) need their own licences. | `LICENSE`; `README.md`, "License" |

## What is built in and what needs an engine

From the table in `README.md`, "What you can analyze".

**Built in, static, no external engine:** desktop installation inventory, Go
binaries, GPU code objects, model artifacts, IL2CPP metadata, Godot PCK, Rust
binaries, PE certificates, PE imports, Windows shortcuts, ELF notes, binary
hardening, Apple IPS crash reports, Linux kernel modules, Erlang BEAM, .NET
assemblies, APK signatures, Flutter apps, PyInstaller executables, Deno
compile binaries. JavaScript and Electron analysis needs only Node.js, npm
and `ps`.

**Needs something you supply:**

| Capability | Requirement stated upstream |
| --- | --- |
| Native pseudocode, assembly, xrefs | Hopper, Ghidra, IDA or Kuna |
| Offline ELF layout | pwntools, Linux x64 |
| Recorded Linux crashes | pwntools, Linux x64, optional GDB/pwndbg |
| EVM bytecode | Linux x64 |
| Android APK decompilation | Headless JADX and a full JDK, x64 |
| WASM artifacts | WABT 1.0.42 |
| Recorded packets, packet summaries | TShark 4.x, capinfos, prlimit, Linux |
| Firmware | Binwalk, Unblob, Linux |
| Websites | A Chrome-family browser |
| Saved network captures | HAR or SAZ file; mitmdump on Linux for native mitmproxy |
| SQLite databases | Node.js 24.x >=24.15 or 26+ |
| Process behaviour | Linux or macOS with a native PTY |
| JEB projects | A caller-started JEB client |
| objdump, Rizin | Caller-supplied; "REA does not install or update either provider" |

## What upstream says about architecture

| Item | Upstream statement | Source | Verdict for the pair |
| --- | --- | --- | --- |
| pwntools ELF layout | "Real verification covers Linux x64 hosts". "Other hosts, including Linux arm64, return `capability_unavailable` naming the detected host before any target is read." | `docs/guides/native/binary-diagnostics.md` | Unavailable. Confirmed x64 only. |
| Recorded Linux crashes | "Caller-supplied pwntools on Linux x64". Verification with "Linux x64, GDB 13.1". | `README.md`; `docs/guides/native/recorded-crashes.md` | Unavailable. Confirmed x64 only. |
| Offline EVM | "Local raw/hex carrier on Linux x64". "Initial real verification covers flat legacy bytecode on Linux x64". | `README.md`; `docs/guides/evm/evm-bytecode.md` | Unavailable. Confirmed x64 only. |
| JADX | "Headless JADX and a full JDK on Linux/macOS/Windows x64". | `README.md` | Unavailable. Confirmed x64 only. |
| Hopper | "The official Linux packages are x86_64-only, so setup does not offer Hopper on other Linux architectures; configure Ghidra there instead." | `docs/getting-started/installation.md`, "Hopper" | Unavailable. |
| IDA | Verified headless on Windows x64 and macOS arm64. No Linux arm64 statement. Commercial licence required. | `docs/providers/ida-provider.md` | Unavailable in practice. |
| Ghidra | "REA connects to an existing Ghidra installation on Linux x64/arm64". Needs Ghidra 12.x (>=12.1) and a matching 64-bit full JDK (21 or newer for 12.1). "Linux ARM64 uses `linux_arm_64`; official release archives may require you to build that native component separately." "REA does not build or install native tools." | `docs/getting-started/installation.md`, "Ghidra" | Realistic but not turnkey. Not in the first slice. |
| GNU objdump | Caller-supplied. REA passes GNU-style arguments and does not verify the implementation or version. | `docs/reverse-engineering-tools.md`, "GNU objdump" | Realistic. No architecture limit stated. Ubuntu binutils exists for arm64. |
| Rizin | Caller-supplied, selected with `REA_RIZIN_COMMAND`. No Linux arm64 verification stated. | `docs/reverse-engineering-tools.md` | Unknown. Not in the first slice. |
| Kuna | Targets include AArch64 ELF. "The 1.990 real-provider checks ran on macOS arm64; Linux and other hosts have not been verified with that release." | `docs/kuna-provider.md` | Unknown on Linux. Not in the first slice. |
| WABT | "The real-tool profile has been exercised on macOS arm64; other hosts require their own verification." | `docs/guides/wasm/wasm-artifacts.md` | Unknown. Not in the first slice. |
| Firmware | "This first firmware milestone is verified on Linux x64." | `docs/guides/firmware/firmware-analysis.md` | Not verified on arm64. Not in the first slice. |
| Model artifacts | Built-in header reader. "REA does not load a model, interpret weights, run inference, import Python, open external-data references or use the network." | `docs/model-artifacts.md` | In the program. |
| GPU code objects | Built-in reader for ELF/PE sections, CUDA fatbins, CUBIN, PTX, HIP bundles. "It neither executes the artifact nor invokes CUDA/ROCm tools." | `docs/gpu-code-objects.md` | In the program. |

Upstream does not publish a statement that the built-in static readers were
verified on Linux arm64. They are JavaScript with no native engine, and the
6.9.0 tarball ships no Linux native binary of its own (the only one under
`native/` is for Windows x64), so there is no known reason for them to fail
on aarch64. That is an inference, not a measurement. The first run of
`re/re.sh model` on Spark 1 is what settles it.

## Resource facts that matter on unified memory

| Fact | Source |
| --- | --- |
| Model inspection default `hash: "header"` reads only the metadata and tensor directory. `hash: "full"` reads the whole file, and cost grows with file size. | `docs/model-artifacts.md`, "Identity and reads" |
| GPU inspection always streams the full file for SHA-256. Limits are 128 MiB per buffer and 512 MiB of payload work per inspection. | `docs/gpu-code-objects.md` |
| On GB10 the GPU driver allocates from `MemFree`, not `MemAvailable`, so page cache growth can starve a loaded lane. | This repo, `scripts/cache_flusher.sh` |

## How an agent calls REA

| Claim | Source |
| --- | --- |
| "Your agent calls REA through MCP to inspect the target and trace relevant code." CLI commands use the same workflows. | `README.md`, "How REA works" |
| "REA analyzes targets locally. Your agent receives the tool results, and its model provider has its own data policy." | `README.md`, FAQ |
| Supported setup clients include `opencode`, `hermes` and `cursor`. | `docs/getting-started/installation.md`, "Supported agents" |
| OpenCode: setup writes the `mcp.rea` entry and honours `OPENCODE_CONFIG_DIR` and `OPENCODE_CONFIG`. | `docs/getting-started/client-configuration.md`, "Other client settings" |
| Hermes: setup writes `mcp_servers.rea` to `config.yaml` under `HERMES_HOME` or `~/.hermes`, with `enabled: true`. | `docs/getting-started/client-configuration.md`, "Hermes" |
| Cursor: setup writes `mcpServers.rea` into the user-level `~/.cursor/mcp.json`. | `docs/getting-started/client-configuration.md`, "Cursor" |
| Manual registration is a stdio server: `node /absolute/path/to/rea/scripts/rea.mjs mcp`. | `docs/getting-started/installation.md`, "MCP Registry" |
| `REA_MCP_TOOL_FAMILIES` restricts the advertised tool families. `REA_MCP_OUTPUT_SCHEMA_PROFILE=compact` or `off` shrinks the catalog. | `docs/reference/mcp-contracts.md`; `README.md`, "Set up your agent" |

REA has no model of its own and takes no model key. The reasoning model is
the one behind the agent the operator already uses.

## Upstream capabilities that are not part of this program

Mentioned so nobody goes looking for them here: pwntools layout and recorded
crash inspection (x64 only, and exploit-adjacent), GDB and Rizin debugger
sessions, Frida instrumentation, process capture, browser and network
capture, ADB device access, firmware extraction, and the RL environment
builder. This desk is static inspection of files the operator may inspect.
