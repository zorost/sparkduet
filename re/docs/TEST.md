# Test record

## Where these ran

A Linux **x86_64** build VM on 2026-11-18. Node.js v22.23.5, npm 10.9.9,
procps-ng 4.0.4, `rea-agents@6.9.0` installed with the exact command that
`re/re.sh install` prints, into a scratch prefix outside the repo.

**None of this ran on a DGX Spark.** A pass here proves the scripts and the
pinned package behave as documented on x86_64. It does not prove anything
about aarch64 GB10. The Spark 1 list is at the end.

## Results in the build VM

| # | Command | Exit | Result |
| --- | --- | --- | --- |
| 1 | `bash -n re/re.sh` | 0 | Shell syntax is valid. |
| 2 | `re/re.sh check` | 0 | Reported `WARN arch x86_64 (not a GB10 ...)`, `OK node v22.23.5`, `OK npm 10.9.9`, `OK ps /bin/ps (ps from procps-ng 4.0.4)`, memory rows, `OK rea ... (6.9.0, matches pin)`, objdump present, Ghidra, Rizin, Kuna, WABT and TShark absent. Ended with `RESULT: core requirements present.` |
| 3 | `re/re.sh check` with a `PATH` holding only coreutils (no node, npm, objdump, java) and no REA | 1 | Did not crash. Reported `MISSING node`, `MISSING npm`, `ABSENT rea`, every optional provider `ABSENT`. Ended with `RESULT: core requirements missing.` |
| 4 | `re/re.sh install` (dry run) | 0 | Printed the plan and stopped. Installed nothing. |
| 5 | `npm install --global --prefix <scratch> --omit=optional --ignore-scripts rea-agents@6.9.0` | 0 | `rea --version` printed `6.9.0`. |
| 6 | `re/re.sh new sample-gguf-header --target <scratch>/tiny.gguf --question "..."` | 0 | Created `QUESTION.md`, `EVIDENCE.md`, `STATUS.md` and `evidence/`. |
| 7 | `re/re.sh new bad/slug --target /x` | 2 | Refused the slug. |
| 8 | `re/re.sh new inside --target <repo>/README.md` | 2 | Refused: target is inside the repo. |
| 9 | `re/re.sh new rel --target foo` | 2 | Refused: target must be absolute. |
| 10 | `re/re.sh model sample-gguf-header` | 0 | Real inspection, details below. |
| 11 | `re/re.sh gpu local-gpu-ls` with target `/bin/ls` | 0 | Real inspection. `/bin/ls` holds no GPU code, so the inventory was empty (0 of 0 examined). This proves the command and flags, not a GPU finding. |
| 12 | `re/re.sh list` | 0 | Listed both projects with `open`. |
| 13 | `git check-ignore` on `evidence/x.json`, a `local-*` project, `foo.har`, `x.pcap` under `re/` | 0 | All four ignored. `git status` showed only scaffold and docs. |
| 14 | `rea setup --client opencode --dry-run`, and the same for `hermes` and `cursor`, with an empty scratch `HOME` | 0 | Each printed `status: planned` with one config file and one skill folder as targets. Nothing was written. |
| 15 | `rea doctor --json` | 1 | Printed a full report. Exit 1 is expected when no analysis engine is configured, which upstream documents. It wrote nothing to the scratch `HOME`. |

### Test 10 in detail

The fixture was a 176-byte GGUF v3 file generated in the VM: two metadata
keys and one 4-element F32 tensor. It is synthetic, lives outside the repo,
and is not committed.

Command the wrapper ran:

```bash
rea inspect-model-artifact <scratch>/tiny.gguf \
  --detail summary \
  --evidence_output re/projects/sample-gguf-header/evidence/model-<time>.evidence.json \
  --json
```

Exit code 0. Facts in the evidence file:

- format `gguf`, version 3, byte order little
- read scope `header`, 146 bytes read of a 176-byte file
- metadata `general.architecture = llama`, `general.name = tiny-fixture`
- one tensor `w`, type F32, shape `[4]`, 16 bytes

Those match what the fixture was built with. The summary also carried REA's
own limitation text: facts are file declarations, and the full-file SHA-256
is unknown because tensor data was not read.

The sample project was deleted after the test, so `re/projects/` ships
empty.

### Other upstream commands run by hand against 6.9.0

| Command | Exit | Note |
| --- | --- | --- |
| `rea inspect-model-artifact <scratch>/tiny.gguf --json` | 0 | Full evidence on stdout. |
| `rea inspect-model-artifact <scratch>/tiny.safetensors --hash full --detail summary --json` | 1 | Refused by design: summary delivery requires `--evidence_output`. |
| `rea inspect-binary-hardening /bin/ls --json` | 0 | |
| `rea inspect-elf-notes /bin/ls --json` | 0 | |
| `rea inspect-gpu-code-objects /bin/ls --json` | 0 | |
| `rea analyze-javascript-application <scratch>/js --json` | 0 | Three-file fixture. |
| `rea inspect-with-objdump /bin/ls file_headers` | 0 | Used the VM's GNU objdump. |
| `rea inspect-elf-dependencies /bin/ls --json` | 1 | `COMMAND_NOT_FOUND`. Documented on upstream `main`, not in 6.9.0. Not used by this desk. |
| Second run with the same `--evidence_output` path | 1 | Refused: evidence file already exists. The wrapper uses a timestamp per run for this reason. |

## Could not run in this VM

| What | Why | Command still to run on Spark 1 |
| --- | --- | --- |
| Host check on GB10 | The VM is x86_64 and cannot reach the Sparks. | `re/re.sh check` |
| Install on aarch64 | Same. | `re/re.sh install --apply` then `re/re.sh check` |
| Header inspection of a real model file | No weights in the VM, and no GB10. | `re/re.sh new <slug> --target /absolute/path/model.gguf` then `re/re.sh model <slug>` |
| GPU code-object inventory of a real CUDA library | The VM has no CUDA libraries. | `re/re.sh new <slug> --target /absolute/path/library.so` then `re/re.sh gpu <slug>` |
| Effect on the resident lane | No lane exists in the VM. | Note `MemFree` and `Cached` from `re/re.sh check` before and after the two inspections above, with the lane loaded, and confirm the lane still answers. |
| OpenCode registration against the local port | No OpenCode and no lane in the VM. | `~/.local/share/sparkduet-re/bin/rea setup --client opencode --dry-run`, review, apply, restart OpenCode, ask it to call `binary_session`. |
| Command Center Apps row | `command-center/index.html` is not in this repo. | Follow `re/docs/APPS-CONTRACT.md` after the first two rows above pass. |

## Spark 1 results

Not run yet. When they are, add a dated table here with the command, the
exit code and the result, the same way as above.
