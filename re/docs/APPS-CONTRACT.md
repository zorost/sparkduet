# Command Center Apps contract: RE desk

The Apps tab on the operator Mac reads its catalog from the `HOUSE_APPS`
array in `command-center/index.html`. That file is not in this repository.
**The live Apps tab has not been edited.** This document is the contract for
whoever adds the row later.

## Proposed row

| Field | Value |
| --- | --- |
| id | `redesk` |
| label | `RE Desk` |
| one-line job | `Static inspection of model files, GPU code objects and binaries with REA. CPU only.` |
| host | Spark 1 (gateway, 192.168.1.220) |
| kind | Catalog entry plus a local project folder. Not a service. |
| Open | The operator doc, `re/README.md` in this repository |
| GPU rule | Leave the resident lane loaded. No swap, no second model. |
| Start / Stop | None. Do not add them. |
| Port | None |
| Health check | None that the tab can poll. Readiness is `re/re.sh check` run by hand on Spark 1. |

The existing rows are `nava`, `jev`, `laya`, `voicestudio` and `dossier`.
Their exact field names are not visible from this repository. Use whatever
keys those rows already use and fill them with the values above. If the
schema requires a field this table does not cover, leave it empty or use the
same value a row without a service uses. Do not invent a URL or a port.

## What Open should point at

`re/README.md`. Two concrete forms, pick the one the tab's other rows use:

- The file in a checkout on the Mac: `<sparkduet checkout>/re/README.md`.
- The file on GitHub:
  `https://github.com/zorost/sparkduet/blob/<branch>/re/README.md`, where
  `<branch>` is the branch this work was pushed to. The link changes if the
  work later moves to another branch.

There is no local web page for this desk and none should be linked.

## What this row is not

- It is not a GPU app. It does not swap the lane or load a model.
- It is not like Nava or VoiceStudio. Those have something to start and
  stop. REA here is a command that runs and exits. Copying their start and
  stop behaviour would add buttons that do nothing, or worse, buttons wired
  to a lane action.
- It does not need `command-center/sync.sh` to be run by this work. Whoever
  edits `index.html` follows the Command Center's own process.

## When to add the row

After both of these have passed on Spark 1 and been recorded in
`re/docs/TEST.md`:

1. `re/re.sh check` exits 0 on the chassis.
2. `re/re.sh model <slug>` exits 0 on one real model file.

Until then the desk is unproven on aarch64, and a catalog row would
advertise something nobody has run. If the catalog has a way to mark a row
as planned, the row may be added in that state earlier.
