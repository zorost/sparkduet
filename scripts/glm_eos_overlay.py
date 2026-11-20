#!/usr/bin/env python3
"""Write a generation_config.json that also ends a reply at extra GLM tokens.

GLM-5.3-Flash can close a reply with <|assistant|>, which the checkpoint's
generation_config.json does not list among its end tokens. An engine that stops
only at the listed ids then writes a second turn until max_tokens. This reads
the staged checkpoint, finds each named token's id, and writes a copy of
generation_config.json with those ids appended to eos_token_id. The checkpoint
itself is never written; the lane mounts the copy over the original read-only.

usage: glm_eos_overlay.py MODEL_DIR OUT_FILE [TOKENS]
  TOKENS  comma-separated token texts (default "<|assistant|>"); an empty
          string writes an unchanged copy.

exit 0 written, 1 a named token or the base end-token list is missing, 2 usage or I/O.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

DEFAULT_TOKENS = "<|assistant|>"


def token_ids(model: Path) -> dict[str, int]:
    """Every special or vocabulary token text the checkpoint names, to its id."""
    ids: dict[str, int] = {}
    tok = model / "tokenizer.json"
    if tok.is_file():
        data = json.loads(tok.read_text())
        vocab = (data.get("model") or {}).get("vocab")
        if isinstance(vocab, dict):
            ids.update({k: int(v) for k, v in vocab.items()})
        for item in data.get("added_tokens") or []:
            ids[item["content"]] = int(item["id"])
    cfg = model / "tokenizer_config.json"
    if cfg.is_file():
        for k, v in (json.loads(cfg.read_text()).get("added_tokens_decoder") or {}).items():
            ids.setdefault(v["content"], int(k))
    return ids


def base_eos(model: Path, gen: dict) -> list[int]:
    """The checkpoint's own end tokens: generation_config.json first, then config.json."""
    for source in (gen, _load(model / "config.json")):
        eos = source.get("eos_token_id")
        if isinstance(eos, int):
            return [eos]
        if isinstance(eos, list) and eos:
            return [int(x) for x in eos]
    return []


def _load(path: Path) -> dict:
    return json.loads(path.read_text()) if path.is_file() else {}


def build(model: Path, tokens: list[str]) -> dict:
    gen = _load(model / "generation_config.json")
    eos = base_eos(model, gen)
    if not eos:
        raise LookupError("no eos_token_id in generation_config.json or config.json")
    known = token_ids(model) if tokens else {}
    for text in tokens:
        if text not in known:
            raise LookupError(f"token {text!r} is not in the checkpoint's tokenizer")
        if known[text] not in eos:
            eos.append(known[text])
    out = dict(gen)
    out["eos_token_id"] = eos
    return out


def main(argv: list[str]) -> int:
    if len(argv) not in (3, 4):
        print("usage: glm_eos_overlay.py MODEL_DIR OUT_FILE [TOKENS]", file=sys.stderr)
        return 2
    model, out = Path(argv[1]), Path(argv[2])
    raw = argv[3] if len(argv) == 4 else DEFAULT_TOKENS
    tokens = [t.strip() for t in raw.split(",") if t.strip()]
    if not model.is_dir():
        print(f"glm_eos_overlay: {model} is not a directory", file=sys.stderr)
        return 2
    try:
        cfg = build(model, tokens)
    except LookupError as e:
        print(f"glm_eos_overlay: {e}", file=sys.stderr)
        return 1
    except (OSError, ValueError, KeyError) as e:
        print(f"glm_eos_overlay: cannot read {model}: {e}", file=sys.stderr)
        return 2
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(out.name + ".tmp")
        tmp.write_text(json.dumps(cfg, indent=2) + "\n")
        os.replace(tmp, out)
    except OSError as e:
        print(f"glm_eos_overlay: cannot write {out}: {e}", file=sys.stderr)
        return 2
    print(f"glm_eos_overlay: eos_token_id {cfg['eos_token_id']} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
