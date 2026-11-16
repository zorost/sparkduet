#!/usr/bin/env python3
"""House adaptive verify length for speculative decode.

The drafter still proposes its configured K. After each step the scheduler
keeps a per-request EMA of accepted draft tokens and, on the next step,
verifies only a prefix from {2, 4, 7}. The prefix is batch-uniform (the
max chosen K in the step) so mixed batches stay on one CUDA-graph shape.

Default off. Set GLM53_ADAPTIVE_K=ema to enable. Public MTP k=2 is a
no-op: the prefix set never exceeds two drafts. Fail-closed if the
scheduler anchors drift.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

P = Path("/usr/local/lib/python3.12/dist-packages/vllm/v1/core/sched/scheduler.py")
MARK = "# [glm53-adaptive-verify-k]"

HELPER_NEEDLE = "from vllm.compilation.cuda_graph import CUDAGraphStat\n"

HELPERS = '''def _glm53_adaptive_k_enabled():
    return os.environ.get("GLM53_ADAPTIVE_K", "").strip().lower() in (
        "ema",
        "1",
        "on",
        "true",
    )


def _glm53_adaptive_k_set():
    raw = os.environ.get("GLM53_ADAPTIVE_K_SET", "2,4,7")
    ks = tuple(int(x) for x in raw.split(",") if x.strip())
    return ks if ks else (2, 4, 7)


def _glm53_adaptive_k_choose(ema, ks):
    for k in ks:
        if k + 0.25 >= ema:
            return k
    return ks[-1]


def _glm53_adaptive_k_observe(request, num_accepted):
    if not _glm53_adaptive_k_enabled():
        return
    alpha = float(os.environ.get("GLM53_ADAPTIVE_K_ALPHA", "0.25"))
    prev = getattr(request, "_glm53_verify_ema", None)
    if prev is None:
        request._glm53_verify_ema = float(num_accepted)
    else:
        request._glm53_verify_ema = (1.0 - alpha) * prev + alpha * float(num_accepted)


def _glm53_adaptive_k_trim(requests, scheduled, num_scheduled_tokens):
    if not scheduled or not _glm53_adaptive_k_enabled():
        return scheduled
    ks = _glm53_adaptive_k_set()
    chosen = []
    for rid in scheduled:
        req = requests.get(rid) if requests is not None else None
        ema = getattr(req, "_glm53_verify_ema", None) if req is not None else None
        chosen.append(ks[-1] if ema is None else _glm53_adaptive_k_choose(ema, ks))
    batch_k = max(chosen)
    out = {}
    for rid, toks in scheduled.items():
        if len(toks) > batch_k:
            dropped = len(toks) - batch_k
            out[rid] = toks[:batch_k]
            if rid in num_scheduled_tokens:
                num_scheduled_tokens[rid] = max(1, num_scheduled_tokens[rid] - dropped)
        else:
            out[rid] = toks
    return out


'''

TRIM_OLD = """        # Dynamic speculative decoding: compute optimal K
        num_spec_tokens_to_schedule = self.num_spec_tokens
        if self.dynamic_sd_lookup is not None and len(num_scheduled_tokens) > 0:
            num_spec_tokens_to_schedule = self.dynamic_sd_lookup[
                len(num_scheduled_tokens)
            ]
"""

TRIM_NEW = """        # Dynamic speculative decoding: compute optimal K
        num_spec_tokens_to_schedule = self.num_spec_tokens
        if self.dynamic_sd_lookup is not None and len(num_scheduled_tokens) > 0:
            num_spec_tokens_to_schedule = self.dynamic_sd_lookup[
                len(num_scheduled_tokens)
            ]
        scheduled_spec_decode_tokens = _glm53_adaptive_k_trim(
            self.requests, scheduled_spec_decode_tokens, num_scheduled_tokens
        )  # [glm53-adaptive-verify-k]
        total_num_scheduled_tokens = sum(num_scheduled_tokens.values())
"""

OBSERVE_OLD = """                num_draft_tokens = len(scheduled_spec_token_ids)
                num_sampled = self.num_sampled_tokens_per_step
                num_accepted = max(len(generated_token_ids) - num_sampled, 0)
                num_rejected = num_draft_tokens - num_accepted
"""

OBSERVE_NEW = """                num_draft_tokens = len(scheduled_spec_token_ids)
                num_sampled = self.num_sampled_tokens_per_step
                num_accepted = max(len(generated_token_ids) - num_sampled, 0)
                num_rejected = num_draft_tokens - num_accepted
                _glm53_adaptive_k_observe(request, num_accepted)  # [glm53-adaptive-verify-k]
"""


def replace_once(text: str, old: str, new: str, label: str) -> str:
    n = text.count(old)
    if n != 1:
        raise SystemExit(f"patch_adaptive_verify_k: expected one {label} target, found {n}")
    return text.replace(old, new, 1)


def apply(path: Path = P) -> int:
    if not path.is_file():
        print(f"patch_adaptive_verify_k: FATAL missing {path}", flush=True)
        return 1
    text = path.read_text()
    if MARK in text and TRIM_NEW in text and OBSERVE_NEW in text:
        print("patch_adaptive_verify_k: already applied")
        return 0
    if TRIM_OLD not in text or OBSERVE_OLD not in text:
        print("patch_adaptive_verify_k: FATAL anchors drifted", flush=True)
        return 1
    if HELPERS not in text:
        if text.count(HELPER_NEEDLE) != 1:
            print("patch_adaptive_verify_k: FATAL helper insert not unique", flush=True)
            return 1
        text = text.replace(HELPER_NEEDLE, HELPERS + HELPER_NEEDLE, 1)
    text = replace_once(text, TRIM_OLD, TRIM_NEW, "verify-trim")
    text = replace_once(text, OBSERVE_OLD, OBSERVE_NEW, "ema-observe")
    path.write_text(text)
    mode = os.environ.get("GLM53_ADAPTIVE_K", "off")
    print(f"patch_adaptive_verify_k: applied (GLM53_ADAPTIVE_K={mode})")
    return 0


class _Req:
    def __init__(self, ema=None):
        self._glm53_verify_ema = ema


def self_check() -> int:
    os.environ["GLM53_ADAPTIVE_K"] = "ema"
    os.environ.pop("GLM53_ADAPTIVE_K_SET", None)
    os.environ.pop("GLM53_ADAPTIVE_K_ALPHA", None)
    ns = {"os": os}
    exec(HELPERS, ns)
    choose = ns["_glm53_adaptive_k_choose"]
    observe = ns["_glm53_adaptive_k_observe"]
    trim = ns["_glm53_adaptive_k_trim"]
    ks = (2, 4, 7)
    assert choose(1.0, ks) == 2
    assert choose(2.0, ks) == 2
    assert choose(2.3, ks) == 4
    assert choose(4.0, ks) == 4
    assert choose(6.5, ks) == 7
    assert choose(7.0, ks) == 7
    r = _Req()
    observe(r, 2)
    assert r._glm53_verify_ema == 2.0
    observe(r, 6)
    assert abs(r._glm53_verify_ema - 3.0) < 1e-9
    scheduled = {"a": [1, 2, 3, 4, 5, 6, 7], "b": [1, 2, 3, 4, 5, 6, 7]}
    tokens = {"a": 8, "b": 8}
    reqs = {"a": _Req(1.5), "b": _Req(6.5)}
    out = trim(reqs, scheduled, tokens)
    assert [len(v) for v in out.values()] == [7, 7]
    assert tokens == {"a": 8, "b": 8}
    tokens = {"a": 8}
    out = trim({"a": _Req(1.5)}, {"a": [1, 2, 3, 4, 5, 6, 7]}, tokens)
    assert out["a"] == [1, 2]
    assert tokens["a"] == 3
    os.environ["GLM53_ADAPTIVE_K"] = "off"
    tokens = {"a": 8}
    out = trim({"a": _Req(1.5)}, {"a": [1, 2, 3, 4, 5, 6, 7]}, tokens)
    assert len(out["a"]) == 7
    assert tokens["a"] == 8
    print("patch_adaptive_verify_k: self-check ok")
    return 0


def main() -> int:
    if "--self-check" in sys.argv:
        return self_check()
    return apply()


if __name__ == "__main__":
    raise SystemExit(main())
