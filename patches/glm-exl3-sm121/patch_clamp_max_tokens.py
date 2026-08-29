#!/usr/bin/env python3
"""Treat oversized max_tokens as an upper bound, not a 400.

OpenCode and the AI SDK send 9999999 as unlimited. This image's
``TokenizeParams`` is frozen and rejects ``max_output > max_model_len``
before generation is clamped to remaining context. MiaAI-Lab's 1M recipe
does not fix that: 9999999 is still larger than 1000000.

The OpenAI meaning of max_tokens is an upper bound. After this patch the
input window is the full ``max_model_len``, and a sentinel larger than the
window is clamped instead of rejected. ``get_max_tokens()`` still bounds
the actual decode to ``max_model_len - prompt_len``.
"""
from __future__ import annotations

from pathlib import Path

P = Path("/usr/local/lib/python3.12/dist-packages/vllm/renderers/params.py")
MARK = "# [clamp-max-tokens]"

INPUT_OLD = """        if self.max_total_tokens is None:
            return None

        return self.max_total_tokens - self.max_output_tokens
"""

INPUT_NEW = """        if self.max_total_tokens is None:
            return None
        # [clamp-max-tokens] OpenAI max_tokens is an upper bound, not a reservation.
        return self.max_total_tokens
"""

RAISE_OLD = """        if (
            max_output_tokens is not None
            and max_total_tokens is not None
            and max_output_tokens > max_total_tokens
        ):
            raise VLLMValidationError(
                f"{self.max_output_tokens_param}={max_output_tokens} "
                f"cannot be greater than "
                f"{self.max_total_tokens_param}={max_total_tokens=}. "
                f"Please request fewer output tokens.",
                parameter=self.max_output_tokens_param,
                value=max_output_tokens,
            )
"""

RAISE_NEW = """        if (
            max_output_tokens is not None
            and max_total_tokens is not None
            and max_output_tokens > max_total_tokens
        ):
            # [clamp-max-tokens] OpenCode/AI SDK send 9999999 as unlimited.
            object.__setattr__(self, "max_output_tokens", max_total_tokens)
            max_output_tokens = max_total_tokens
"""


def main() -> int:
    text = P.read_text()
    if MARK in text and INPUT_NEW in text and RAISE_NEW in text:
        print("patch_clamp_max_tokens: already applied")
        return 0
    if INPUT_OLD not in text or RAISE_OLD not in text:
        print("patch_clamp_max_tokens: FATAL anchors drifted", flush=True)
        return 1
    text = text.replace(INPUT_OLD, INPUT_NEW, 1).replace(RAISE_OLD, RAISE_NEW, 1)
    P.write_text(text)
    print("patch_clamp_max_tokens: applied")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
