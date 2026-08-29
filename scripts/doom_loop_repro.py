#!/usr/bin/env python3
"""Reproduce Flash-Next's leftover token-id-0 (`!`) doom loop.

House port of MiaAI-Lab evals/doom_loop_repro.py (MIT, commit 0f95001).
A loop is reproduced when reasoning+content contain at least 16 consecutive
`!` characters. After the long probe, a short follow-up checks whether the
server poisoned later requests (sticky until both TP ranks restart).

Only run this against a resident Lane N. Do not run it on DeepSeek.

Usage:
  python3 scripts/doom_loop_repro.py --self-check
  python3 scripts/doom_loop_repro.py --base-url http://127.0.0.1:30000/v1
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request


def completion(base_url, key, model, body, timeout):
    body["model"] = model
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps(body).encode(),
        headers=headers,
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def summarize(result):
    choice = result["choices"][0]
    message = choice["message"]
    text = (message.get("reasoning_content") or "") + (message.get("content") or "")
    bang_runs = [len(match.group()) for match in re.finditer(r"!+", text)]
    return {
        "completion_tokens": result.get("usage", {}).get("completion_tokens"),
        "finish_reason": choice.get("finish_reason"),
        "longest_bang_run": max(bang_runs, default=0),
        "total_bangs": text.count("!"),
        "replacement_chars": text.count(chr(0xFFFD)),
    }


def _self_check() -> None:
    clean = summarize({
        "usage": {"completion_tokens": 8},
        "choices": [{"finish_reason": "stop", "message": {"content": "Hello"}}],
    })
    assert clean["longest_bang_run"] == 0
    looped = summarize({
        "usage": {"completion_tokens": 20},
        "choices": [{"finish_reason": "length", "message": {
            "reasoning_content": "plan",
            "content": "ok " + ("!" * 16),
        }}],
    })
    assert looped["longest_bang_run"] == 16
    print("ok")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:30000/v1")
    parser.add_argument("--model", default="RadixArk/Qwen3.8-Flash-Next-NVFP4")
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--tokens", type=int, default=1600)
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()

    key = os.environ.get("SGLANG_API_KEY") or os.environ.get("API_KEY") or ""

    reproduced = False
    poisoned = False
    try:
        for attempt in range(1, args.attempts + 1):
            result = completion(
                args.base_url,
                key,
                args.model,
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": (
                                "Develop a detailed architecture and rollout plan for a reliable "
                                "distributed inference service, including failure recovery, "
                                "observability, security, capacity planning, and validation."
                            ),
                        }
                    ],
                    "temperature": 0.6,
                    "max_tokens": args.tokens,
                    "min_tokens": args.tokens,
                    "ignore_eos": True,
                    "chat_template_kwargs": {"enable_thinking": True},
                },
                args.timeout,
            )
            summary = summarize(result)
            print(f"long attempt {attempt}: {json.dumps(summary, sort_keys=True)}")
            if summary["longest_bang_run"] >= 16:
                reproduced = True
                break

        follow_up = completion(
            args.base_url,
            key,
            args.model,
            {
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "Give me three words that rhyme with 'spark', then use one in a sentence."
                        ),
                    }
                ],
                "temperature": 1.0,
                "max_tokens": 256,
                "chat_template_kwargs": {"enable_thinking": True},
            },
            args.timeout,
        )
        follow_summary = summarize(follow_up)
        print(f"short follow-up: {json.dumps(follow_summary, sort_keys=True)}")
        if follow_summary["longest_bang_run"] >= 16:
            poisoned = True
    except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as error:
        print(f"request failed: {error}", file=sys.stderr)
        return 2

    if reproduced and poisoned:
        print("REPRODUCED: token-0 loop and later requests are poisoned")
        return 1
    if reproduced:
        print("REPRODUCED: token-0 loop on the long request (follow-up stayed clean)")
        return 1
    if poisoned:
        print("REPRODUCED: follow-up poisoned without a long-request bang run")
        return 1
    print("NOT REPRODUCED")
    return 0


if __name__ == "__main__":
    if "--self-check" in sys.argv:
        _self_check()
    else:
        raise SystemExit(main())
