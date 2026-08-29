#!/usr/bin/env python3
"""Does a long-context retrieval on the SGLang Flash-Next lane survive a
concurrent request?

Background: the 64k sweep of MiaAI-Lab's NVFP4 KV eval returned 64 tokens of
token id 0 ("!!!!") for niah/32768@0% while a second request was in flight, and
passed 3/3 at the same depth when run serially. Token id 0 with HTTP 200 is the
SM121 silent-corruption signature (sglang#36537). A lane that only lies while
two clients talk to it is worse than a slow lane, because Chat, OpenCode, and
Hermes all overlap requests.

This plants a passkey at the front of a ~33k-token haystack (the position that
failed), then fires it together with a filler request, repeatedly. It reports
per trial whether the passkey came back, whether the output degenerated to
token 0, and what the partner returned.

Not a benchmark: latency here is contended by construction.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

WORDS = ("alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo "
         "lima mike november oscar papa quebec romeo sierra tango uniform "
         "victor whiskey xray yankee zulu").split()


def haystack(lines: int, seed: int) -> str:
    rng = random.Random(seed)
    return "\n".join(
        f"log {i:05d}: " + " ".join(rng.choice(WORDS) for _ in range(6))
        for i in range(lines)
    )


def post(url: str, payload: dict, timeout: float) -> tuple[str, int, float]:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    t = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        doc = json.load(resp)
    text = (doc["choices"][0]["message"].get("content") or "")
    used = (doc.get("usage") or {}).get("prompt_tokens") or 0
    return text, used, time.time() - t


def lines_for_depth(url: str, model: str, target: int, timeout: float) -> tuple[int, int]:
    """Ask the server how long a sample prompt actually is, then scale.

    Guessing tokens-per-line is how a '32k' test silently becomes a 16k test,
    which is exactly the depth the corruption did not appear at.
    """
    probe_lines = 200
    body = haystack(probe_lines, seed=1)
    _, prompt_tokens, _ = post(url, {
        "model": model, "temperature": 0, "max_tokens": 1,
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": [{"role": "user", "content": body}],
    }, timeout)
    per_line = prompt_tokens / probe_lines
    return max(1, round(target / per_line)), prompt_tokens


def is_token_zero(text: str) -> bool:
    """The corruption prints as a run of '!' and nothing else."""
    s = text.strip()
    return len(s) >= 8 and set(s) == {"!"}


def is_token0_id_run(ids, run: int = 16) -> bool:
    """Server-side abort rule from MiaAI-Lab 0f95001."""
    return len(ids) >= run and all(token == 0 for token in ids[-run:])


def trial(url: str, model: str, needle: str, prompt: str, partner: str,
          timeout: float) -> dict:
    long_req = {
        "model": model, "temperature": 0, "max_tokens": 64,
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": [{"role": "user", "content": prompt}],
    }
    fill_req = {
        "model": model, "temperature": 0, "max_tokens": 200,
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": [{"role": "user", "content": partner}],
    }
    with ThreadPoolExecutor(max_workers=2) as pool:
        a = pool.submit(post, url, long_req, timeout)
        time.sleep(0.15)  # let the long prefill start first, as in the sweep
        b = pool.submit(post, url, fill_req, timeout)
        long_text, long_prompt_tok, long_s = a.result()
        fill_text, _, fill_s = b.result()
    return {
        "prompt_tokens": long_prompt_tok,
        "found": needle in long_text,
        "long_token_zero": is_token_zero(long_text),
        "partner_token_zero": is_token_zero(fill_text),
        "long_s": round(long_s, 1),
        "partner_s": round(fill_s, 1),
        "long_head": long_text.strip()[:60],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:30000")
    ap.add_argument("--model", default="RadixArk/Qwen3.8-Flash-Next-NVFP4")
    ap.add_argument("--trials", type=int, default=6)
    ap.add_argument("--target-tokens", type=int, default=33000,
                    help="calibrated prompt depth, not a line count")
    ap.add_argument("--timeout", type=float, default=600.0)
    args = ap.parse_args()

    url = args.base_url.rstrip("/") + "/v1/chat/completions"
    needle = "2ZDJ-L2BU-RNA5"
    lines, probe_tok = lines_for_depth(url, args.model, args.target_tokens, args.timeout)
    print(f"calibrated {probe_tok/200:.1f} tok/line -> {lines} lines "
          f"for ~{args.target_tokens} prompt tokens\n")
    bad = 0
    for i in range(args.trials):
        # Fresh salt per trial so no trial can be answered from the radix cache.
        body = haystack(lines, seed=1000 + i)
        prompt = (f"The passkey is {needle}. Remember it.\n\n" + body +
                  "\n\nWhat is the passkey? Reply with the passkey only.")
        partner = f"Write {40 + i} words about how a lighthouse works. Plain prose."
        r = trial(url, args.model, needle, prompt, partner, args.timeout)
        flag = "TOKEN-ZERO" if r["long_token_zero"] else ("ok" if r["found"] else "MISS")
        if r["long_token_zero"] or not r["found"] or r["partner_token_zero"]:
            bad += 1
        print(f"trial {i+1}/{args.trials}  {flag:11} depth={r['prompt_tokens']:>6} tok "
              f"long={r['long_s']:6.1f}s partner={r['partner_s']:6.1f}s "
              f"partner_token_zero={r['partner_token_zero']} | {r['long_head']!r}", flush=True)

    print(f"\n{args.trials - bad}/{args.trials} concurrent trials clean")
    if bad:
        print("REPRODUCED: the lane corrupts or misses under concurrency. "
              "Do not leave it serving long context.")
    else:
        print("No corruption reproduced under concurrency at this depth.")
    return 1 if bad else 0


def _self_check() -> None:
    assert is_token_zero("!!!!!!!!!!!!")
    assert is_token_zero("  !!!!!!!!!!  ")
    assert not is_token_zero("!!!")                     # too short to be the loop
    assert not is_token_zero("Wow!!!!!!!!! great")      # real text with bangs
    assert not is_token_zero("")
    assert not is_token0_id_run([0] * 15)
    assert is_token0_id_run([7, 3] + [0] * 16)
    assert not is_token0_id_run([0] * 15 + [1])
    h = haystack(10, seed=1)
    assert len(h.splitlines()) == 10
    assert haystack(10, seed=1) == h, "same seed must give the same haystack"
    assert haystack(10, seed=2) != h, "different seed must change the filler"
    print("ok")


if __name__ == "__main__":
    if "--self-check" in sys.argv:
        _self_check()
    else:
        sys.exit(main())
