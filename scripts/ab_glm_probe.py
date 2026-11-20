#!/usr/bin/env python3
"""ab_glm_probe.py, the Lane G engine A/B: what an agent loop feels, per engine.

  run      measure the engine on BASE_URL and write one JSON file
  table    turn two run files (before, after) into a markdown table
  verdict  exit 0 when AFTER passes every gate against BEFORE, 1 otherwise

  python3 scripts/ab_glm_probe.py run --base-url http://127.0.0.1:30000/v1 \\
      --model "$G_EXL3_SERVED_NAME" --label exl3 --out results/ab/exl3.json
  python3 scripts/ab_glm_probe.py table results/ab/exl3.json results/ab/tensorfold.json
  python3 scripts/ab_glm_probe.py verdict results/ab/exl3.json results/ab/tensorfold.json

What `run` measures, each on fresh salted prompts so no earlier run's cache helps:
  decode      greedy, thinking off, 512 tokens; tokens after the first token / time
              after it; median of --trials; code and prose prompts; one stream
  aggregate   four prose streams at once; completion tokens / wall time
  cold        one ~--long-words prompt, max_tokens 1: TTFT and prompt tok/s
  warm        the same conversation's next turn: TTFT, and cached_tokens when the
              engine reports prompt_tokens_details
  tool        one offered tool; the reply must call it with parseable arguments
  stop        a one-sentence answer must end by itself (finish_reason stop, short,
              no <|assistant|> or <|user|> text)
  window      max_model_len from GET /v1/models

Stdlib only. Nothing here starts, stops or reconfigures an engine.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import statistics
import sys
import time
import urllib.request
import uuid
from pathlib import Path

CODE = ("Write a complete Python module implementing a thread-safe LRU cache with TTL "
        "eviction, full type hints and docstrings. Code only.")
PROSE = ("Write a long, detailed essay about the history of lighthouses, their keepers, "
         "and how navigation changed when they were automated.")
FILLER = ("The harbor log for this day records wind from the northwest, a falling glass, "
          "three vessels inbound and one outbound, lamps trimmed at dusk, and no damage. ")
TOOL = {"type": "function", "function": {
    "name": "get_weather", "description": "Current weather for a city.",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                   "required": ["city"]}}}
NO_THINK = {"enable_thinking": False}


def _post(base: str, body: dict, timeout: float) -> urllib.request.Request:
    return urllib.request.Request(base.rstrip("/") + "/chat/completions",
                                  data=json.dumps(body).encode(),
                                  headers={"Content-Type": "application/json"})


def stream(base: str, body: dict, timeout: float = 1800.0) -> dict:
    """One streamed chat completion: first-token time, end time, usage, text, finish."""
    body = dict(body, stream=True, stream_options={"include_usage": True})
    t0 = time.monotonic()
    first = None
    text, usage, finish, tool_calls = [], {}, None, {}
    with urllib.request.urlopen(_post(base, body, timeout), timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            if chunk.get("usage"):
                usage = chunk["usage"]
            for ch in chunk.get("choices") or []:
                d = ch.get("delta") or {}
                piece = (d.get("content") or "") + (d.get("reasoning_content") or d.get("reasoning") or "")
                if (piece or d.get("tool_calls")) and first is None:
                    first = time.monotonic()
                text.append(d.get("content") or "")
                for tc in d.get("tool_calls") or []:
                    slot = tool_calls.setdefault(tc.get("index", 0), {"name": "", "arguments": ""})
                    fn = tc.get("function") or {}
                    slot["name"] += fn.get("name") or ""
                    slot["arguments"] += fn.get("arguments") or ""
                if ch.get("finish_reason"):
                    finish = ch["finish_reason"]
    end = time.monotonic()
    return {"t0": t0, "first": first if first is not None else end, "end": end,
            "usage": usage, "text": "".join(text), "finish": finish,
            "tool_calls": [tool_calls[k] for k in sorted(tool_calls)]}


def salt() -> str:
    return f"[run {uuid.uuid4().hex[:12]}] "


def decode_once(base, model, prompt, max_tokens):
    r = stream(base, {"model": model, "temperature": 0, "max_tokens": max_tokens,
                      "chat_template_kwargs": NO_THINK,
                      "messages": [{"role": "user", "content": salt() + prompt}]})
    n = int(r["usage"].get("completion_tokens") or 0)
    after = r["end"] - r["first"]
    return (n - 1) / after if n > 1 and after > 0 else 0.0


def measure_decode(base, model, prompt, trials, max_tokens):
    return round(statistics.median(decode_once(base, model, prompt, max_tokens) for _ in range(trials)), 2)


def measure_aggregate(base, model, streams, max_tokens):
    body = lambda: {"model": model, "temperature": 0, "max_tokens": max_tokens,
                    "chat_template_kwargs": NO_THINK,
                    "messages": [{"role": "user", "content": salt() + PROSE}]}
    t0 = time.monotonic()
    with cf.ThreadPoolExecutor(streams) as ex:
        runs = list(ex.map(lambda _: stream(base, body()), range(streams)))
    wall = time.monotonic() - t0
    total = sum(int(r["usage"].get("completion_tokens") or 0) for r in runs)
    return round(total / wall, 2) if wall > 0 else 0.0


def measure_cold_warm(base, model, words):
    reps = max(1, words // len(FILLER.split()))
    long_user = salt() + FILLER * reps + "\nSummarize the log above in one sentence."
    turn1 = [{"role": "user", "content": long_user}]
    a = stream(base, {"model": model, "temperature": 0, "max_tokens": 32,
                      "chat_template_kwargs": NO_THINK, "messages": turn1})
    prompt_tokens = int(a["usage"].get("prompt_tokens") or 0)
    ttft_cold = a["first"] - a["t0"]
    turn2 = turn1 + [{"role": "assistant", "content": a["text"] or "Noted."},
                     {"role": "user", "content": "Now name the wind direction in one word."}]
    b = stream(base, {"model": model, "temperature": 0, "max_tokens": 16,
                      "chat_template_kwargs": NO_THINK, "messages": turn2})
    details = b["usage"].get("prompt_tokens_details") or {}
    return {
        "prompt_tokens": prompt_tokens,
        "cold_ttft_s": round(ttft_cold, 2),
        "cold_prefill_tps": round(prompt_tokens / ttft_cold, 1) if ttft_cold > 0 else 0.0,
        "warm_ttft_s": round(b["first"] - b["t0"], 2),
        "warm_prompt_tokens": int(b["usage"].get("prompt_tokens") or 0),
        "warm_cached_tokens": details.get("cached_tokens"),
    }


def measure_tool(base, model):
    r = stream(base, {"model": model, "temperature": 0, "max_tokens": 1024,
                      "chat_template_kwargs": NO_THINK, "tools": [TOOL], "tool_choice": "auto",
                      "messages": [{"role": "user", "content": salt() + "What is the weather in Lisbon right now? Use the tool."}]})
    for call in r["tool_calls"]:
        try:
            args = json.loads(call["arguments"] or "{}")
        except ValueError:
            continue
        if call["name"] == "get_weather" and isinstance(args, dict) and args.get("city"):
            return True
    return False


def measure_stop(base, model):
    r = stream(base, {"model": model, "temperature": 0, "max_tokens": 2048,
                      "chat_template_kwargs": NO_THINK,
                      "messages": [{"role": "user", "content": salt() + "Say hello in one short sentence."}]})
    n = int(r["usage"].get("completion_tokens") or 0)
    leaked = any(t in r["text"] for t in ("<|assistant|>", "<|user|>", "<|observation|>"))
    return r["finish"] == "stop" and 0 < n < 512 and not leaked


def window(base, model):
    with urllib.request.urlopen(base.rstrip("/") + "/models", timeout=60) as r:
        data = json.load(r).get("data") or []
    for m in data:
        if m.get("id") == model:
            return m.get("max_model_len") or m.get("context_length")
    return None


def cmd_run(a) -> int:
    started = time.time()
    res = {"label": a.label, "model": a.model, "base_url": a.base_url,
           "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "ready_s": a.ready_s, "window": window(a.base_url, a.model)}
    res["decode_code_tps"] = measure_decode(a.base_url, a.model, CODE, a.trials, a.max_tokens)
    res["decode_prose_tps"] = measure_decode(a.base_url, a.model, PROSE, a.trials, a.max_tokens)
    res["aggregate4_tps"] = measure_aggregate(a.base_url, a.model, 4, a.max_tokens)
    res.update(measure_cold_warm(a.base_url, a.model, a.long_words))
    res["tool_call_ok"] = measure_tool(a.base_url, a.model)
    res["stops_ok"] = measure_stop(a.base_url, a.model)
    res["probe_s"] = round(time.time() - started, 1)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps(res, indent=2))
    return 0


ROWS = [  # key, label, better ("up", "down", "same", "true")
    ("window", "Served window (max_model_len)", "same"),
    ("ready_s", "Switch to ready, incl. warmup (s)", "down"),
    ("decode_code_tps", "Decode, code, 1 stream (tok/s)", "up"),
    ("decode_prose_tps", "Decode, prose, 1 stream (tok/s)", "up"),
    ("aggregate4_tps", "Decode, prose, 4 streams total (tok/s)", "up"),
    ("prompt_tokens", "Cold prompt size (tokens)", "same"),
    ("cold_ttft_s", "Cold prompt, first token (s)", "down"),
    ("cold_prefill_tps", "Cold prompt read (tok/s)", "up"),
    ("warm_ttft_s", "Next turn, first token (s)", "down"),
    ("warm_cached_tokens", "Next turn, cached tokens", "up"),
    ("tool_call_ok", "Tool call parsed", "true"),
    ("stops_ok", "Reply stops on its own", "true"),
]


def _fmt(v):
    if v is None:
        return "n/a"
    if isinstance(v, bool):
        return "yes" if v else "NO"
    return f"{v:,}" if isinstance(v, int) else f"{v:,.2f}".rstrip("0").rstrip(".")


def _delta(b, a, better):
    if better == "true":
        return ""
    if not isinstance(b, (int, float)) or not isinstance(a, (int, float)) or isinstance(b, bool) or b == 0:
        return ""
    pct = (a - b) / b * 100
    return f"{pct:+.1f}%"


def table(before: dict, after: dict) -> str:
    out = ["<!-- ladder: M-here -->",
           f"| Measure | Before: {before['label']} | After: {after['label']} | Change |",
           "|---|---:|---:|---:|"]
    for key, label, better in ROWS:
        b, a = before.get(key), after.get(key)
        out.append(f"| {label} | {_fmt(b)} | {_fmt(a)} | {_delta(b, a, better)} |")
    out.append("<!-- ladder: end -->")
    out.append("")
    out.append(f"Model `{after['model']}`, greedy, thinking off, fresh salted prompts. "
               f"Before {before['date_utc']}, after {after['date_utc']}.")
    return "\n".join(out)


def gates(before: dict, after: dict) -> list[str]:
    fails = []
    if not after.get("tool_call_ok"):
        fails.append("tool call did not parse")
    if not after.get("stops_ok"):
        fails.append("a reply did not stop on its own")
    if before.get("window") and after.get("window") and after["window"] < before["window"]:
        fails.append(f"served window shrank ({before['window']} -> {after['window']})")
    num = lambda d, k: d.get(k) if isinstance(d.get(k), (int, float)) else None
    wa, wb = num(after, "warm_ttft_s"), num(before, "warm_ttft_s")
    if wa is None or wb is None or wa >= wb:
        fails.append("next-turn first token is not faster")
    for key, what in (("decode_code_tps", "code"), ("decode_prose_tps", "prose")):
        xa, xb = num(after, key), num(before, key)
        if xa is None or xb is None or xa < 0.95 * xb:
            fails.append(f"one-stream {what} decode lost more than 5 percent")
    return fails


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--base-url", default="http://127.0.0.1:30000/v1")
    r.add_argument("--model", required=True)
    r.add_argument("--label", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--trials", type=int, default=3)
    r.add_argument("--max-tokens", type=int, default=512)
    r.add_argument("--long-words", type=int, default=24000, help="about 32k tokens")
    r.add_argument("--ready-s", type=float, default=None)
    for name in ("table", "verdict"):
        s = sub.add_parser(name)
        s.add_argument("before")
        s.add_argument("after")
    a = ap.parse_args(argv)
    if a.cmd == "run":
        return cmd_run(a)
    before, after = (json.loads(Path(p).read_text()) for p in (a.before, a.after))
    if a.cmd == "table":
        print(table(before, after))
        return 0
    fails = gates(before, after)
    for f in fails:
        print("GATE FAIL:", f)
    print("verdict:", "after passes every gate" if not fails else "keep before")
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
