#!/usr/bin/env python3
"""CPU-only check of scripts/ab_glm_probe.py against a fake streaming OpenAI server."""
import json
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

root = Path(__file__).resolve().parent.parent
probe = root / "scripts/ab_glm_probe.py"
fails = 0


def check(name, cond):
    global fails
    print(("PASS" if cond else "FAIL"), name)
    fails += 0 if cond else 1


def make_handler(leak_eos=False, break_tools=False, cached=None):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            body = json.dumps({"object": "list", "data": [{"id": "m", "max_model_len": 262144}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            prompt_words = sum(len(str(m.get("content", "")).split()) for m in req["messages"])
            n = min(req.get("max_tokens", 16), 40)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()

            def send(obj):
                self.wfile.write(b"data: " + json.dumps(obj).encode() + b"\n\n")
                self.wfile.flush()

            if req.get("tools") and not break_tools:
                send({"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "function": {
                    "name": "get_weather", "arguments": '{"city": "Lis'}}]}}]})
                send({"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "function": {
                    "arguments": 'bon"}'}}]}, "finish_reason": "tool_calls"}]})
                n = 12
            else:
                for i in range(n):
                    send({"choices": [{"index": 0, "delta": {"content": "tok "}}]})
                if leak_eos:
                    send({"choices": [{"index": 0, "delta": {"content": "<|assistant|>"}}]})
                send({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
            usage = {"prompt_tokens": int(prompt_words * 1.3), "completion_tokens": n}
            if cached is not None:
                usage["prompt_tokens_details"] = {"cached_tokens": cached}
            send({"choices": [], "usage": usage})
            self.wfile.write(b"data: [DONE]\n\n")
    return H


def run_against(handler, label, out):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        r = subprocess.run([sys.executable, str(probe), "run", "--base-url",
                            f"http://127.0.0.1:{srv.server_address[1]}/v1", "--model", "m",
                            "--label", label, "--out", str(out), "--trials", "1",
                            "--max-tokens", "16", "--long-words", "300", "--ready-s", "10"],
                           capture_output=True, text=True, timeout=120)
    finally:
        srv.shutdown()
    return r


with tempfile.TemporaryDirectory() as d:
    d = Path(d)
    r = run_against(make_handler(), "exl3", d / "before.json")
    check("probe run exits 0", r.returncode == 0)
    b = json.loads((d / "before.json").read_text())
    check("window read from /v1/models", b["window"] == 262144)
    check("decode measured", b["decode_code_tps"] > 0 and b["decode_prose_tps"] > 0)
    check("aggregate measured", b["aggregate4_tps"] > 0)
    check("cold prompt measured", b["prompt_tokens"] > 300 and b["cold_ttft_s"] >= 0)
    check("warm turn measured", b["warm_ttft_s"] >= 0 and b["warm_prompt_tokens"] > b["prompt_tokens"])
    check("cached tokens absent when not reported", b["warm_cached_tokens"] is None)
    check("streamed tool call reassembled and parsed", b["tool_call_ok"] is True)
    check("clean reply counts as stopping", b["stops_ok"] is True)

    r = run_against(make_handler(leak_eos=True, break_tools=True, cached=4096), "tensorfold", d / "bad.json")
    a = json.loads((d / "bad.json").read_text())
    check("leaked <|assistant|> fails the stop check", a["stops_ok"] is False)
    check("no tool call fails the tool check", a["tool_call_ok"] is False)
    check("cached tokens read when reported", a["warm_cached_tokens"] == 4096)

    t = subprocess.run([sys.executable, str(probe), "table", str(d / "before.json"), str(d / "bad.json")],
                       capture_output=True, text=True)
    check("table renders", t.returncode == 0 and "| Measure | Before: exl3 | After: tensorfold |" in t.stdout)
    check("table is ladder-labelled", "<!-- ladder: M-here -->" in t.stdout)
    check("table has every row", t.stdout.count("\n|") >= 13)
    v = subprocess.run([sys.executable, str(probe), "verdict", str(d / "before.json"), str(d / "bad.json")],
                       capture_output=True, text=True)
    check("verdict refuses a broken after", v.returncode == 1 and "tool call did not parse" in v.stdout)

    good = dict(b, label="tensorfold", warm_ttft_s=b["warm_ttft_s"] - 0.001 if b["warm_ttft_s"] > 0.001 else 0,
                decode_code_tps=b["decode_code_tps"] * 1.2, decode_prose_tps=b["decode_prose_tps"] * 1.2)
    slow = dict(b, warm_ttft_s=good["warm_ttft_s"] + 1.0)
    (d / "slow.json").write_text(json.dumps(slow))
    (d / "good.json").write_text(json.dumps(good))
    v = subprocess.run([sys.executable, str(probe), "verdict", str(d / "slow.json"), str(d / "good.json")],
                       capture_output=True, text=True)
    check("verdict accepts a faster, correct after", v.returncode == 0)

drv = (root / "scripts/ab_glm_engine.sh").read_text()
check("driver restores the original engine on failure", "trap restore_on_failure ERR" in drv and "set -Eeuo" in drv)
check("driver adopts only on a passing verdict", '"$ADOPT" == 1 && "$verdict_rc" == 0' in drv)
check("driver backs up env files it edits", drv.count(".ab-$STAMP.bak") >= 2)

raise SystemExit(fails)
