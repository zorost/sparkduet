#!/usr/bin/env python3
"""
router.py, SparkDuet lane arbiter and OpenAI-compatible front door.

OPTIONAL component: run it only when serving more than one lane. For daily
single-lane serving, point clients at the lane port directly.

One endpoint (:ROUTER_PORT) → per-request lane selection → backend vLLM.

Routing rules (explicit, logged, no silent fallbacks):
  1. `X-SparkDuet-Lane: depth|fleet|split` header or model suffix `@lane` pins the lane.
  2. prompt_tokens >= P_PROMPT_THRESHOLD            → Lane P (split) if configured, else D.
  3. Lane D inflight > D_HIGH_WATER_SEQS and prompt is short → spill to Lane F.
  4. default → Lane D.

Streaming is passed through chunk-by-chunk (SSE works end to end).
Overload is a typed refusal (413/429 with the shortfall), never silent queueing.
Dependency-free (stdlib only) so it can be audited in one sitting.
"""
from __future__ import annotations
import http.client, json, os, threading, time, urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name, "")
    if not v:
        return default
    if not v.isdigit():
        raise SystemExit(f"router: {name} must be a decimal integer, got {v!r}")
    return int(v)

ROUTER_PORT   = _env_int("ROUTER_PORT", 30008)
PD_THRESHOLD  = _env_int("P_PROMPT_THRESHOLD", 8192)
D_HIGH_WATER  = _env_int("D_HIGH_WATER_SEQS", 5)
LANE_MAX      = _env_int("LANE_MAX_INFLIGHT", 32)
D_CEILING     = _env_int("D_MAX_MODEL_LEN", 1048576)
HEAD          = os.environ.get("MASTER_ADDR", "127.0.0.1")
WORKER        = os.environ.get("WORKER_HOST", "127.0.0.1")

BACKENDS = {  # lane -> list of (host, port, label); fleet spreads, others singletons
    "depth": [(HEAD, _env_int("D_PORT", 30000), "depth")],
    "fleet": [(HEAD, _env_int("F_PORT_A", 30010), "fleet-A"),
              (WORKER, _env_int("F_PORT_B", 30010), "fleet-B")],
    "split": [(HEAD, _env_int("P_DECODE_PORT", 30020), "split-decode")],
}
HOP_HEADERS = {"connection", "keep-alive", "transfer-encoding", "te", "trailer",
               "proxy-authenticate", "proxy-authorization", "upgrade", "content-length"}

class State:
    def __init__(self):
        self.lock = threading.Lock()
        self.inflight = {label: 0 for bs in BACKENDS.values() for _, _, label in bs}
        self.draining = False
        self.spec_recommendation: dict = {}

    def pick(self, lane: str):
        with self.lock:
            cands = BACKENDS[lane]
            total = sum(self.inflight[l] for _, _, l in cands)
            if total >= LANE_MAX:
                return None
            host, port, label = min(cands, key=lambda c: self.inflight[c[2]])
            self.inflight[label] += 1
            return host, port, label

    def release(self, label: str):
        with self.lock:
            self.inflight[label] = max(0, self.inflight[label] - 1)

    def depth_load(self) -> int:
        with self.lock:
            return self.inflight.get("depth", 0)

STATE = State()

def count_prompt_tokens(payload: dict) -> int:
    """Cheap upper-bound tokenizer: ~4 chars/token over message content."""
    n = 0
    for m in payload.get("messages", []):
        c = m.get("content")
        n += len(c) // 4 if isinstance(c, str) else 256
    return n

def choose_lane(payload: dict, headers) -> tuple[str, str]:
    explicit = headers.get("X-SparkDuet-Lane")
    model = payload.get("model", "")
    if "@" in model:
        explicit = model.rsplit("@", 1)[1]
        payload["model"] = model.rsplit("@", 1)[0]
    if explicit in BACKENDS:
        return explicit, "pinned"
    ptok = count_prompt_tokens(payload)
    if ptok >= PD_THRESHOLD and os.environ.get("LANE_SPLIT_ENABLE", "0") == "1":
        return "split", f"prompt>={PD_THRESHOLD}"
    if STATE.depth_load() > D_HIGH_WATER and os.environ.get("LANE_FLEET_ENABLE", "0") == "1":
        return "fleet", f"depth-high-water>{D_HIGH_WATER}"
    return "depth", "default"

def refused(code: int, kind: str, detail: dict) -> bytes:
    return json.dumps({"error": {"type": kind, "code": code, **detail}}).encode()

class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):  # quiet; decisions logged explicitly below
        pass

    def _admin(self, method: str, path: str, body: bytes) -> bool:
        if path == "/admin/health":
            self._reply(200, json.dumps({"ok": True, "draining": STATE.draining,
                                         "inflight": STATE.inflight,
                                         "spec_recommendation": STATE.spec_recommendation}).encode())
            return True
        if method == "POST" and path == "/admin/drain":
            STATE.draining = True
            self._reply(200, b'{"draining":true}')
            return True
        if method == "POST" and path == "/admin/undrain":
            STATE.draining = False
            self._reply(200, b'{"draining":false}')
            return True
        if method == "POST" and path == "/admin/spec-k":
            # Records the SpecAdvisor recommendation for operators. Draft depth is
            # an engine-boot parameter: applying it means a restart with the new
            # D_MTP_NUM_TOKENS at a quiet moment. The router never lies about that.
            try:
                STATE.spec_recommendation = json.loads(body or b"{}")
            except json.JSONDecodeError:
                self._reply(400, refused(400, "invalid_json", {}))
                return True
            self._reply(200, json.dumps({"recorded": True,
                                         "apply": "restart lane with recommended k"}).encode())
            return True
        return False

    def _proxy(self, method: str, path: str, body: bytes):
        if not path.startswith("/v1/"):
            self.send_error(404)
            return
        payload = {}
        if body:
            try:
                payload = json.loads(body)
            except json.JSONDecodeError:
                self._reply(400, refused(400, "invalid_json", {}))
                return

        lane, rule = ("depth", "passthrough")
        if path in ("/v1/chat/completions", "/v1/completions", "/v1/responses"):
            if STATE.draining:
                self._reply(503, refused(503, "draining", {"retry_after_s": 30}))
                return
            lane, rule = choose_lane(payload, self.headers)
            ptok = count_prompt_tokens(payload)
            max_tok = int(payload.get("max_tokens") or 32768)
            if lane == "depth" and ptok + max_tok > D_CEILING:
                self._reply(413, refused(413, "prompt_too_large_for_lane",
                            {"prompt_tokens_est": ptok, "max_tokens": max_tok,
                             "lane_ceiling": D_CEILING}))
                return

        picked = STATE.pick(lane)
        if picked is None:
            self._reply(429, refused(429, "lane_saturated",
                                     {"lane": lane, "max_inflight": LANE_MAX,
                                      "retry_after_s": 5}))
            return
        host, port, label = picked
        t0 = time.time()
        streamed = False
        try:
            conn = http.client.HTTPConnection(host, port, timeout=7200)
            conn.request(method, path, body=json.dumps(payload).encode() if body else None,
                         headers={"Content-Type": "application/json"})
            resp = conn.getresponse()
            self.send_response(resp.status)
            for k, v in resp.getheaders():
                if k.lower() not in HOP_HEADERS:
                    self.send_header(k, v)
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            while True:
                chunk = resp.read(16384)
                if not chunk:
                    break
                streamed = True
                self.wfile.write(f"{len(chunk):x}\r\n".encode() + chunk + b"\r\n")
            self.wfile.write(b"0\r\n\r\n")
            conn.close()
            print(json.dumps({"ts": int(t0), "lane": lane, "backend": label, "rule": rule,
                              "path": path, "status": resp.status,
                              "ms": int((time.time() - t0) * 1000)}), flush=True)
        except Exception as e:
            if not streamed:  # headers not sent yet → typed failure the client can act on
                try:
                    self._reply(502, refused(502, "backend_unreachable",
                                             {"lane": lane, "backend": label,
                                              "error": type(e).__name__}))
                except Exception:
                    pass
        finally:
            STATE.release(label)

    def _reply(self, code: int, data: bytes, ctype: str = "application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self._admin("GET", self.path, b""):
            return
        self._proxy("GET", self.path, b"")

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(n) if n else b""
        if self._admin("POST", self.path, body):
            return
        self._proxy("POST", self.path, body)

if __name__ == "__main__":
    srv = ThreadingHTTPServer((os.environ.get("ROUTER_HOST", "127.0.0.1"), ROUTER_PORT), Handler)
    print(f"sparkduet router on :{ROUTER_PORT} lanes={list(BACKENDS)}", flush=True)
    srv.serve_forever()
