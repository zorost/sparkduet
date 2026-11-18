#!/usr/bin/env python3
"""CPU-only check of the published OpenCode template and harness page.

The permission logic is a port of OpenCode v1.19.6 (wildcard match, the
tools-to-permission conversion, last-matching-rule-wins, and the rule that
hides a tool from the model). It proves what the template does, not what the
chassis runs.
"""
import json
import re
from pathlib import Path

root = Path(__file__).resolve().parent.parent
oc_dir = root / "configs/opencode"
fails = 0


def check(name, cond):
    global fails
    print(("PASS" if cond else "FAIL"), name)
    fails += 0 if cond else 1


# ---- OpenCode v1.19.6 semantics ----------------------------------------------
def match(text, pattern):
    """core/src/util/wildcard.ts: * is any run, ? one char, anchored, dotall."""
    esc = re.sub(r"[.+^${}()|\[\]\\]", lambda m: "\\" + m.group(0), pattern.replace("\\", "/"))
    esc = esc.replace("*", ".*").replace("?", ".")
    if esc.endswith(" .*"):
        esc = esc[:-3] + "( .*)?"
    return re.fullmatch(esc, text.replace("\\", "/"), re.S) is not None


def agent_rules(agent):
    """v1/config/agent.ts normalize, then permission/index.ts fromConfig."""
    perm = {}
    for tool, on in (agent.get("tools") or {}).items():
        action = "allow" if on else "deny"
        if tool in ("write", "edit", "patch"):
            perm["edit"] = action
        else:
            perm[tool] = action
    perm.update(agent.get("permission") or {})
    rules = []
    for key, value in perm.items():
        if isinstance(value, str):
            rules.append((key, "*", value))
        else:
            rules.extend((key, pat, act) for pat, act in value.items())
    return rules


# Build's built-in rules that matter here: everything allowed, then the agent.
DEFAULTS = [("*", "*", "allow")]


def evaluate(permission, pattern, rules):
    for perm, pat, act in reversed(rules):
        if match(permission, perm) and match(pattern, pat):
            return act
    return "ask"


def hidden(tools, rules):
    out = set()
    for tool in tools:
        for perm, pat, act in reversed(rules):
            if match(tool, perm):
                if pat == "*" and act == "deny":
                    out.add(tool)
                break
    return out


BUILTIN = ["bash", "read", "edit", "write", "glob", "grep", "list", "task",
           "webfetch", "websearch", "todowrite", "skill", "lsp", "question"]
MCP = ["rea_binary_session", "rea_gpu_code_objects",
       "context7_resolve-library-id", "playwright-app_browser_navigate"]

# ---- opencode.json.example ---------------------------------------------------
raw = (oc_dir / "opencode.json.example").read_text()
try:
    cfg = json.loads(raw)
except ValueError as e:
    cfg = None
    print("FAIL example JSON parses:", e)
    fails += 1
if cfg is not None:
    check("example JSON parses", True)
    prov_id, _, model_id = cfg["model"].partition("/")
    prov = cfg["provider"][prov_id]
    row = prov["models"].get(model_id)
    check("model points at a declared provider row", row is not None)
    key = prov.get("options", {}).get("apiKey")
    check("no API key in the example", key is None or key.startswith("{env:"))
    check("no bearer or sk- literal", not re.search(r"sk-[A-Za-z0-9]{8}|Bearer [A-Za-z0-9]", raw))
    check("baseURL is the loopback Headroom hop",
          prov["options"]["baseURL"].startswith("http://127.0.0.1:18787"))

    # Measured 2026-11-18: engine max_model_len 1000000. Pickers carry 85%.
    ENGINE = 1_000_000
    check("context is 85% of the live window", row["limit"]["context"] == ENGINE * 85 // 100 == 850000)
    check("context never above the engine", row["limit"]["context"] <= ENGINE)
    check("output cap 16384", row["limit"]["output"] == 16384)
    check("output reservation stays above 2048", row["limit"]["output"] > 2048)

    comp = cfg.get("compaction", {})
    check("native compaction auto is off", comp.get("auto") is False)
    check("native compaction prune is off", comp.get("prune") is False)
    check("Magic Context is the compactor",
          any("opencode-magic-context" in p for p in cfg.get("plugin", [])))

    build, plan, hard = (cfg["agent"][k] for k in ("build", "plan", "hard"))
    check("build steps 60", build.get("steps") == 60)
    check("no deprecated maxSteps", "maxSteps" not in build)
    check("build effort low", build.get("reasoningEffort") == "low")
    check("plan effort high", plan.get("reasoningEffort") == "high")
    check("hard effort high", hard.get("reasoningEffort") == "high")
    check("hard is a primary agent with a description",
          hard.get("mode") == "primary" and bool(hard.get("description")))

    bash = build["permission"]["bash"]
    check("bash catch-all is the first key and allows",
          next(iter(bash.items())) == ("*", "allow"))
    for pat in ("sleep *", "*:30000*", "*:20127*", "*:20128*"):
        check(f"bash denies {pat}", bash.get(pat) == "deny")

    rules = DEFAULTS + agent_rules(build)
    for cmd in ("sleep 30", "sleep",
                "curl -s http://127.0.0.1:30000/v1/models",
                "curl 127.0.0.1:20128/v1/chat/completions -d x",
                "wget -qO- localhost:20127/health"):
        check(f"denied: {cmd}", evaluate("bash", cmd, rules) == "deny")
    for cmd in ("make -j8", "git status --porcelain", "python3 scripts/test_lane_glm.py",
                "curl -fsS http://127.0.0.1:8080/health"):
        check(f"allowed: {cmd}", evaluate("bash", cmd, rules) == "allow")

    gone = hidden(BUILTIN + MCP, rules)
    check("REA tools are off Build", {t for t in MCP if t.startswith("rea_")} <= gone)
    check("no built-in tool is hidden from Build", not (gone & set(BUILTIN)))
    check("read still works in Build", evaluate("read", "scripts/router.py", rules) == "allow")
    check("context7 and playwright-app stay on Build",
          not any(t in gone for t in MCP if not t.startswith("rea_")))
    # The trap this template avoids: a bare rea* glob matches the read tool.
    trap = DEFAULTS + agent_rules({"tools": {"rea*": False}})
    check("a bare rea* glob would hide read (keep the underscore)", "read" in hidden(BUILTIN, trap))
    check("no tools glob lacks its server underscore",
          all("_" in k for k in build.get("tools", {}) if k.endswith("*")))
    check("mcp.rea is not disabled here", cfg.get("mcp", {}).get("rea", {}).get("enabled", True) is True)


# ---- magic-context.jsonc.example --------------------------------------------
def mc_threshold(served):
    return min(36000, max(12000, served * 55 // 100))


mc_raw = (oc_dir / "magic-context.jsonc.example").read_text()
mc = json.loads("\n".join(l for l in mc_raw.splitlines() if not l.lstrip().startswith("//")))
tokens = mc.get("execute_threshold_tokens", {})
check("Magic Context token threshold is a per-model map", isinstance(tokens, dict))
check("Magic Context default is 36000", tokens.get("default") == 36000)
check("36000 is the formula at the advertised window", mc_threshold(850000) == 36000)
check("formula floor and cap hold", mc_threshold(16384) == 12000 and mc_threshold(10**7) == 36000)
check("no token threshold above the 36000 cap", all(v <= 36000 for v in tokens.values()))
check("percentage is not the only threshold",
      "execute_threshold_percentage" not in mc or bool(tokens))

# ---- rules file and harness page ---------------------------------------------
rules_md = (oc_dir / "AGENTS.md.example").read_text()
check("rules name the three ports", all(p in rules_md for p in ("30000", "20127", "20128")))
check("rules forbid sleep-polling", "sleep" in rules_md)
check("rules stay short", len(rules_md.splitlines()) <= 60)

harness = (root / "docs/HARNESS.md").read_text()
for needle in ("850000", "16384", "36000", "rea_*", "one model", "gateway",
               "unverified on chassis", "compaction"):
    check(f"HARNESS.md states {needle}", needle.lower() in harness.lower())

mine = [oc_dir / n for n in ("opencode.json.example", "magic-context.jsonc.example",
                             "AGENTS.md.example", "README.md")] + [root / "docs/HARNESS.md"]
for path in mine:
    text = path.read_text()
    rel = path.relative_to(root)
    check(f"{rel}: no em dash", chr(0x2014) not in text)
    check(f"{rel}: old 262144-door picker value is gone", "222822" not in text)
    check(f"{rel}: no restricted speculator", "dflash" not in text.lower())

raise SystemExit(fails)
