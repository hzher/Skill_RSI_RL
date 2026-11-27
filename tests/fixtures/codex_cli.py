#!/usr/bin/env python3
"""Synthetic CLI; only regression tests use these task-independent responses."""
import json
import os
from pathlib import Path
import sys
import tomllib
import urllib.request

if "--version" in sys.argv:
    print("codex-cli synthetic-rl-test")
    raise SystemExit
assert not (Path(os.environ["CODEX_HOME"]) / "config.toml").exists()
assert "CODEX_TEST_KEY" not in os.environ
settings = {}
for position, argument in enumerate(sys.argv):
    if argument == "-c":
        key, value = sys.argv[position + 1].split("=", 1)
        settings[key] = tomllib.loads("value=" + value)["value"]
assert not any(key.startswith("mcp_servers") for key in settings)
if "model_providers.skill_rsi_capture" in settings:
    provider = settings["model_providers.skill_rsi_capture"]
    body = {"model": "synthetic", "input": [{"role": "user", "content": sys.argv[-1]}], "stream": False}
    request = urllib.request.Request(provider["base_url"] + "/responses", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=10) as response:
        assert json.load(response)["status"] == "completed"
reply = {"answer": "ok"}
if '"thinking"' in sys.argv[-1] and '"action"' in sys.argv[-1]:
    reply = {"thinking": "Synthetic protocol regression", "action": "FOLD"}
print(json.dumps({"type": "thread.started", "thread_id": "synthetic-rl-thread"}))
print(json.dumps({"type": "item.completed", "item": {"type": "agent_message", "text": json.dumps(reply)}}))
print(json.dumps({"type": "turn.completed", "usage": {
    "input_tokens": 15 if "resume" in sys.argv else 12,
    "output_tokens": 5 if "resume" in sys.argv else 3}}))
