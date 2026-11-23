"""Local memory-policy transport for the DSH evaluation runner."""

from __future__ import annotations

import hashlib
import http.client
import json
import math
import random
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from skill_rsi_rl import contracts

CONTRACT_PATH = Path(contracts.__file__).resolve()
_RESPONSE_BYTES = 1024 * 1024


def _digest(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_config(value: object) -> dict:
    if not isinstance(value, dict) or not {"endpoint", "model", "checkpoint"} <= value.keys() <= {
        "endpoint", "model", "checkpoint", "timeout",
    }:
        raise ValueError("Memory policy config requires endpoint, model, and checkpoint, with an optional timeout")
    for key in ("endpoint", "model", "checkpoint"):
        if not isinstance(value[key], str) or not value[key].strip() or len(value[key]) > 512:
            raise ValueError(f"Memory policy {key} must be a nonempty, bounded string")
    endpoint = urlsplit(value["endpoint"])
    if (
        endpoint.scheme != "http" or endpoint.hostname not in {"127.0.0.1", "::1", "localhost"}
        or not endpoint.port or endpoint.username is not None or endpoint.password is not None
        or endpoint.path not in {"/decide", "/act", "/sample"} or endpoint.query or endpoint.fragment
    ):
        raise ValueError("Memory policy endpoint must use loopback HTTP with an explicit port and /decide, /act or /sample")
    timeout = value.get("timeout", 10.0)
    if type(timeout) not in {int, float} or not math.isfinite(timeout) or not 0 < timeout <= 60:
        raise ValueError("Memory policy timeout must be in (0, 60] seconds")
    return {**value, "timeout": float(timeout)}


def _load_contract():
    return contracts


class MemoryPolicy:
    def __init__(self, config: dict, *, initial_memory=None, sampling_seed=0, capture=False, verify_model=False):
        self.config = validate_config(config)
        self.endpoint = urlsplit(self.config["endpoint"])
        self.contract = _load_contract()
        self.initial_memory = dict(initial_memory or {})
        self.contract.validate_memory(self.initial_memory)
        self.sampling_seed = sampling_seed
        self.capture = capture
        self.verify_model = verify_model
        self.reset()

    def reset(self):
        self.memory = dict(self.initial_memory)
        self.feedback_history = []
        self.decisions = []
        self._random = random.Random(self.sampling_seed)
        self._query = {}

    def begin_query(self, query_id, instance_id, instance_index):
        self._query = {"query_id": query_id, "instance_id": instance_id, "instance_index": instance_index}

    def _request(self, method: str, path: str, data: dict | None, timeout: float) -> dict:
        connection = http.client.HTTPConnection(self.endpoint.hostname, self.endpoint.port, timeout=timeout)
        try:
            body = json.dumps(data, ensure_ascii=False).encode("utf-8") if data is not None else None
            headers = {"Content-Type": "application/json"} if body is not None else {}
            connection.request(method, path, body=body, headers=headers)
            with connection.getresponse() as response:
                if response.status != 200:
                    raise ValueError("policy_http_status")
                payload = response.read(_RESPONSE_BYTES + 1)
            if len(payload) > _RESPONSE_BYTES:
                raise ValueError("policy_response_too_large")
            result = json.loads(payload)
            if not isinstance(result, dict):
                raise ValueError("invalid_policy_response")
            return result
        finally:
            connection.close()

    def _verify_policy_id(self, result: dict) -> None:
        if result.get("policy_id") != self.config["checkpoint"]:
            raise ValueError("policy_identity_mismatch")

    def check(self) -> dict:
        result = self._request("GET", "/health", None, self.config["timeout"])
        self._verify_policy_id(result)
        if result.get("model") != self.config["model"]:
            raise ValueError("policy_model_mismatch")
        return {"policy_id": result["policy_id"], "model": result["model"]}

    def decide(self, query: str, feedback: list[str], remaining: float) -> tuple[str, dict]:
        started = time.monotonic()
        before = _digest(self.memory)
        receipt = {
            "checkpoint": self.config["checkpoint"], "status": "fallback", "op": "noop",
            "input_sha256": None, "memory_before_sha256": before,
            "memory_after_sha256": before, "memo_sha256": None, "injected": False,
        }
        memo = ""
        record = {**self._query, "policy_id": self.config["checkpoint"]}
        try:
            recent = self.feedback_history + feedback
            query, feedback, memory = self.contract.bound_state(query, recent, self.memory)
            self.feedback_history = feedback
            request = {"query": query, "feedback": feedback, "memory": memory}
            options = self.contract.propose_options(query, feedback, memory)
            if self.capture:
                record.update(state=request.copy(), options=options,
                              messages=self.contract.build_messages(query, feedback, memory, options))
            receipt["input_sha256"] = _digest(request)
            if self.endpoint.path == "/sample":
                request = {**request, "query_id": self._query["query_id"], "seed": self._random.randrange(2**63)}

            if remaining <= 0:
                raise TimeoutError
            response = self._request("POST", self.endpoint.path, request, min(remaining, self.config["timeout"]))
            self._verify_policy_id(response)
            action = response.get("action")
            index = next((i for i, option in enumerate(options) if action == option["action"]), None)
            if index is None:
                raise ValueError("policy_out_of_support")
            if self.verify_model and response.get("model") != self.config["model"]:
                raise ValueError("policy_model_mismatch")
            if self.endpoint.path == "/sample":
                self._verify_sample(response, request, options, index)
                record.update({name: response[name] for name in (
                    "seed", "prompt_token_ids", "option_token_ids", "behavior_logprobs")})
            record.update(action=action, option_index=index)
            updated_memory, memo = self.contract.apply_action(action, memory)
            applied = dict(
                status="applied", op=action["op"], reason=None,
                memory_after_sha256=_digest(updated_memory),
                memo_sha256=hashlib.sha256(memo.encode("utf-8")).hexdigest() if memo else None,
                injected=bool(memo),
            )
            if action["op"] in {"recall", "compact"}:
                applied["keys"] = action["keys"]
            elif action["op"] == "write":
                applied["key"] = action["key"]
                applied["text_sha256"] = hashlib.sha256(action["text"].encode("utf-8")).hexdigest()
            receipt.update(applied)
            self.memory = updated_memory
        except TimeoutError:
            receipt["reason"] = "policy_timeout"
        except (OSError, http.client.HTTPException):
            receipt["reason"] = "policy_connection_error"
        except (ValueError, TypeError, KeyError) as exc:
            reasons = {"policy_http_status", "policy_response_too_large", "invalid_policy_response",
                       "policy_identity_mismatch", "policy_model_mismatch", "invalid_policy_sample", "policy_out_of_support"}
            receipt["reason"] = str(exc) if str(exc) in reasons else "invalid_policy_response"
        receipt["elapsed_seconds"] = round(time.monotonic() - started, 6)
        if self.capture:
            self.decisions.append({**record, "applied": receipt["status"] == "applied", "receipt": receipt.copy()})
        return memo if receipt["status"] == "applied" else "", receipt

    @staticmethod
    def _verify_sample(response, request, options, index):
        ids, tokens, logps = (response.get(name) for name in (
            "prompt_token_ids", "option_token_ids", "behavior_logprobs"))
        if (response.get("query_id") != request["query_id"] or response.get("seed") != request["seed"]
                or type(response.get("option_index")) is not int or response["option_index"] != index
                or not isinstance(ids, list) or not ids or not isinstance(tokens, list)
                or len(options) != len(tokens) or any(type(x) is not int or x < 0 for x in (*ids, *tokens))
                or len(set(tokens)) != len(tokens) or not isinstance(logps, list) or len(logps) != len(tokens)
                or any(type(x) not in (float, int) or not math.isfinite(x) or x > 0.00001 for x in logps)
                or abs(sum(math.exp(x) for x in logps) - 1.0) > 0.0001):
            raise ValueError("invalid_policy_sample")
