"""Serve greedy or sampled decisions from a frozen Qwen3 memory policy."""

import argparse
import hashlib
import json
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import BoundedSemaphore, Lock

from .contracts import (
    MODEL_PATH,
    bound_state,
    build_messages,
    digest,
    option_token_ids,
    propose_options,
    render_prompt,
    validate_options,
)


MAX_BODY_BYTES = 256 * 1024

# Changing only the default model location preserves the historical rule identity.
# Any further change to contracts.py automatically produces a new policy identity.
_LEGACY_RULES = {
    "641caf9dd3173a16f14d911b277cb36c1a8156d6f3be47f17c3701fa84e09536":
        "04c5abdc6484576b426a681a5f942b2da11d45caefea04bb60cecb7f498786f0",
}


def _file_digest(path):
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def checkpoint_policy_ids(model_path, adapter_paths):
    """Fingerprint each adapter against the same base model and intervention rules."""
    base = Path(model_path).resolve(strict=True)
    model_files = set(base.glob("*.json")) | set(base.glob("*.safetensors")) | set(base.glob("*.jinja"))
    if not (base / "config.json").is_file() or not any(path.suffix == ".safetensors" for path in model_files):
        raise ValueError("model directory must contain config.json and safetensors")
    rules = _file_digest(Path(__file__).with_name("contracts.py"))
    files = {"base": {path.name: _file_digest(path) for path in sorted(model_files)},
             "rules": _LEGACY_RULES.get(rules, rules)}
    base_id = "sha256:" + digest(files)
    policy_ids = []
    names = ("adapter_config.json", "adapter_model.safetensors")
    for adapter_path in adapter_paths:
        adapter = Path(adapter_path).resolve(strict=True)
        adapter_files = {name: _file_digest(adapter / name) for name in names}
        policy_ids.append("sha256:" + digest({**files, "adapter": adapter_files}))
    return base_id, policy_ids


def checkpoint_identities(model_path, adapter_path=None):
    """Return the base and active policy identities from one read of the model files."""
    paths = () if adapter_path is None else (adapter_path,)
    base_id, policy_ids = checkpoint_policy_ids(model_path, paths)
    return base_id, policy_ids[0] if policy_ids else base_id


def checkpoint_identity(model_path, adapter_path=None):
    """Fingerprint model, tokenizer, intervention rules and optional adapter."""
    return checkpoint_identities(model_path, adapter_path)[1]


class QwenPolicy:
    def __init__(self, model_path, adapter_path, device, max_prompt_tokens=4096):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = torch.device(device)
        if max_prompt_tokens < 1:
            raise ValueError("max_prompt_tokens must be positive")
        if self.device.type == "cuda":
            with torch.cuda.device(self.device):
                if not torch.cuda.is_bf16_supported():
                    raise ValueError("The selected CUDA device does not support bfloat16")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, use_fast=True)
        dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True, dtype=dtype,
                                                         attn_implementation="sdpa")
        if adapter_path:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter_path, is_trainable=False)
        self.model.to(self.device).eval()
        self.max_prompt_tokens = min(max_prompt_tokens, self.model.config.max_position_embeddings)
        self.vocab_size = self.model.config.vocab_size
        self.lock = Lock()

    def action_tokens(self, messages, options):
        validate_options(options)
        prompt_ids = render_prompt(self.tokenizer, messages)
        choices = option_token_ids(self.tokenizer, [option["label"] for option in options])
        if len(prompt_ids) > self.max_prompt_tokens:
            raise ValueError(f"Prompt needs {len(prompt_ids)} tokens, above {self.max_prompt_tokens}")
        if any(token >= self.vocab_size for token in (*prompt_ids, *choices)):
            raise ValueError("Prompt or option token lies outside the policy vocabulary")
        return prompt_ids, choices

    def _option_logprobs(self, prompt_ids, choices):
        from .train import option_logprobs

        ids = self.torch.tensor([prompt_ids], dtype=self.torch.long, device=self.device)
        options = self.torch.tensor([choices], dtype=self.torch.long, device=self.device)
        with self.lock, self.torch.inference_mode():
            logged = option_logprobs(self.model, ids, self.torch.ones_like(ids), options,
                                     self.torch.ones_like(options, dtype=self.torch.bool))[0]
            if not self.torch.isfinite(logged).all():
                raise RuntimeError("model returned non-finite option logits")
            return logged.to(device="cpu", dtype=self.torch.float64)

    def choose_tokens(self, prompt_ids, choices):
        return int(self._option_logprobs(prompt_ids, choices).argmax())

    def choose(self, messages, options):
        return self.choose_tokens(*self.action_tokens(messages, options))

    def decide(self, query, feedback, memory):
        options = propose_options(query, feedback, memory)
        if len(options) == 1:
            return options[0]["action"]
        messages = build_messages(query, feedback, memory, options)
        return options[self.choose(messages, options)]["action"]

    def sample(self, query, feedback, memory, query_id, seed):
        options = propose_options(query, feedback, memory)
        messages = build_messages(query, feedback, memory, options)
        prompt_ids, token_ids = self.action_tokens(messages, options)
        if len(options) == 1:
            index, logprobs = 0, [0.0]
        else:
            logged = self._option_logprobs(prompt_ids, token_ids).log_softmax(dim=-1)
            rng = self.torch.Generator(device="cpu").manual_seed(seed)
            index = int(self.torch.multinomial(logged.exp(), 1, generator=rng).item())
            logprobs = logged.tolist()
        return {"action": options[index]["action"], "option_index": index, "prompt_token_ids": prompt_ids,
                "option_token_ids": token_ids, "behavior_logprobs": logprobs}


def make_server(host, port, model_id, policy_id, decide, *, sample=None, max_inflight=4, read_timeout=5.0):
    if max_inflight < 1 or not 0 < read_timeout <= 60:
        raise ValueError("Invalid inference server request limits")
    identity = {"model": model_id, "policy_id": policy_id}
    slots = BoundedSemaphore(max_inflight)

    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            self.request.settimeout(read_timeout)
            super().setup()

        def log_message(self, format_string, *args):
            logging.debug(format_string, *args)

        def respond(self, status, result):
            body = json.dumps(result, ensure_ascii=False, separators=(",", ":")).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError, TimeoutError):
                pass

        def do_GET(self):
            if self.path == "/health":
                self.respond(200, identity)
            else:
                self.respond(404, {"error": "unknown route"})

        def do_POST(self):
            if self.path not in ("/decide", "/act") and not (self.path == "/sample" and sample is not None):
                self.respond(404, {"error": "unknown route"})
                return
            if not slots.acquire(blocking=False):
                self.respond(503, {"error": "policy busy"})
                return
            try:
                self.decide_request()
            finally:
                slots.release()

        def decide_request(self):
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= MAX_BODY_BYTES:
                    raise ValueError("invalid request size")
                state = json.loads(self.rfile.read(size))
                if not isinstance(state, dict):
                    raise ValueError("expected a JSON object")
                sampling = self.path == "/sample"
                if sampling:
                    query_id, seed = state.pop("query_id", None), state.pop("seed", None)
                    if not isinstance(query_id, str) or not query_id.strip():
                        raise ValueError("sampling requires a query ID")
                    if type(seed) is not int or not 0 <= seed < 2**64:
                        raise ValueError("sampling requires an unsigned 64-bit seed")
                if set(state) != {"query", "feedback", "memory"}:
                    raise ValueError("expected query, feedback and memory")
                query, feedback, memory = bound_state(**state)
                support = [option["action"] for option in propose_options(query, feedback, memory)]
            except TimeoutError:
                self.respond(408, {"error": "request timed out"})
                return
            except (ValueError, UnicodeError, RecursionError):
                self.respond(400, {"error": "invalid state"})
                return
            try:
                if sampling:
                    result = sample(query=query, feedback=feedback, memory=memory, query_id=query_id, seed=seed)
                    result = {**result, "query_id": query_id, "seed": seed}
                    index = result["option_index"]
                    if type(index) is not int or not 0 <= index < len(support) or result["action"] != support[index]:
                        raise RuntimeError("sampled action differs from its candidate index")
                else:
                    result = {"action": decide(query=query, feedback=feedback, memory=memory)}
                if result["action"] not in support:
                    raise RuntimeError("inference result outside the action support")
            except ValueError:
                self.respond(422, {"error": "state exceeds policy limits"})
            except RuntimeError as exc:
                logging.error("policy forward failed: %s", type(exc).__name__)
                self.respond(500, {"error": "policy forward failed"})
            else:
                self.respond(200, {**result, **identity})

    return ThreadingHTTPServer((host, port), Handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL_PATH)
    parser.add_argument("--model-id", help="model name in the harness configuration (defaults to the model directory name)")
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--port", type=int, default=8711)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-prompt-tokens", type=int, default=4096)
    parser.add_argument("--max-inflight", type=int, default=4)
    args = parser.parse_args()
    model_id = args.model_id or Path(args.model).name
    if not 0 < args.port < 65536 or min(args.max_prompt_tokens, args.max_inflight) < 1:
        parser.error("invalid port, prompt limit, or inflight request limit")
    policy_id = checkpoint_identity(args.model, args.adapter)
    policy = QwenPolicy(args.model, args.adapter, args.device, args.max_prompt_tokens)
    with make_server("127.0.0.1", args.port, model_id, policy_id, policy.decide,
                     sample=policy.sample, max_inflight=args.max_inflight) as server:
        print(json.dumps({"model": model_id, "policy_id": policy_id, "port": args.port,
                          "max_prompt_tokens": policy.max_prompt_tokens, "max_inflight": args.max_inflight}), flush=True)
        server.serve_forever()


if __name__ == "__main__":
    main()
