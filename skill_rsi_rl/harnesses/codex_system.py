"""Codex executor using the installed Skill_RSI recorded Responses gateway.

The official task and scorer stay in CL-Bench. This executor owns a temporary
workspace and CODEX_HOME; no Skill RSI plugins or extension packages are loaded.
"""

import json
import os
from pathlib import Path
import shutil
from tempfile import mkdtemp
from uuid import uuid4

from . import activate_benchmark

activate_benchmark()

from src.interface import ContinualLearningSystem, Response, observation_marks_instance_complete
from src.systems.utils.structured_output import extract_json, schema_to_prompt_instruction, validate_with_coercion
from src.usage import UsageEvent

from ..skill_rsi_bridge import framework_identity
from ..contracts import digest


def _copy_tree(source, target):
    """Copy local state without following links or exporting authentication."""
    target.mkdir(parents=True, exist_ok=True)
    refs = {}
    import hashlib

    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        if relative.parts[0] in {"auth.json", "config.toml", "logs"}:
            continue
        if path.is_symlink():
            raise ValueError("Codex state snapshot contains a symbolic link")
        destination = target / relative
        if path.is_dir():
            destination.mkdir(exist_ok=True)
        elif path.is_file():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
            refs[str(relative)] = hashlib.sha256(destination.read_bytes()).hexdigest()
        else:
            raise ValueError("Codex state snapshot contains a special file")
    return refs


class CodexSystem(ContinualLearningSystem):
    def __init__(self, model, *, skill_rsi_root=None, source_id=None, runtime_root=None,
                 command="codex", timeout=300, single_conversation=True,
                 upstream=None, credential_env=None, reasoning_effort="high"):
        _, identity = framework_identity(skill_rsi_root)
        if source_id is not None and identity != source_id:
            raise ValueError("Skill_RSI implementation changed since the experiment was pinned")
        if bool(upstream) != bool(credential_env):
            raise ValueError("upstream and credential_env must be supplied together")
        if type(single_conversation) is not bool:
            raise ValueError("single_conversation must be boolean")
        from skill_rsi.adapters.codex import Adapter

        class IsolatedAdapter(Adapter):
            def environment(adapter, output, options):
                env = super().environment(output, options)
                env["CODEX_HOME"] = str(self.home)
                return env

        self.adapter = IsolatedAdapter()
        self.source_id = identity
        evidence_root = runtime_root
        self.root = Path(evidence_root).resolve() if evidence_root else Path(mkdtemp(prefix="codex-rl-evidence-"))
        self.root.mkdir(parents=True, exist_ok=True)
        self.model, self.command, self.timeout = model, command, timeout
        self.upstream, self.credential_env = upstream, credential_env
        self.single_conversation, self.reasoning_effort = single_conversation, reasoning_effort
        self.temporary = None
        self.captures = []
        self.reset()

    @property
    def name(self):
        return "codex"

    def reset(self):
        self.close()
        self.temporary = Path(mkdtemp(prefix="skill-rsi-rl-codex-"))
        self.workspace, self.home = self.temporary / "workspace", self.temporary / "home"
        self.workspace.mkdir()
        self.home.mkdir(mode=0o700)
        # Copy only authentication for account mode, never global plugin settings.
        if not self.upstream:
            auth = Path(os.environ.get("CODEX_HOME", str(Path.home() / ".codex"))) / "auth.json"
            if auth.is_file():
                shutil.copyfile(auth, self.home / "auth.json")
                (self.home / "auth.json").chmod(0o600)
        self.session = None
        self.previous_usage = None
        self.pending_feedback = []
        self._usage_events = []

    def respond(self, query):
        feedback, self.pending_feedback = self.pending_feedback, []
        if query.feedback and query.feedback.content not in feedback:
            feedback.append(query.feedback.content)
        prompt = "\n\n".join([*("PUBLIC FEEDBACK: " + text for text in feedback), query.prompt,
                                  schema_to_prompt_instruction(query.response_schema)])
        options = {"output": self.root / f"capture-{uuid4().hex}", "command": self.command,
                   "model": self.model, "timeout": self.timeout, "trace": True,
                   "native_plugin": False, "baseline_tools": False, "public_actions": True,
                   "args": ["-c", 'sandbox_mode="workspace-write"', "-c", 'approval_policy="never"',
                            "-c", 'model_reasoning_effort=' + json.dumps(self.reasoning_effort)]}
        if self.upstream:
            options["api"] = {"upstream": self.upstream, "credential_env": self.credential_env,
                              "timeout": self.timeout}
        before = self.session
        result = self.adapter.launch(project=self.workspace, state_dir=self.temporary / "state",
                                     prompt=prompt, resume=before, options=options)
        from skill_rsi.evidence.build import build
        from skill_rsi.evidence.verify import verify

        public = Path(result["path"]) / "derived/skill-rsi-v1"
        build(Path(result["path"]), "codex", format="cli", output=public, seal=True)
        verify(public)
        self.captures.append({"path": result["path"], "status": result["status"],
                              "source_id": self.source_id, "instance_id": query.instance_id})
        if (result.get("status") != "completed" or result.get("last_turn_status") != "completed"
                or not result.get("final_response")):
            raise RuntimeError(f"Codex did not complete the native turn; inspect {result['path']}")
        session = result.get("session_id") or before
        if before and session != before:
            raise ValueError("Codex resumed a different conversation")
        if not session:
            raise ValueError("Codex supplied no persistent thread identity")
        from .action_output import explicit_past_action
        fields = getattr(query.response_schema, 'model_fields', {})
        explicit = explicit_past_action(result['final_response']) if set(fields) == {'tool_calls', 'final_output'} else None
        action = validate_with_coercion(explicit if explicit is not None else extract_json(result["final_response"]),
                                        query.response_schema)
        raw = result.get("usage")
        usage = None
        if isinstance(raw, dict) and all(type(raw.get(k)) is int and raw[k] >= 0
                                         for k in ("input_tokens", "output_tokens")):
            fields = ("input_tokens", "output_tokens", "cached_input_tokens")
            current = {k: raw[k] for k in fields if type(raw.get(k)) is int and raw[k] >= 0}
            if before is None:
                usage = current
            elif self.previous_usage is not None and all(
                    k in self.previous_usage and v >= self.previous_usage[k] for k, v in current.items()):
                usage = {k: v - self.previous_usage[k] for k, v in current.items()}
            self.previous_usage = current
        else:
            self.previous_usage = None
        self.session = session
        self.record_usage_event(UsageEvent(call_type="completion", model=self.model, provider="configured_codex",
                                          input_tokens=(usage or {}).get("input_tokens"),
                                          output_tokens=(usage or {}).get("output_tokens"),
                                          cached_input_tokens=(usage or {}).get("cached_input_tokens"),
                                          raw_usage=raw, pricing_error="configured gateway pricing unknown",
                                          metadata={"scope": "thread_cumulative_delta" if usage is not None else "unknown",
                                                    "capture": result["path"], "session_id": session}))
        return Response(action=action, metadata={"capture_path": result["path"], "thread_id": session,
                         "resumed": before is not None, "skill_rsi_source_id": self.source_id,
                         "extensions": [], "plugins": [], "token_usage": usage,
                         "wire_path": result.get("wire_path"), "evidence_path": str(public)})

    def observe(self, observation, next_query=None):
        if observation.content:
            self.pending_feedback.append(observation.content)
        if not self.single_conversation and observation_marks_instance_complete(observation):
            self.session = None
            self.previous_usage = None

    def get_run_artifacts(self):
        return {"skill_rsi_source_id": self.source_id, "captures": list(self.captures),
                "plugins": [], "extensions": []}

    def snapshot(self, destination):
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=False)
        refs = {"home": _copy_tree(self.home, destination / "home"),
                "workspace": _copy_tree(self.workspace, destination / "workspace")}
        metadata = dict(source_id=self.source_id, files=refs, session=self.session,
                        previous_usage=self.previous_usage)
        (destination / "snapshot.json").write_text(json.dumps(metadata))
        return digest(metadata)

    def restore(self, source, *, fresh_session):
        source = Path(source)
        metadata = json.loads((source / "snapshot.json").read_text())
        if metadata["source_id"] != self.source_id:
            raise ValueError("Codex anchor has a different executor version")
        # Preserve absolute workspace/home paths embedded in native conversation files.
        for label, target in (("home", self.home), ("workspace", self.workspace)):
            auth = (self.home / "auth.json").read_bytes() if label == "home" and (self.home / "auth.json").exists() else None
            shutil.rmtree(target)
            refs = _copy_tree(source / label, target)
            if refs != metadata["files"][label]:
                raise ValueError("Codex anchor files changed after snapshot")
            if auth is not None:
                (target / "auth.json").write_bytes(auth)
                (target / "auth.json").chmod(0o600)
        self.session = None if fresh_session else metadata["session"]
        self.previous_usage = None if fresh_session else metadata["previous_usage"]
        self.pending_feedback = []
        return digest(metadata)

    def close(self):
        if self.temporary is not None:
            shutil.rmtree(self.temporary)
            self.temporary = None

    # The original rollout runner invokes this lifecycle method for non-DSH systems.
    _stop_container = close
