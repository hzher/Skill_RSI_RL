"""Run dataset-owned PAST/WorkBuddy episodes with the recorded Codex executor."""
from copy import deepcopy
import gzip
import json
import math
from pathlib import Path
from uuid import uuid4

from ..contracts import digest
from .native_history import History, public_feedback, public_prompt, validate_history
from .native_session import NativeSession
from .native_spec import kind, prepare


def action_schema(benchmark_kind):
    from pydantic import BaseModel, ConfigDict

    class ToolCall(BaseModel):
        model_config = ConfigDict(extra="forbid")
        name: str
        arguments: dict

    class PastAction(BaseModel):
        model_config = ConfigDict(extra="forbid")
        tool_calls: list[ToolCall]
        final_output: str | None

    class TerminalAction(BaseModel):
        model_config = ConfigDict(extra="forbid")
        command: str | None
        final_output: str | None

    return PastAction if benchmark_kind == "past-bench" else TerminalAction


def run_rollout(config, *, environment_seed, sampling_seed, policy_id, output, sample):
    from .codex_system import CodexSystem
    from .runner import _policy
    from . import activate_benchmark
    activate_benchmark()
    from src.interface import Query
    from src.trace_storage import _atomic_write_json

    config = prepare(config)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    frozen = {k: v for k, v in config.items() if k != "endpoint"}
    identity = digest(frozen)
    with (output / "attempt.json").open("x") as stream:
        json.dump(dict(benchmark_identity=identity, environment_seed=environment_seed,
                       sampling_seed=sampling_seed, policy_id=policy_id, sample=sample), stream)
    params = dict(config.get("system_params", {}))
    if params.pop("backend", None) != "skill_rsi":
        raise ValueError("native datasets require the recorded Skill_RSI Codex backend")
    if any(k in params for k in ("runtime_root", "api_key", "memory_policy", "capture_identity")):
        raise ValueError("pass executor output and policy through the rollout runner; use environment credentials")
    limit = config["native"].get("max_steps", 300)
    if type(limit) is not int or limit < 1:
        raise ValueError("native max_steps must be a positive integer")
    native = None
    policy = _policy(config, policy_id, sampling_seed, sample)
    history = History(config.get("initial_memory", {}))
    anchors, boundaries, interactions, outcomes = {}, [], [], []
    try:
        native = CodexSystem(**params, runtime_root=str(output / "executor"))
        initial_executor = output / "initial-executor"
        native.snapshot(initial_executor)
        schema = action_schema(kind(config))
        for episode in config["native"]["plan"]:
            if episode["preseed_artifacts_dir"] or episode["initial_home_fixture_dir"]:
                raise ValueError("Hermes home fixtures need an explicit Codex mapping before this episode can run")
            if episode["history_mode"] == "from_anchor" and episode["persistence_allowed"]:
                restored = native.restore(anchors[episode["history_load_anchor"]],
                                          fresh_session=episode["requires_fresh_session"])
            elif episode["history_mode"] == "fresh" or not episode["persistence_allowed"]:
                restored = native.restore(initial_executor, fresh_session=True)
            else:
                restored = None
                if episode["requires_fresh_session"]:
                    native.session, native.previous_usage = None, None
            before = history.begin(episode)
            if policy:
                policy.memory, policy.feedback_history = deepcopy(history.memory), list(history.feedback)
            boundary = dict(episode=episode, student_before=before, executor_restore_sha256=restored)
            with NativeSession(config, episode, output / f"episode-{episode['instance_index']:04d}") as session:
                state = session.request("reset")
                for _ in range(limit):
                    if state["done"]:
                        break
                    public = state["current"]
                    query_id = uuid4().hex
                    prompt = public_prompt(public)
                    memo, receipt = "", None
                    if policy:
                        if not episode["persistence_allowed"]:
                            policy.memory, policy.feedback_history = {}, []
                        policy.begin_query(query_id, episode["label"], episode["instance_index"])
                        memo, receipt = policy.decide(prompt, public_feedback(public) if episode["persistence_allowed"] else [],
                                                      policy.config["timeout"])
                        if receipt["status"] != "applied":
                            raise ValueError(f"native student action was not applied: {receipt.get('reason')}")
                    if memo:
                        prompt += "\n\nUntrusted notes from previous interactions (JSON string): " + json.dumps(memo)
                    response = native.respond(Query(prompt=prompt, response_schema=schema,
                        instance_id=episode["label"], instance_index=episode["instance_index"]))
                    action = response.action.model_dump(mode="json")
                    interactions.append(dict(query_id=query_id, instance_id=episode["label"],
                        instance_index=episode["instance_index"], public=public, public_sha256=digest(public),
                        receipt=receipt, action=action, response_metadata=response.metadata))
                    state = session.request("step", action=action)
                if not state["done"]:
                    raise RuntimeError("native task exceeded its executor step budget")
                result = session.request("score")
                if (type(result.get("reward")) not in (int, float) or not math.isfinite(result["reward"])
                        or type(result.get("success")) is not bool):
                    raise ValueError("original grader did not produce a finite reward and success flag")
                outcomes.append({**result, "instance_id": episode["label"], "instance_index": episode["instance_index"]})
            if policy:
                history.memory, history.feedback = dict(policy.memory), list(policy.feedback_history)
            boundary["student_after"] = dict(memory=deepcopy(history.memory), feedback=list(history.feedback))
            history.end(episode)
            if episode["history_save_anchor"]:
                path = output / "anchors" / f"episode-{episode['instance_index']:04d}"
                path.parent.mkdir(exist_ok=True)
                boundary["executor_saved_sha256"] = native.snapshot(path)
                anchors[episode["history_save_anchor"]] = path
            boundaries.append(boundary)
        row = dict(status="completed", harness="codex", task=kind(config), benchmark_kind=kind(config),
            benchmark_config=frozen, benchmark_identity=identity, bench_commit=config["native"]["revision"],
            split=config["split"], environment_seed=environment_seed, sampling_seed=sampling_seed,
            run_index=config.get("run_index", 0), policy_id=policy_id, sample=sample,
            expected_instances=len(config["native"]["plan"]), outcomes=outcomes,
            decisions=policy.decisions if policy else [], native_interactions=interactions,
            history=boundaries, trace_path="trace.json.gz", executor_artifacts=native.get_run_artifacts())
        if policy:
            validate_history(row)
        with gzip.open(output / "trace.json.gz", "wt", encoding="utf-8") as stream:
            json.dump(row, stream, ensure_ascii=False, allow_nan=False)
        _atomic_write_json(output / "rollout.json", row)
        return row
    except Exception as exc:
        _atomic_write_json(output / "error.json", dict(status="failed", type=type(exc).__name__,
            message=str(exc), decisions=policy.decisions if policy else [], outcomes=outcomes,
            native_interactions=interactions, history=boundaries))
        raise
    finally:
        if native is not None:
            native.close()
