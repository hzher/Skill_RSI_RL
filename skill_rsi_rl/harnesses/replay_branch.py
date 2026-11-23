"""Score a fixed DSH memory action from a verified historical decision boundary."""

from __future__ import annotations

import math
import statistics
from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from .historical_replay import CompletedSource, build_prefix, verify_boundary
from .dsh_system import DSHSystem
from .memory_policy import _load_contract
from .poker_native_resume import _NativePrefix, prepare_native_branch, restore_native_resume
from .poker_resume import _query_payload, discard_replayed_task, replay_prefix
from .run import canonical_hash
from src.interface import require_query_instance_identity
from src.usage import serialize_usage_events, summarize_usage_events


@dataclass(frozen=True)
class BranchStart:
    source: CompletedSource
    prefix: dict
    native: _NativePrefix
    expected_query: dict
    query: str
    feedback: list[str]
    memory: dict[str, str]
    target_outcomes: tuple[tuple[str, int], ...]
    max_queries: int
    receipt: dict


def _score_window_sha256(targets: tuple[tuple[str, int], ...], max_queries: int) -> str:
    return canonical_hash([targets, max_queries, "first_supported_recall"])


def prepare_branch(source: CompletedSource, retained_interactions: int, *, memory: dict[str, str],
                   score_instances: int = 2, max_queries: int = 24) -> BranchStart:
    """Freeze the same official environment, native context and learner inputs for every branch."""
    if type(score_instances) is not int or type(max_queries) is not int or min(score_instances, max_queries) < 1:
        raise ValueError("Branch instance and Query limits must be positive integers")
    prefix, expected_query = build_prefix(source, retained_interactions)
    offset = len(prefix["instance_outcomes"])
    window = source.trace["instance_outcomes"][offset:offset + score_instances]
    if len(window) != score_instances:
        raise ValueError("The requested score window extends beyond the official schedule")
    targets = tuple((row["instance_id"], row["instance_index"]) for row in window)
    receipt = verify_boundary(source, retained_interactions)
    native = prepare_native_branch(source.batch, source.attempt, prefix)
    contract = _load_contract()
    from skill_rsi_rl.branch_data import observation_from_prefix

    observation = observation_from_prefix(prefix, expected_query, memory=memory)
    native_receipt = {key: value for key, value in native.lineage.items()
                      if key.endswith("_sha256") or key in {"session_terminal_seq", "retained_usage_rows"}}
    native_receipt["memory_file_count"] = len(native.memory)
    native_receipt["memory_files_sha256"] = canonical_hash({
        name: sha256(body).hexdigest() for name, body in native.memory.items()
    })
    receipt["native"] = native_receipt
    receipt["environment_boundary_sha256"] = receipt["boundary_sha256"]
    receipt["boundary_sha256"] = canonical_hash([receipt["environment_boundary_sha256"], native_receipt])
    receipt["state_sha256"] = contract.digest(observation)
    receipt["score_window_sha256"] = _score_window_sha256(targets, max_queries)
    receipt["scope"] = "official_environment_next_query_and_native_state"
    return BranchStart(source, prefix, native, deepcopy(expected_query), observation["query"], observation["feedback"],
                       observation["memory"], targets, max_queries, receipt)


class _BranchPolicy:
    """Use the sampled first action, then one fixed recall rule for all candidates."""

    def __init__(self, start: BranchStart, action: dict):
        self.start = start
        self.action = deepcopy(action)
        self.contract = _load_contract()
        self.memory = dict(start.memory)
        self.feedback = list(start.feedback)
        self.receipts = []
        state = {"query": start.query, "feedback": self.feedback, "memory": self.memory}
        if self.contract.digest(state) != start.receipt["state_sha256"]:
            raise ValueError("Branch input differs from the frozen learner state")
        options = self.contract.propose_options(start.query, self.feedback, self.memory)
        if not any(self.action == option["action"] for option in options):
            raise ValueError("Branch action is outside the frozen candidate support")

    def decide(self, query: str, feedback: list[str], remaining: float) -> tuple[str, dict]:
        phase = "continuation" if self.receipts else "candidate"
        if phase == "candidate":
            if query != self.start.expected_query["prompt"]:
                raise ValueError("Branch executor received a changed starting Query")
            query, action = self.start.query, self.action
        else:
            query, self.feedback, _ = self.contract.bound_state(query, self.feedback + feedback, self.memory)
            options = self.contract.propose_options(query, self.feedback, self.memory)
            action = next((option["action"] for option in options if option["action"]["op"] == "recall"), {"op": "noop"})
        before = self.contract.digest(self.memory)
        inputs = self.contract.digest({"query": query, "feedback": self.feedback, "memory": self.memory})
        self.memory, memo = self.contract.apply_action(action, self.memory)
        receipt = {
            "status": "applied", "phase": phase, "op": action["op"], "action_sha256": self.contract.digest(action),
            "input_sha256": inputs, "memory_before_sha256": before,
            "memory_after_sha256": self.contract.digest(self.memory),
            "memo_sha256": sha256(memo.encode()).hexdigest() if memo else None, "injected": bool(memo),
        }
        self.receipts.append(receipt)
        return memo, receipt


def _window_outcomes(task, offset: int, targets: tuple[tuple[str, int], ...]) -> list[dict]:
    scored = task.get_instance_outcomes()[offset:offset + len(targets)]
    outcomes = []
    for outcome, target in zip(scored, targets):
        if (outcome.instance_id, outcome.instance_index) != target:
            raise ValueError("Branch outcome identity differs from the frozen scoring window")
        if type(outcome.reward) not in {int, float} or not math.isfinite(outcome.reward) or type(outcome.success) is not bool:
            raise ValueError("Branch window has a non-scorable official outcome")
        outcomes.append({"instance_id": outcome.instance_id, "instance_index": outcome.instance_index,
                         "reward": outcome.reward, "native_success": outcome.success})
    return outcomes


def execute_branch(start: BranchStart, *, action: dict, system_params: dict, runtime_root: str | Path,
                   system_factory=DSHSystem) -> dict:
    """Run one candidate in a fresh DSH SDK runtime. Run siblings in separate workers or serially."""
    if any(system_params.get(name) is not None for name in ("memory_policy", "skill_path", "skill_sha256")):
        raise ValueError("Branch executor must use only the frozen branch policy")
    if system_params.get("tools_mode", "memory") != "memory":
        raise ValueError("Historical branch execution requires the scoped DSH memory tools")
    runtime_root = Path(runtime_root).resolve()
    if runtime_root.is_relative_to(start.source.batch):
        raise ValueError("Branch runtimes require a directory outside the source archive")
    if _score_window_sha256(start.target_outcomes, start.max_queries) != start.receipt["score_window_sha256"]:
        raise ValueError("Branch scoring window differs from the frozen plan")
    policy = _BranchPolicy(start, action)
    task, query = replay_prefix(start.source.job, start.prefix)
    system = None
    try:
        restored = _query_payload(query, start.expected_query, task, None)
        if restored != start.expected_query or canonical_hash(restored) != start.receipt["next_query_sha256"]:
            raise ValueError("Branch task restored a different starting Query")
        offset = len(start.prefix["instance_outcomes"])
        result = {
            "schema_version": 1, "status": "unscorable", "reason": "max_queries_exhausted", "queries": 0,
            "target_outcomes": [list(target) for target in start.target_outcomes], "outcomes": [],
            "reward": None, "native_success": None, "success_rate": None,
            "boundary_sha256": start.receipt["boundary_sha256"], "state_sha256": start.receipt["state_sha256"],
            "score_window_sha256": start.receipt["score_window_sha256"],
            "action_sha256": policy.contract.digest(action), "policy_receipts": policy.receipts,
        }
        system = system_factory(**{**system_params, "runtime_root": str(runtime_root)})
        try:
            lineage = restore_native_resume(system, start.native, start.prefix)
            system._memory_policy = policy
            result["runtime_directory"] = str(system.directory)
            result["relocation_feedback_added"] = lineage["relocation_feedback_added"]
            for number in range(1, start.max_queries + 1):
                require_query_instance_identity(query, context="Branch Query")
                result["queries"] = number
                response = system.respond(query)
                if len(policy.receipts) != number or (response.metadata or {}).get("memory_policy") != policy.receipts[-1]:
                    raise ValueError("Branch response has no unique policy action receipt")
                step = task.step(response)
                system.observe(step.observation, step.next_query)
                result["outcomes"] = _window_outcomes(task, offset, start.target_outcomes)
                if len(result["outcomes"]) == len(start.target_outcomes):
                    rows = result["outcomes"]
                    result.update(status="scored", reason=None, reward=statistics.fmean(row["reward"] for row in rows),
                                  native_success=all(row["native_success"] for row in rows),
                                  success_rate=statistics.fmean(row["native_success"] for row in rows))
                    break
                if step.done or step.next_query is None:
                    result["reason"] = "task_ended_before_score_window"
                    break
                query = step.next_query
        except (OSError, RuntimeError, ValueError) as exc:
            result.update(status="failed", reason="branch_execution_error", error_type=type(exc).__name__, error=str(exc))
        result["usage"] = summarize_usage_events(serialize_usage_events(system.consume_usage_events()))
        artifacts = system.get_run_artifacts()
        result["executor"] = {key: artifacts.get(key) for key in ("model", "provider")}
        return result
    finally:
        try:
            if system is not None:
                system.close()
        finally:
            discard_replayed_task(start.source.job["task"], task)
