"""Reconstruct a Codebase rollout at a verified scored issue boundary."""

from dataclasses import asdict
import json
from pathlib import Path

from . import activate_benchmark

activate_benchmark()

from src.interface import ContinualLearningTask
from src.runs.common import filter_init_params
from src.tasks.codebase_adaptation.task import BashCommandResponse, CodebaseAdaptationTask
from src.trace_storage import serialize_instance_outcome


def _require(condition, message):
    if not condition:
        raise ValueError(f"Codebase resume: {message}")


def _capture_receipts(batch, attempt):
    starts, ends, observations = [], {}, {}
    for value in attempt["runtime_directories"]:
        directory = batch / value
        _require(directory.resolve() == directory and directory.parent == batch / "native/codebase_adaptation",
                 "Runtime lies outside the attempted task")
        path = directory / "capture/queries.jsonl"
        if not path.is_file():
            continue
        _require(path.resolve() == path, "Capture path is symlinked")
        for row in map(json.loads, path.read_bytes().splitlines()):
            kind = row["kind"]
            if kind == "query_start":
                starts.append(row)
            elif kind == "observation" or (kind == "query_end" and row["status"] == "completed"):
                key = (str(directory), row["query_request_id"])
                destination = observations if kind == "observation" else ends
                _require(key not in destination, "Duplicate submission or observation receipt")
                destination[key] = row
    return starts, ends, observations


def _check_boundary_query(query, records, accepted, starts):
    feedback = asdict(query.feedback) if query.feedback is not None else None
    if feedback is not None:
        feedback["metadata"] = feedback.get("metadata") or {}
    if len(records) > accepted:
        saved = records[accepted]["query"]
        _require(saved == {"prompt": query.prompt, "instance_id": query.instance_id,
                           "instance_index": query.instance_index, "metadata": query.metadata or {},
                           "response_schema": query.response_schema.__name__, "feedback": feedback},
                 "Restart would produce a different next-issue query")
        return

    matching = [row for row in starts if row.get("instance_id") == query.instance_id
                and row.get("instance_index") == query.instance_index
                and row.get("interaction_step") == accepted + 1]
    _require(matching, "No native query_start proves the next issue prompt")
    schema = query.response_schema.model_json_schema()
    for row in matching:
        saved_feedback = row.get("query_feedback")
        if saved_feedback is not None:
            saved_feedback = {**saved_feedback, "metadata": saved_feedback.get("metadata") or {}}
        _require(row["prompt"] == query.prompt and row.get("query_metadata", {}) == (query.metadata or {})
                 and row["response_schema"] == schema and saved_feedback == feedback,
                 "Native query_start differs from the reconstructed next issue")


def restore_task_prefix(job: dict, attempt: dict, trace: dict):
    """Verify the already hash-checked trace and return a task with no container started.

    A prefix with no scores cannot prove a fresh issue boundary. Historical shell
    actions are data only, and callers start the next issue's container explicitly.
    """
    batch = Path(job["batch"]).resolve(strict=True)
    index = job["index"]
    params = trace["task"]["params"]
    trace_path = batch / attempt["trace_path"]
    attempt_root = batch / "checkpoints/codebase_adaptation/attempts" / f"run-{index}" / attempt["attempt_id"]
    _require(job["task"] == attempt["task"] == trace["task"]["name"] == "codebase_adaptation"
             and job["phase"] == attempt["phase"] == trace["phase"] == "run"
             and attempt["status"] in ("failed", "interrupted") and trace["status"] == "running"
             and attempt["fingerprint"] == job["fingerprint"]
             and attempt.get("execution_fingerprint", job["fingerprint"]) in job["accepted_executions"]
             and trace_path.parent == attempt_root and trace_path.resolve() == trace_path
             and attempt["index"] == trace["execution"]["run_index"] == index
             and (params["run_index"], params["rollout_index"]) == (index, index)
             and all(params[key] == job["task_params"][key] for key in ("schedule", "run_index", "rollout_index"))
             and trace["execution"]["run_group_id"] == job["group"], "Rollout or attempt identity differs")
    records = trace["interactions"]
    outcomes = trace["instance_outcomes"]
    _require(outcomes, "No scored issue boundary. An unfinished issue requires its original container or a separate retry")
    _require(trace["execution"]["total_interactions"] == len(records), "Interaction count differs")
    task = CodebaseAdaptationTask(**filter_init_params(CodebaseAdaptationTask, params))
    task.prepare_run(index)
    task.build_canonical_run_state()
    ContinualLearningTask.select_run_instances(task, None)
    _require(task.num_instances == params["expected_num_instances"] == job["expected_instances"]
             and 0 < len(outcomes) < len(task.instances), "Scheduled instance count differs")
    completions = [row for row in records if row["observation"]["instance_complete"]]
    _require(len(completions) == len(outcomes), "Completed observations do not match scored issues")
    accepted = completions[-1]["step_number"]
    prefix = records[:accepted]
    _require(prefix and prefix[-1] == completions[-1] and not prefix[-1]["done"], "Invalid scored issue boundary")
    starts, acknowledgements, observed = _capture_receipts(batch, attempt)

    issue = current_steps = 0
    for step_number, row in enumerate(prefix, 1):
        query = row["query"]
        observation = row["observation"]
        metadata = query["metadata"]
        instance_id = task.instances[issue].instance_id
        _require(row["step_number"] == step_number and query["instance_id"] == instance_id
                 and query["instance_index"] == metadata["issue_index"] == metadata["active_issue_index"] == issue
                 and metadata["step"] == current_steps and metadata["max_steps"] == task.max_steps_per_issue,
                 "Saved interaction ordering, issue identity or step counter differs")
        response = row["response"]
        action = BashCommandResponse.model_validate(response["action"])
        capture = response["metadata"]
        key = (capture["runtime_directory"], capture["query_request_id"])
        ack = acknowledgements.get(key)
        receipt = observed.get(key, {}).get("observation")
        _require(ack and ack["tool_call_id"] == capture["tool_call_id"]
                 and ack["normalized_action"] == response["action"], "Action lacks matching DSH acknowledgement")
        _require(receipt and {**receipt, "metadata": receipt.get("metadata") or {}} == observation,
                 "Saved observation differs from its native receipt")
        if action.command.strip():
            current_steps += 1
        if observation["instance_complete"]:
            saved = outcomes[issue]
            detail = saved["metadata"]
            timed_out = detail["timed_out"]
            _require(saved["instance_id"] == instance_id and saved["instance_index"] == issue
                     and detail["steps"] == current_steps and not detail.get("system_error"),
                     "Saved issue result has different identity or step count")
            if timed_out:
                _require(not saved["success"] and current_steps == task.max_steps_per_issue
                         and observation["content"].startswith(f"Max steps ({task.max_steps_per_issue})"),
                         "Timed-out issue receipt differs")
            else:
                verdict = "PASSED" if saved["success"] else "FAILED"
                _require(observation["content"].startswith(f"Submission {verdict} (eval status: {detail['eval_status']})."),
                         "Submission assessment receipt differs")
            entry = {"issue_id": instance_id, "issue_index": issue, "canonical_issue_index": issue,
                     "steps": current_steps, "success": saved["success"]}
            if timed_out:
                entry.update(timed_out=True, tests_passed=0, tests_failed=0)
            else:
                entry["eval_status"] = detail["eval_status"]
            task.issue_history.append(entry)
            calculated = serialize_instance_outcome(task._build_issue_instance_outcome(entry))
            _require(calculated == {key: saved[key] for key in calculated}, "Official issue scorer differs")
            issue += 1
            current_steps = 0
    _require(issue == len(outcomes) and not current_steps, "Resume boundary ends inside a scored issue")
    task.total_interactions = accepted
    task.current_issue_idx = issue
    task.current_steps = 0
    task._update_rollout_context_for_issue(issue - 1, announce=False)
    task._update_rollout_context_for_issue(issue, announce=True)
    next_query = task.build_current_query()
    _check_boundary_query(next_query, records, accepted, starts)
    _require(task._env is None, "Reconstruction unexpectedly started a container")
    kept = {**trace, "interactions": prefix,
            "execution": {**trace["execution"], "total_interactions": accepted}}
    return task, next_query, kept
