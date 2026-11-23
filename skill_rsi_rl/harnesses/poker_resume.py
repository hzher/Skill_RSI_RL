"""Replay an accepted task prefix and continue it with the official runner."""

from contextlib import suppress
from copy import deepcopy
from pathlib import Path
import re

from . import activate_benchmark

activate_benchmark()

from src.interface import ContinualLearningTask, Query, Response, observation_marks_instance_complete, run_task
from src.registry import get_task_class
from src.runs.common import filter_init_params
from src.trace_storage import InteractionStep, TraceRecorder, serialize_instance_outcome


PREFIX_TASKS = frozenset({
    "exploitable_poker", "blind_spectrum_monitoring", "database_exploration", "codebase_adaptation", "sales_prediction",
})


def _observation_payload(observation):
    return {
        "content": observation.content,
        "instance_complete": observation_marks_instance_complete(observation),
        "metadata": observation.metadata or {},
    }


def discard_replayed_task(task_name: str, task: ContinualLearningTask) -> None:
    if task_name == "database_exploration":
        with suppress(Exception):
            task.evaluate()
    elif task_name in {"codebase_adaptation", "sales_prediction"}:
        task._cleanup_container()


def _query_payload(query, saved, task, database_paths, runtime=None):
    metadata = dict(query.metadata or {})
    if database_paths is not None and (new_path := task._temp_db_path) is not None:
        previous = saved["metadata"].get("db_path")
        if not isinstance(previous, str):
            raise ValueError("Database replay has no recorded private database path")
        old_path = Path(previous)
        if not old_path.is_absolute() or not re.fullmatch(r"tmp[a-z0-9_]+\.db", old_path.name):
            raise ValueError("Database replay path is not a task-private temporary database")
        current = str(new_path)
        if metadata.get("db_path") != current:
            raise ValueError("Database query does not reference the active task-private copy")
        mapped = database_paths.setdefault((runtime, current), previous)
        if mapped != previous or len(set(database_paths.values())) != len(database_paths):
            raise ValueError("Database replay changed the private path mapping within a runtime and stage")
        metadata["db_path"] = previous
    return {
        "prompt": query.prompt, "instance_id": query.instance_id, "instance_index": query.instance_index,
        "metadata": metadata, "response_schema": query.response_schema.__name__,
        "feedback": _observation_payload(query.feedback) if query.feedback is not None else None,
    }


def replay_prefix(job: dict, prefix: dict, *, db_boundaries=None) -> tuple[ContinualLearningTask, Query]:
    """Recover task state by applying saved actions without contacting a model."""
    params = job["task_params"]
    recorded_task = prefix["task"]
    recorded_params = recorded_task["params"]
    execution = prefix["execution"]
    task_name = job["task"]
    index = job["index"]
    expected = job["expected_instances"]
    interactions = prefix["interactions"]
    outcomes = prefix["instance_outcomes"]
    if (
        task_name not in PREFIX_TASKS
        or job["phase"] != "run"
        or prefix["phase"] != "run"
        or prefix["status"] != "running"
        or recorded_task["name"] != task_name
        or prefix["schedule"] != params["schedule"]
        or recorded_params != {**params, "expected_num_instances": expected}
        or any(source.get("run_index") != index or source.get("rollout_index") != index
               for source in (params, recorded_params))
        or execution["run_group_id"] != job["group"]
        or execution["run_index"] != index
        or execution["total_interactions"] != len(interactions)
        or prefix["result"] is not None
        or not interactions
        or len(outcomes) >= expected
    ):
        raise ValueError(f"Saved {task_name} prefix does not match the requested rollout")

    task_class = get_task_class(task_name)
    task = task_class(**filter_init_params(task_class, params))
    if task.num_instances != expected:
        raise ValueError(f"Official {task_name} schedule length differs from the saved rollout")
    database_paths = {} if task_name == "database_exploration" else None
    db_boundaries = db_boundaries or {}
    previous_runtime = None
    try:
        task.prepare_run(index)
        query = task.reset()
        for step_number, interaction in enumerate(interactions, 1):
            prior_query = interaction["query"]
            saved = interaction["response"]
            runtime = saved["metadata"].get("runtime_directory") if database_paths is not None else None
            if database_paths is not None and step_number > 1:
                permitted = db_boundaries.get(step_number, (previous_runtime, previous_runtime))
                if (previous_runtime, runtime) != permitted:
                    raise ValueError(f"Database replay changed runtime at unverified interaction {step_number}")
            previous_runtime = runtime
            if interaction["step_number"] != step_number or prior_query != _query_payload(
                query, prior_query, task, database_paths, runtime,
            ):
                raise ValueError(f"{task_name} replay query differs at interaction {step_number}")
            action = query.response_schema.model_validate(saved["action"])
            if saved["action_type"] != "structured" or action.model_dump() != saved["action"]:
                raise ValueError(f"{task_name} replay action differs at interaction {step_number}")
            step = task.step(Response(action=action, metadata=deepcopy(saved["metadata"])))
            if (
                step.done != interaction["done"]
                or _observation_payload(step.observation) != interaction["observation"]
                or step.next_query is None
            ):
                raise ValueError(f"{task_name} replay feedback differs at interaction {step_number}")
            query = step.next_query

        actual = [serialize_instance_outcome(outcome) for outcome in task.get_instance_outcomes()]
        if len(actual) != len(outcomes) or any(
            result != {key: saved[key] for key in result}
            for result, saved in zip(actual, outcomes, strict=True)
        ):
            raise ValueError(f"{task_name} replay changed the saved instance outcomes")
    except BaseException:
        discard_replayed_task(task_name, task)
        raise
    return task, query


class _ResumedTraceRecorder(TraceRecorder):
    def __init__(self, job, prefix, live_path, task, system):
        super().__init__(
            system_name="dsh",
            task_name=job["task"],
            system_params={
                **job["system_params"], "capture_identity": system.capture_identity,
            },
            task_params=deepcopy(prefix["task"]["params"]),
            run_group_id=job["group"],
            run_index=job["index"],
            task_brief=task.get_agent_brief(),
            phase="run",
            live_trace_path=live_path,
        )
        self.start_time = prefix["execution"]["start_time"]
        self._prefix_interactions = deepcopy(prefix["interactions"])
        self._prefix_outcomes = deepcopy(prefix["instance_outcomes"])
        self.interactions = [InteractionStep(**deepcopy(row)) for row in self._prefix_interactions]
        self.instance_outcomes = list(task.get_instance_outcomes())
        self._recording_interaction = False

    def record_interaction(self, step_number, *args, **kwargs):
        self._recording_interaction = True
        try:
            return super().record_interaction(
                step_number + len(self._prefix_interactions), *args, **kwargs,
            )
        finally:
            self._recording_interaction = False

    def write_live_snapshot(self, result=None, *, extra=None):
        # The official runner records the interaction before syncing its outcomes.
        if not self._recording_interaction:
            super().write_live_snapshot(result, extra=extra)

    def sync_instance_outcomes(self, outcomes):
        changed = super().sync_instance_outcomes(outcomes)
        if not changed:
            self.write_live_snapshot()
        return changed

    def _build_trace_payload(self, *, result, include_artifacts=True):
        trace = super()._build_trace_payload(
            result=result, include_artifacts=include_artifacts,
        )
        trace["interactions"][:len(self._prefix_interactions)] = deepcopy(self._prefix_interactions)
        trace["instance_outcomes"][:len(self._prefix_outcomes)] = deepcopy(self._prefix_outcomes)
        if trace["result"] is not None:
            trace["result"]["instance_outcomes"] = trace["instance_outcomes"]
        return trace


def resume_rollout(job: dict, system, prefix: dict, live_path: Path, *, replayed=None) -> dict:
    """Run the remaining decisions using the already attached DSH session."""
    task, query = replayed if replayed is not None else replay_prefix(job, prefix)
    recorder = _ResumedTraceRecorder(job, prefix, live_path, task, system)
    result = run_task(
        task, system, trace_recorder=recorder, show_progress=False,
        reset_system=False, initial_query=query, phase="run",
    )
    return recorder.finalize(result)
