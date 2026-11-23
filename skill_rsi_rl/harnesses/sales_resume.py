"""Recover a Sales rollout's Python state from sealed observations and a Docker image."""

from copy import deepcopy
from dataclasses import asdict
from datetime import datetime
import gzip
from hashlib import sha256
import json
from pathlib import Path
import re
import subprocess

from . import activate_benchmark

activate_benchmark()

from src.interface import Query, Response
from src.runs.common import filter_init_params
from src.tasks.sales_prediction.task import SalesPredictionTask
from src.trace_storage import serialize_instance_outcome

from .poker_resume import _observation_payload, _query_payload


_OUTPUT = re.compile(
    r"(?:<exception>(.*?)</exception>\n)?"
    r"<returncode>(-?[0-9]+)</returncode>\n<output>\n(.*)</output>",
    re.DOTALL,
)
_ELIDED = re.compile(
    r"(?:<exception>(.*?)</exception>\n)?<returncode>(-?[0-9]+)</returncode>\n"
    r"<warning>\n.*?\n</warning><output_head>\n(.{5000})\n</output_head>"
    r"\s*<elided_chars>\n([0-9]+) characters elided\n</elided_chars>"
    r"\s*<output_tail>\n(.{5000})\n</output_tail>", re.DOTALL,
)
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")


def _recorded_result(command: str, observation: str, step: int, max_steps: int) -> dict:
    if match := _OUTPUT.fullmatch(observation):
        exception_info, code, output = match.groups()
    elif match := _ELIDED.fullmatch(observation):
        exception_info, code, head, omitted, tail = match.groups()
        # The benchmark records only the first 2000 characters internally.
        # The observation renderer uses only the head, tail, and omitted length.
        output = head + " " * int(omitted) + tail
    elif (
        observation == "Analysis phase complete. Provide your predictions now."
        and command.strip() == "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
    ):
        return {"output": "", "returncode": 0, "exception_info": None, "submitted": True}
    elif step == max_steps and observation == (
        f"Max steps ({max_steps}) reached. Provide your predictions now."
    ):
        raise ValueError("Sales max-step extraction hides the preceding command's Docker result")
    else:
        raise ValueError("Sales saved Docker observation cannot be decoded")
    return {"output": output, "returncode": int(code), "exception_info": exception_info}


class _SalesReplay(SalesPredictionTask):
    _offline = True
    _data_room = False
    _pending_command = None
    checkpoint_query = None

    def step(self, response):
        result = super().step(response)
        self.checkpoint_query = result.next_query
        return result

    def _start_container(self):
        if not self._offline:
            super()._start_container()

    def _push_data_room(self):
        self._data_room = True
        try:
            return super()._push_data_room()
        finally:
            self._data_room = False

    def _clean_agent_workspace(self):
        if not self._offline:
            super()._clean_agent_workspace()

    def _execute(self, command: str) -> dict:
        if not self._offline:
            return super()._execute(command)
        if self._data_room:
            return {"output": "", "returncode": 0, "exception_info": None}
        if self._pending_command is None or self._pending_command[0] != command:
            raise ValueError("Sales replay attempted an unrecorded agent command")
        _, result = self._pending_command
        self._pending_command = None
        return result

    def replay_step(self, query: Query, action, saved: dict, observation: str):
        if query.response_schema.__name__ == "BashCommandResponse" and action.command.strip():
            self._pending_command = (
                action.command,
                _recorded_result(action.command, observation, self.current_steps + 1,
                                 self.max_steps_per_instance),
            )
        result = self.step(Response(action=action, metadata=deepcopy(saved["metadata"])))
        if self._pending_command is not None:
            raise ValueError("Sales replay failed to consume its recorded command")
        return result


def _check_next_query(batch: Path, attempt: dict, prefix: dict, query: Query) -> None:
    interactions = prefix["interactions"]
    last_metadata = interactions[-1]["response"]["metadata"]
    runtime = Path(last_metadata["runtime_directory"])
    if (
        runtime.parent != batch / "native" / "sales_prediction"
        or runtime.resolve() != runtime
        or runtime.relative_to(batch).as_posix() not in attempt["runtime_directories"]
    ):
        raise ValueError("Sales next query source differs from the sealed attempt")
    capture = runtime / "capture" / "queries.jsonl"
    if capture.resolve() != capture:
        raise ValueError("Sales query capture must be attempt-local")
    with capture.open(encoding="utf-8") as stream:
        next_rows = [
            row for line in stream
            if (row := json.loads(line)).get("kind") == "query_start"
            and row.get("interaction_step") == len(interactions) + 1
        ]
    expected = {
        "instance_id": query.instance_id, "instance_index": query.instance_index,
        "prompt": query.prompt, "response_schema": query.response_schema.model_json_schema(),
        "query_metadata": query.metadata, "query_feedback": asdict(query.feedback) if query.feedback else None,
        "queued_feedback": [interactions[-1]["observation"]["content"]],
    }
    if len(next_rows) != 1 or expected != {key: next_rows[0].get(key) for key in expected}:
        raise ValueError("Sales next query differs from the recorded pending DSH query")


def replay_sales_prefix(job: dict, prefix: dict, attempt: dict) -> tuple[_SalesReplay, Query]:
    """Reconstruct exactly the accepted Sales prefix without shell, Docker, or model execution."""
    params = job["task_params"]
    index = job["index"]
    expected = job["expected_instances"]
    recorded_task = prefix["task"]
    execution = prefix["execution"]
    interactions = prefix["interactions"]
    outcomes = prefix["instance_outcomes"]
    batch = Path(job["batch"]).resolve(strict=True)
    source = batch / attempt["trace_path"]
    if (
        job["task"] != "sales_prediction" or job["phase"] != "run"
        or attempt["task"] != job["task"] or attempt["phase"] != job["phase"]
        or attempt["index"] != index or attempt["status"] not in {"failed", "interrupted"}
        or attempt["fingerprint"] != job["fingerprint"]
        or attempt.get("execution_fingerprint", job["fingerprint"]) not in job["accepted_executions"]
        or source.resolve() != source
        or source != (batch / "checkpoints" / job["task"] / "attempts"
                      / f"run-{index}" / attempt["attempt_id"] / "live.json.gz")
        or prefix["status"] != "running" or prefix["phase"] != "run" or prefix["result"] is not None
        or recorded_task["name"] != job["task"] or prefix["schedule"] != params["schedule"]
        or recorded_task["params"] != {**params, "expected_num_instances": expected}
        or execution["run_index"] != index or execution["run_group_id"] != job["group"]
        or execution["total_interactions"] != len(interactions)
        or not interactions or len(outcomes) >= expected
    ):
        raise ValueError("Sales prefix does not match a sealed attempt of this rollout")

    compressed = source.read_bytes()
    if sha256(compressed).hexdigest() != attempt["trace_sha256"] or json.loads(gzip.decompress(compressed)) != prefix:
        raise ValueError("Sales replay payload differs from the sealed source")
    task = _SalesReplay(**filter_init_params(SalesPredictionTask, params))
    if task.num_instances != expected:
        raise ValueError("Official Sales schedule length differs from the saved rollout")
    if (run_index := params.get("run_index", params.get("rollout_index"))) is not None:
        task.prepare_run(run_index)
    query = task.reset()
    for step_number, interaction in enumerate(interactions, 1):
        prior = interaction["query"]
        if step_number != interaction["step_number"] or prior != _query_payload(query, prior, task, None):
            raise ValueError(f"Sales query differs at interaction {step_number}")
        saved = interaction["response"]
        action = query.response_schema.model_validate(saved["action"])
        if saved["action_type"] != "structured" or action.model_dump() != saved["action"]:
            raise ValueError(f"Sales action differs at interaction {step_number}")
        observed = interaction["observation"]
        try:
            step = task.replay_step(query, action, saved, observed["content"])
        except ValueError as exc:
            raise ValueError(f"Sales execution boundary differs at interaction {step_number}: {exc}") from exc
        if (
            step.done != interaction["done"] or step.next_query is None
            or _observation_payload(step.observation) != observed
        ):
            raise ValueError(f"Sales observation differs at interaction {step_number}")
        query = step.next_query

    actual = [serialize_instance_outcome(row) for row in task.get_instance_outcomes()]
    if len(actual) != len(outcomes) or any(
        row != {key: saved[key] for key in row} for row, saved in zip(actual, outcomes, strict=True)
    ):
        raise ValueError("Sales replay changed the scored instance outcomes")
    _check_next_query(batch, attempt, prefix, query)
    return task, query


def _utc(timestamp: str) -> datetime:
    parsed = datetime.fromisoformat(timestamp)
    if parsed.utcoffset() is None or parsed.utcoffset().total_seconds() != 0:
        raise ValueError("Sales snapshot has an invalid UTC timestamp")
    return parsed


def sales_snapshot_registration(batch: Path, attempt: dict, prefix: dict, task: _SalesReplay) -> dict:
    """Bind a recovery registration to the exact sealed source, before any container startup."""
    batch = batch.resolve(strict=True)
    source = batch / "recovery" / "sales_prediction" / f"{attempt['attempt_id']}.json"
    if source.resolve() != source or source.parent != batch / "recovery" / "sales_prediction":
        raise ValueError("Sales snapshot registration must be batch-local")
    registration = json.loads(source.read_text(encoding="utf-8"))
    runtime = Path(prefix["interactions"][-1]["response"]["metadata"]["runtime_directory"])
    if (
        registration.get("schema_version") != 1
        or registration.get("attempt_id") != attempt["attempt_id"]
        or registration.get("job_id") != attempt["job_id"]
        or registration.get("trace_sha256") != attempt["trace_sha256"]
        or runtime.parent != batch / "native" / "sales_prediction"
        or registration.get("runtime_directory") != runtime.relative_to(batch).as_posix()
        or registration["runtime_directory"] not in attempt["runtime_directories"]
        or registration.get("source_image") != task.docker_image
        or registration.get("working_dir") != "/app" or type(registration.get("mount_count")) is not int
        or registration["mount_count"] != 0
        or not re.fullmatch(r"[0-9a-f]{64}", registration["container_id"])
        or not re.fullmatch(r"minisweagent-[0-9a-f]{8}", registration["container_name"])
        or _DIGEST.fullmatch(registration["image_id"]) is None
        or ("parent_image_id" in registration and _DIGEST.fullmatch(registration["parent_image_id"]) is None)
        or not (_utc(attempt["created_at"]) <= _utc(registration["container_created_at"])
                <= _utc(attempt["ended_at"]) <= _utc(registration["saved_at"]))
        or registration.get("total_interactions", task.total_interactions) != task.total_interactions
    ):
        raise ValueError("Sales snapshot registration differs from the sealed rollout and container boundary")
    return registration


def _docker_inspect(executable: str, object_type: str, name: str) -> dict:
    result = subprocess.run(
        [executable, object_type, "inspect", name], capture_output=True, text=True, check=True, timeout=30,
    )
    return json.loads(result.stdout)[0]


def verify_sales_image(
    task: _SalesReplay, snapshot_image_id: str, docker: str = "docker", *, parent_image_id: str | None = None,
) -> None:
    """Check a supplied snapshot's immutable image identity and local image lineage."""
    if _DIGEST.fullmatch(snapshot_image_id) is None:
        raise ValueError("Sales recovery requires an immutable image ID")
    base = _docker_inspect(docker, "image", task.docker_image)
    snapshot = _docker_inspect(docker, "image", snapshot_image_id)
    base_layers = base["RootFS"]["Layers"]
    snapshot_layers = snapshot["RootFS"]["Layers"]
    parent_layers = base_layers
    if parent_image_id is not None:
        if _DIGEST.fullmatch(parent_image_id) is None:
            raise ValueError("Sales recovery requires an immutable parent image ID")
        parent = _docker_inspect(docker, "image", parent_image_id)
        parent_layers = parent["RootFS"]["Layers"]
        if parent["Id"] != parent_image_id or parent_layers[:len(base_layers)] != base_layers:
            raise ValueError("Sales snapshot parent does not descend from the official base image")
    if (
        snapshot["Id"] != snapshot_image_id
        or snapshot["Config"].get("Cmd") != ["sleep", "2h"]
        or snapshot["Os"] != base["Os"] or snapshot["Architecture"] != base["Architecture"]
        or snapshot_layers[:-1] != parent_layers
        or snapshot["Config"].get("Volumes") or snapshot["Config"]["WorkingDir"] != "/app"
        or snapshot["Config"].get("Entrypoint") != base["Config"].get("Entrypoint")
        or snapshot["Config"].get("User") != base["Config"].get("User")
        or snapshot["Config"].get("Env") != base["Config"].get("Env")
    ):
        raise ValueError("Sales snapshot image does not match the official container ancestry and configuration")


def attach_sales_snapshot(
    task: _SalesReplay, snapshot_image_id: str, docker: str = "docker", *, parent_image_id: str | None = None,
) -> str:
    """Attach a fresh network-isolated container backed by the inspected immutable image."""
    from minisweagent.environments.docker import DockerEnvironment

    if not task._offline or task._env is not None:
        raise ValueError("Sales snapshot requires replayed Python state before attachment")
    verify_sales_image(task, snapshot_image_id, docker, parent_image_id=parent_image_id)
    environment = DockerEnvironment(
        image=snapshot_image_id, cwd="/app", timeout=120, pull_timeout=600,
        executable=docker, run_args=["--rm", "--network=none", "--pull=never"],
    )
    try:
        container = _docker_inspect(docker, "container", environment.container_id)
        if (
            container["Id"] != environment.container_id or container["Image"] != snapshot_image_id
            or not container["State"]["Running"] or container["Mounts"]
            or container["HostConfig"]["NetworkMode"] != "none"
        ):
            raise ValueError("Sales resumed container does not match the verified image or isolation")
    except BaseException:
        environment.cleanup()
        raise
    task._env = environment
    task._offline = False
    return environment.container_id
