"""Save an owned Sales container at an already sealed DSH failure boundary."""

from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import re
import subprocess

from . import activate_benchmark

activate_benchmark()

from src.tasks.sales_prediction.task import SalesPredictionTask
from src.trace_storage import serialize_instance_outcome

from .sales_resume import _DIGEST, _check_next_query, _docker_inspect


def sales_task_with_checkpoint_owner(holders: list) -> type[SalesPredictionTask]:
    """Keep the live Sales task reachable until the runner seals the attempt."""
    class CapturedSales(SalesPredictionTask):
        checkpoint_query = None

        def _start_container(self):
            super()._start_container()
            holders.append(self)

        def step(self, response):
            result = super().step(response)
            self.checkpoint_query = result.next_query
            return result

    return CapturedSales


def capture_sales_failure(batch: Path, attempt: dict, prefix: dict, task, *, save_json) -> dict:
    """Commit the owned sandbox locally after verifying its completed interaction counter."""
    batch = batch.resolve(strict=True)
    trace = batch / attempt["trace_path"]
    interactions = prefix["interactions"]
    if (
        attempt["task"] != "sales_prediction" or attempt["phase"] != "run"
        or attempt["status"] not in {"failed", "interrupted"}
        or prefix["status"] != "running" or not interactions
        or trace.resolve() != trace
        or trace != (batch / "checkpoints" / "sales_prediction" / "attempts"
                     / f"run-{attempt['index']}" / attempt["attempt_id"] / "live.json.gz")
        or sha256(trace.read_bytes()).hexdigest() != attempt["trace_sha256"]
        or task.total_interactions != len(interactions) or task._env is None
    ):
        raise ValueError("Sales Docker checkpoint does not correspond to the sealed live trace")

    outcomes = prefix["instance_outcomes"]
    expected = [serialize_instance_outcome(row) for row in task.get_instance_outcomes()]
    if len(expected) != len(outcomes) or any(
        row != {key: saved[key] for key in row} for row, saved in zip(expected, outcomes, strict=True)
    ):
        raise ValueError("Sales Docker checkpoint has a different scored outcome prefix")
    if task.checkpoint_query is None:
        raise ValueError("Sales checkpoint lacks the last accepted action's next query")
    _check_next_query(batch, attempt, prefix, task.checkpoint_query)

    environment = task._env
    docker = environment.config.executable
    container_id = environment.container_id
    if re.fullmatch(r"[0-9a-f]{64}", container_id) is None:
        raise ValueError("Sales Docker checkpoint has no immutable container identity")
    container = _docker_inspect(docker, "container", container_id)
    parent_ref = environment.config.image
    parent_image = _docker_inspect(docker, "image", parent_ref)
    official_image = _docker_inspect(docker, "image", task.docker_image)
    official_layers = official_image["RootFS"]["Layers"]
    if (
        container["Id"] != container_id or container["Image"] != parent_image["Id"]
        or container["Mounts"] or not container["State"]["Running"]
        or container["HostConfig"]["NetworkMode"] != "none"
        or container["Config"]["WorkingDir"] != "/app"
        or parent_image["RootFS"]["Layers"][:len(official_layers)] != official_layers
    ):
        raise ValueError("Sales container identity, isolation, or image ancestry differs")

    result = subprocess.run(
        [docker, "top", container_id, "-eo", "pid,comm"], capture_output=True, text=True, check=True, timeout=30,
    )
    processes = [line.split(maxsplit=1) for line in result.stdout.splitlines()[1:]]
    if len(processes) != 1 or len(processes[0]) != 2 or not processes[0][0].isdigit() or processes[0][1] != "sleep":
        raise ValueError("Sales checkpoint sandbox has unfinished subprocesses")

    destination = batch / "recovery" / "sales_prediction" / f"{attempt['attempt_id']}.json"
    if destination.resolve() != destination or destination.exists():
        raise ValueError("Sales Docker recovery registration already exists or moved")
    local_tag = f"clbench-recovery/sales:{attempt['attempt_id']}"
    result = subprocess.run(
        [docker, "commit", container_id, local_tag], capture_output=True, text=True, check=True, timeout=180,
    )
    image_id = result.stdout.strip()
    if _DIGEST.fullmatch(image_id) is None:
        raise ValueError("Docker commit did not return an immutable Sales image ID")
    image = _docker_inspect(docker, "image", image_id)
    if image["Id"] != image_id or image["RootFS"]["Layers"][:-1] != parent_image["RootFS"]["Layers"]:
        raise ValueError("Committed Sales image differs from its owned container's parent")

    final_runtime = Path(interactions[-1]["response"]["metadata"]["runtime_directory"]).relative_to(batch).as_posix()
    registration = {
        "schema_version": 1,
        "attempt_id": attempt["attempt_id"],
        "job_id": attempt["job_id"],
        "trace_sha256": attempt["trace_sha256"],
        "total_interactions": len(interactions),
        "container_id": container_id,
        "container_name": container["Name"].removeprefix("/"),
        "container_created_at": container["Created"],
        "runtime_directory": final_runtime,
        "mount_count": 0,
        "source_image": task.docker_image,
        "parent_image_id": parent_image["Id"],
        "image_id": image_id,
        "local_tag": local_tag,
        "working_dir": "/app",
        "saved_at": datetime.now(timezone.utc).isoformat(),
    }
    save_json(destination, registration)
    return registration
