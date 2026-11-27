"""Versioned configuration for dataset-native PAST and WorkBuddy workers."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from ..contracts import digest

KINDS = {"past-bench", "workbuddy"}


def kind(benchmark):
    return benchmark.get("kind", "clbench")


def contained(root, name):
    path = (root / name).resolve(strict=True)
    if not path.is_relative_to(root):
        raise ValueError("native resource escapes its declared repository")
    return path


def file_sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_source(repository, inventory, revision):
    root = Path(repository).resolve(strict=True)
    source = Path(inventory).resolve(strict=True)
    raw = json.loads(source.read_text())
    if isinstance(raw, list):
        refs = {row["path"]: row["sha256"] for row in raw}
    else:
        if raw.get("revision") != revision:
            raise ValueError("native source manifest revision differs")
        refs = raw["files"]
    if not refs or any(not isinstance(x, str) or len(x) != 64 for x in refs.values()):
        raise ValueError("invalid native source manifest")
    for name, expected in refs.items():
        if file_sha(contained(root, name)) != expected:
            raise ValueError(f"native source digest differs: {name}")
    return root, {"revision": revision, "inventory_sha256": file_sha(source),
                  "files_sha256": digest(refs), "files": len(refs)}


def _past_plan(root, settings):
    import yaml

    manifest = contained(root, settings["manifest"])
    document = yaml.safe_load(manifest.read_text())
    episodes, anchors, families = [], set(), set()
    for index, row in enumerate(document["episodes"]):
        task = contained(root, str((manifest.parent / row["task"]).resolve().relative_to(root)))
        if not (task / "task.yaml").is_file():
            raise ValueError("PAST episode has no native task.yaml")
        mode = row.get("history_mode", "fresh" if row.get("requires_fresh_session", True) else "continue")
        load, save = row.get("history_load_anchor", ""), row.get("history_save_anchor", "")
        if mode not in {"continue", "fresh", "from_anchor"} or mode == "from_anchor" and load not in anchors:
            raise ValueError("PAST history plan references an unavailable anchor")
        if mode != "from_anchor" and load:
            raise ValueError("only from_anchor may load a PAST anchor")
        if save:
            if save in anchors:
                raise ValueError("duplicate PAST history anchor")
            anchors.add(save)
        family = row["family_id"]
        families.add(family)
        episode = dict(instance_index=index, task=str(task.relative_to(root)), label=row["label"],
                       family_id=family, history_mode=mode, history_load_anchor=load, history_save_anchor=save,
                       requires_fresh_session=bool(row.get("requires_fresh_session", True)),
                       persistence_allowed=bool(row.get("persistence_allowed", True)))
        for field in ("preseed_artifacts_dir", "initial_home_fixture_dir"):
            value = row.get(field)
            episode[field] = str(contained(root, str((manifest.parent / value).resolve().relative_to(root))).relative_to(root)) if value else ""
        episodes.append(episode)
    if not episodes or len(families) != 1:
        raise ValueError("one native PAST family per rollout")
    count = settings.get("episode_limit", len(episodes))
    if type(count) is not int or not 1 <= count <= len(episodes):
        raise ValueError("invalid PAST training prefix length")
    return episodes[:count], next(iter(families))


def _workbuddy_plan(root, settings):
    task = contained(root, settings["task_dir"])
    import tomllib

    config = tomllib.loads((task / "task.toml").read_text())
    if config.get("steps"):
        raise ValueError("multi-step WorkBuddy tasks require the original Harbor multi-step runner")
    if not (task / "instruction.md").is_file() or not (task / "environment").is_dir():
        raise ValueError("WorkBuddy requires an original task environment and instruction")
    if config.get("environment", {}).get("mcp_servers"):
        raise ValueError("this WorkBuddy recipe requires native tools without extra MCP packages")
    task_id = config["task"]["name"]
    return [dict(instance_index=0, task=str(task.relative_to(root)), label=task_id, family_id=task_id,
                 history_mode="fresh", history_load_anchor="", history_save_anchor="",
                 requires_fresh_session=True, persistence_allowed=True,
                 preseed_artifacts_dir="", initial_home_fixture_dir="")], task_id


def prepare(benchmark, split=None):
    if kind(benchmark) not in KINDS:
        return benchmark
    result = deepcopy(benchmark)
    if result["harness"] != "codex":
        raise ValueError("native dataset integration currently supports Codex")
    if result.get("system_params", {}).get("backend") != "skill_rsi":
        raise ValueError("native datasets require the recorded Skill_RSI Codex backend")
    settings = result["native"]
    allowed = {"repository", "source_manifest", "revision", "python", "manifest", "episode_limit", "task_dir",
               "split_manifest", "split_manifest_sha256", "source_id", "plan", "lineage_id", "judge", "rpc_timeout", "max_steps"}
    if set(settings) - allowed:
        raise ValueError(f"unknown native settings: {sorted(set(settings) - allowed)}")
    root, source = verify_source(settings["repository"], settings["source_manifest"], settings["revision"])
    plan, lineage = (_past_plan if kind(result) == "past-bench" else _workbuddy_plan)(root, settings)
    if any(x["preseed_artifacts_dir"] or x["initial_home_fixture_dir"] for x in plan):
        raise ValueError("Hermes home fixtures need an explicit Codex mapping before collection")
    source_id = "sha256:" + digest(source)
    if settings.get("source_id", source_id) != source_id or settings.get("plan", plan) != plan:
        raise ValueError("native source or episode plan changed after pinning")
    partition = json.loads(Path(settings["split_manifest"]).read_text())
    if set(partition) != {"train", "dev", "holdout"} or any(not isinstance(v, list) or not v for v in partition.values()):
        raise ValueError("declare nonempty train/dev/holdout native lineages before collection")
    flattened = [x for values in partition.values() for x in values]
    if len(flattened) != len(set(flattened)):
        raise ValueError("native lineages cross train/dev/holdout partitions")
    split = split or result.get("split")
    if split not in {"train", "dev", "holdout"} or lineage not in partition[split]:
        raise ValueError("native lineage is not assigned to the requested split")
    split_sha = file_sha(Path(settings["split_manifest"]))
    if settings.get("split_manifest_sha256", split_sha) != split_sha or settings.get("lineage_id", lineage) != lineage:
        raise ValueError("native split or lineage changed after pinning")
    settings.update(source_id=source_id, plan=plan, lineage_id=lineage)
    result["split"] = split
    result["task"] = kind(result)
    result["native"]["split_manifest_sha256"] = split_sha
    return result


def benchmark_commit(benchmark):
    if kind(benchmark) in KINDS:
        return benchmark["native"]["revision"]
    from ..sources import verify_source as pinned_source
    return pinned_source("clbench")[1]


def validate_instances(row):
    if row.get("benchmark_kind") not in KINDS:
        return
    config = row["benchmark_config"]
    if digest(config) != row["benchmark_identity"] or config["native"]["revision"] != row["bench_commit"]:
        raise ValueError("native rollout provenance differs from its frozen benchmark")
    plan = config["native"]["plan"]
    if [(x["instance_index"], x["instance_id"]) for x in row["outcomes"]] != [
            (x["instance_index"], x["label"]) for x in plan]:
        raise ValueError("native outcomes differ from the original episode schedule")
