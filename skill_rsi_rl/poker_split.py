"""Keep train/dev deals separate from the benchmark's published Poker schedules."""

import json

from .sources import verify_source


def validate_development_split(config):
    split, bench, seeds = config.get("split"), config["benchmark"], config.get("environment_seeds")
    if split not in {"train", "dev"} or bench.get("split", split) != split:
        raise ValueError("expected a consistent train/dev split")
    if (not isinstance(seeds, list) or not seeds or any(type(x) is not int or x < 0 for x in seeds)
            or len(seeds) != len(set(seeds))):
        raise ValueError("provide distinct nonnegative environment_seeds")
    if bench["task"] != "exploitable_poker":
        return
    params = bench.get("task_params", {})
    if params.get("schedule") or params.get("variant"):
        raise ValueError("Poker train/dev needs custom deals; use opponent_policy and num_instances")
    if "seed" in params and (len(seeds) != 1 or seeds[0] != params["seed"]):
        raise ValueError("task_params.seed conflicts with the Poker train/dev environment_seeds")
    root, _ = verify_source("clbench")
    schedules = root / "src/tasks/exploitable_poker/schedules"
    published = set()
    for name in ("quick_test.json", "default.json"):
        stages = json.loads((schedules / name).read_text(encoding="utf-8"))["stages"]
        published.update(stage["schedule"]["seed"] for stage in stages)
    overlap = published.intersection(seeds)
    if overlap:
        raise ValueError(f"Poker train/dev includes published holdout seeds: {sorted(overlap)}")


def verify_deal_ids(task, seed, outcomes):
    if task != "exploitable_poker":
        return
    expected = {f"poker:default:hand:seed{seed}_hand{n}" for n in range(1, len(outcomes) + 1)}
    if not expected or {row["instance_id"] for row in outcomes} != expected:
        raise ValueError(f"Poker outcomes do not match the custom deals for environment seed {seed}")
