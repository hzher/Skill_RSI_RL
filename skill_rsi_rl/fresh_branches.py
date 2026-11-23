"""Score memory interventions from identical fresh CL-Bench starts, on any native harness."""

import argparse
from contextlib import contextmanager
import json
import math
from pathlib import Path
from statistics import fmean
from threading import Thread
from uuid import uuid4

from .contracts import digest, propose_options, validate_memory
from .poker_split import validate_development_split, verify_deal_ids
from .replay_spec import validate_scored_state, write_spec
from .serve import make_server

PROTOCOL = "fresh_start_first_action_then_recall_v1"


@contextmanager
def forced_choice(first_index, policy_id):
    """Apply the designated first option, then use the same recall rule on all branches."""
    first = True

    def decide(query, feedback, memory):
        nonlocal first
        options = propose_options(query, feedback, memory)
        if first:
            first = False
            return options[first_index]["action"]
        return next((o["action"] for o in options if o["action"]["op"] == "recall"), options[0]["action"])

    with make_server("127.0.0.1", 0, "fixed-branch", policy_id, decide) as server:
        worker = Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            yield f"http://127.0.0.1:{server.server_port}/decide"
        finally:
            server.shutdown()
            worker.join()


def score_state(bench, seed, directory, plan_id, *, recorded=False):
    from .harnesses.run import sha256
    from .harnesses.runner import run_rollout

    decisions = None
    scores = []
    evidence = []
    window = None
    index = 0
    while decisions is None or index < len(decisions["options"]):
        policy_id = "sha256:" + digest([plan_id, seed, index])
        target = directory / f"candidate-{index}"
        if recorded:
            result = json.loads((target / "rollout.json").read_text())
        else:
            with forced_choice(index, policy_id) as endpoint:
                result = run_rollout({**bench, "endpoint": endpoint, "policy_model": "fixed-branch"},
                                     environment_seed=seed, sampling_seed=0, policy_id=policy_id,
                                     output=target, sample=False)
        steps, outcomes = result["decisions"], result["outcomes"]
        verify_deal_ids(bench["task"], seed, outcomes)
        if (result["status"] != "completed" or result["environment_seed"] != seed or result["sample"] is not False
                or result["policy_id"] != policy_id or not steps or len(outcomes) != result["expected_instances"]
                or result["harness"] != bench["harness"] or result["task"] != bench["task"] or not outcomes
                or any(not s["applied"] or s["policy_id"] != policy_id for s in steps)):
            raise ValueError(f"incomplete execution or policy delivery at {target}")
        start = steps[0]
        if start["option_index"] != index:
            raise ValueError(f"requested candidate was not applied at {target}")
        for step in steps[1:]:
            options = step["options"]
            fixed = next((o for o in options if o["action"]["op"] == "recall"), options[0])
            if step["action"] != fixed["action"]:
                raise ValueError(f"follow-up rule differs between branches at {target}")
        support = {"options": start["options"], **start["state"]}
        ids = [(o["instance_id"], o["instance_index"]) for o in outcomes]
        if decisions is None:
            decisions, window = support, ids
            if len(support["options"]) < 2:
                raise ValueError("first query has no choice; supply public initial memory from the training split")
        elif support != decisions or ids != window:
            raise ValueError(f"candidate branches have different initial inputs or scoring windows at {target}")
        if any(type(o["success"]) is not bool or type(o["reward"]) not in (int, float)
               or not math.isfinite(o["reward"]) for o in outcomes):
            raise ValueError(f"invalid official outcomes at {target}")
        scores.append({"reward": fmean(o["reward"] for o in outcomes),
                       "native_success": all(o["success"] for o in outcomes)})
        evidence.append({"rollout_sha256": sha256(target / "rollout.json"), "trace_sha256": sha256(target / "trace.json.gz")})
        index += 1

    scenario = digest([bench["task"], bench.get("task_params", {}), seed, window])
    state = {**decisions, "scenario_id": scenario, "state_id": digest([scenario, decisions]), "scores": scores}
    validate_scored_state(state)
    return state, evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="train/dev experiment JSON")
    parser.add_argument("--memory", required=True, type=Path, help="public initial student memory JSON")
    parser.add_argument("--source-ref", required=True, help="source of training scenarios and public seed memory")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    from .harnesses.run import exclusive_lock, save_json
    from .sources import verify_source

    cfg = json.loads(args.config.read_text())
    if cfg.get("split") not in ("train", "dev") or not args.source_ref.strip():
        parser.error("expected a train/dev split with a source reference")
    memory = json.loads(args.memory.read_text())
    validate_memory(memory)
    if not memory:
        parser.error("provide at least one public memory entry for first-query candidate scoring")
    validate_development_split(cfg)
    seeds = cfg["environment_seeds"]
    bench = {**cfg["benchmark"], "split": cfg["split"], "initial_memory": memory}
    manifest = {"benchmark": bench, "environment_seeds": seeds, "split": cfg["split"],
                "source_ref": args.source_ref, "protocol": PROTOCOL, "bench_commit": verify_source("clbench")[1]}
    plan_id = digest(manifest)
    root = args.output.resolve()
    with exclusive_lock(root / ".lock"):
        plan = root / "manifest.json"
        if plan.exists() and json.loads(plan.read_text()) != manifest:
            raise ValueError(f"output already belongs to another experiment: {root}")
        save_json(plan, manifest)
        states = []
        for seed in seeds:
            folder = root / f"seed-{seed}"
            sealed = folder / "state.json"
            if not sealed.exists():
                attempt = folder / f"attempt-{uuid4().hex}"
                state, evidence = score_state(bench, seed, attempt, plan_id)
                save_json(sealed, {"plan_sha256": plan_id, "raw": attempt.name, "state": state, "evidence": evidence})
            else:
                record = json.loads(sealed.read_text())
                name = record["raw"]
                if Path(name).name != name or not name.startswith("attempt-") or record["plan_sha256"] != plan_id:
                    raise ValueError(f"invalid frozen branch reference for seed {seed}")
                state, evidence = score_state(bench, seed, folder / name, plan_id, recorded=True)
                if state != record["state"] or evidence != record["evidence"]:
                    raise ValueError(f"official outcomes or frozen branch evidence changed for seed {seed}")
            states.append(state)
        spec = {"dataset_id": f"{bench['harness']}-{bench['task']}-{plan_id[:12]}",
                "split": cfg["split"], "score_protocol": PROTOCOL, "authorization_ref": args.source_ref,
                "training_permission": "weight_training", "plan_sha256": plan_id, "states": states}
        path = root / "scored.json"
        if path.exists():
            if json.loads(path.read_text()) != spec:
                raise ValueError("existing spec differs from the sealed states")
        else:
            write_spec(path, spec)
        print(path)


if __name__ == "__main__":
    main()
