"""Assemble already authorized, scored replay specifications into one split."""

from argparse import ArgumentParser
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile

from .contracts import build_messages, digest, propose_options


SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def _text(value):
    return isinstance(value, str) and bool(value.strip())


def _sha256(value):
    return isinstance(value, str) and SHA256.fullmatch(value) is not None


def _source_reference(sources):
    return "source_specs:sha256:" + digest(sources)


def _validate_sources(spec):
    sources = spec["source_specs"]
    if not isinstance(sources, list) or not sources or spec["authorization_ref"] != _source_reference(sources):
        raise ValueError("Assembled replay authorization reference must bind the source specifications")
    offset = 0
    fingerprints = set()
    for source in sources:
        if (not isinstance(source, dict) or not _text(source.get("dataset_id")) or
                not _text(source.get("authorization_ref")) or not _sha256(source.get("spec_sha256")) or
                not _sha256(source.get("plan_sha256")) or type(source.get("state_offset")) is not int or
                source["state_offset"] != offset or type(source.get("state_count")) is not int or
                source["state_count"] < 1 or source["spec_sha256"] in fingerprints):
            raise ValueError("Assembled replay has invalid or duplicate source provenance")
        fingerprints.add(source["spec_sha256"])
        offset += source["state_count"]
    if offset != len(spec["states"]):
        raise ValueError("Assembled replay source provenance does not cover every state")


def load_spec(path, split=None):
    raw = Path(path).read_bytes()
    spec = json.loads(raw)
    if (not isinstance(spec, dict) or spec.get("training_permission") != "weight_training" or
            not _text(spec.get("authorization_ref"))):
        raise ValueError("Replay spec requires weight_training permission and an authorization_ref")
    if spec.get("split") not in ("train", "dev") or (split is not None and spec["split"] != split):
        raise ValueError("Replay spec split must be train or dev and match --split when supplied")
    if (not _text(spec.get("dataset_id")) or not _text(spec.get("score_protocol")) or
            not isinstance(spec.get("states"), list) or not spec["states"]):
        raise ValueError("Replay spec requires dataset_id, score_protocol, and a nonempty states list")
    if "source_specs" in spec:
        _validate_sources(spec)
    return spec, hashlib.sha256(raw).hexdigest()


def _valid_score(score):
    if (not isinstance(score, dict) or type(score.get("reward")) not in (int, float) or
            type(score.get("native_success")) is not bool):
        return False
    try:
        return math.isfinite(score["reward"])
    except OverflowError:
        return False


def validate_scored_state(state):
    """Validate the frozen option support and return the exact learner messages."""
    if not isinstance(state, dict) or any(not _text(state.get(key)) for key in ("state_id", "scenario_id")):
        raise ValueError("Scored state requires state_id and scenario_id")
    options, scores = state["options"], state["scores"]
    query, feedback, memory = state["query"], state["feedback"], state["memory"]
    if options != propose_options(query, feedback, memory):
        raise ValueError("Scored options must match the shared proposal policy and ordering")
    messages = build_messages(query, feedback, memory, options)
    if len(options) < 2 or not isinstance(scores, list) or len(scores) != len(options):
        raise ValueError("Each state needs multiple options and one score per option")
    if not all(map(_valid_score, scores)):
        raise ValueError("Each option score needs a finite reward and a boolean native_success")
    return messages


def write_spec(path, spec):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".scored-replay-")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(spec, stream, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
        os.link(temporary, path)
    finally:
        os.unlink(temporary)


def assemble_specs(paths, dataset_id, output):
    if not _text(dataset_id):
        raise ValueError("Assembled replay requires a dataset_id")
    inputs = {}
    identity = None
    for path in paths:
        path = Path(path)
        spec, sha = load_spec(path)
        if "source_specs" in spec or not _sha256(spec.get("plan_sha256")):
            raise ValueError(f"Assembly requires a per-branch spec with plan_sha256: {path.name}")
        if sha in inputs:
            raise ValueError(f"Duplicate source replay spec: {path.name}")
        current = spec["split"], spec["score_protocol"]
        if identity is not None and identity != current:
            raise ValueError(f"Source specs must have the same split and score_protocol: {path.name}")
        identity = current
        inputs[sha] = path, spec
    if not inputs:
        raise ValueError("Provide at least one scored replay spec")

    states, source_specs = [], []
    state_ids, observations = set(), set()
    for sha, (path, spec) in sorted(inputs.items()):
        source_specs.append({
            "dataset_id": spec["dataset_id"], "authorization_ref": spec["authorization_ref"],
            "spec_sha256": sha, "plan_sha256": spec["plan_sha256"],
            "state_offset": len(states), "state_count": len(spec["states"]),
        })
        for index, state in enumerate(spec["states"]):
            try:
                observation = digest(validate_scored_state(state))
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"{path.name}, state {index}: {exc}") from exc
            state_id = state["state_id"]
            if state_id in state_ids:
                raise ValueError(f"Duplicate state ID in {path.name}, state {index}")
            if observation in observations:
                raise ValueError(f"Duplicate canonical observation in {path.name}, state {index}")
            state_ids.add(state_id)
            observations.add(observation)
            states.append(state)

    split, protocol = identity
    output = Path(output)
    write_spec(output, {"dataset_id": dataset_id, "split": split, "score_protocol": protocol,
                         "training_permission": "weight_training", "authorization_ref": _source_reference(source_specs),
                         "source_specs": source_specs, "states": states})
    return {"output": str(output), "split": split, "score_protocol": protocol,
            "sources": len(source_specs), "states": len(states)}


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("specs", nargs="+", type=Path)
    args = parser.parse_args()
    try:
        result = assemble_specs(args.specs, args.dataset_id, args.output)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, f"{type(exc).__name__}: {exc}\n")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
