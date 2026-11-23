from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from skill_rsi_rl.collect import _load_replay_spec, _prepare_group
from skill_rsi_rl.contracts import digest, propose_options
from skill_rsi_rl.replay_spec import assemble_specs


def source_spec(name, *, split="train"):
    query = f"Find record {name}"
    feedback = [f"Last known location for {name}"]
    options = propose_options(query, feedback, {})
    return {
        "dataset_id": name, "split": split, "training_permission": "weight_training",
        "authorization_ref": f"approval/{name}", "score_protocol": "frozen-window",
        "plan_sha256": digest([name, "plan"]),
        "states": [{"state_id": name, "scenario_id": "shared-environment", "query": query, "feedback": feedback,
                    "memory": {}, "options": options,
                    "scores": [{"reward": index, "native_success": bool(index)}
                               for index in range(len(options))]}],
    }


def write_source(path, spec):
    raw = (json.dumps(spec, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    path.write_bytes(raw)
    return sha256(raw).hexdigest()


class ReplaySpecTests(unittest.TestCase):
    def test_assembled_sources_are_deterministic_and_collector_compatible(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            first, second = root / "first.json", root / "second.json"
            inputs = {write_source(path, spec): (path, spec)
                      for path, spec in ((first, source_spec("a")), (second, source_spec("b")))}
            merged, reordered = root / "combined.json", root / "reordered.json"
            report = assemble_specs([first, second], "both", merged)
            self.assertEqual((report["sources"], report["states"]), (2, 2))
            assemble_specs([second, first], "both", reordered)
            self.assertEqual(merged.read_bytes(), reordered.read_bytes())
            states, provenance = _load_replay_spec(merged, "train")
            sources = provenance["source_specs"]
            self.assertEqual(provenance["source_sha256"], sha256(merged.read_bytes()).hexdigest())
            self.assertEqual(provenance["authorization_ref"], "source_specs:sha256:" + digest(sources))
            for index, (sha, (path, original)) in enumerate(sorted(inputs.items())):
                self.assertEqual(sources[index], {
                    "dataset_id": original["dataset_id"], "spec_sha256": sha,
                    "plan_sha256": original["plan_sha256"], "authorization_ref": original["authorization_ref"],
                    "state_offset": index, "state_count": 1,
                })
                self.assertEqual(states[index], original["states"][0])
                _prepare_group(states[index], index, 17)
                self.assertEqual(sha256(path.read_bytes()).hexdigest(), sha)
                self.assertEqual(_load_replay_spec(path, "train")[1]["plan_sha256"], original["plan_sha256"])
            with self.assertRaises(FileExistsError):
                assemble_specs([first, second], "another", merged)
            self.assertEqual(merged.read_bytes(), reordered.read_bytes())

    def test_rejects_mixed_or_unproven_sources_and_invalid_scores(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            first, second, output = root / "first.json", root / "second.json", root / "combined.json"
            write_source(first, source_spec("a"))
            valid = source_spec("b")
            cases = [
                ({**valid, "split": "dev"}, "same split and score_protocol"),
                ({**valid, "score_protocol": "different"}, "same split and score_protocol"),
                ({**valid, "training_permission": "evaluation_only"}, "weight_training"),
                ({**valid, "plan_sha256": None}, "per-branch spec with plan_sha256"),
            ]
            bad_score = deepcopy(valid)
            bad_score["states"][0]["scores"][0]["reward"] = float("nan")
            cases.append((bad_score, "finite reward"))
            for spec, error in cases:
                with self.subTest(error=error):
                    write_source(second, spec)
                    with self.assertRaisesRegex(ValueError, error):
                        assemble_specs([first, second], "both", output)
                    self.assertFalse(output.exists())

    def test_rejects_duplicate_id_and_relabelled_observation(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            first, second, output = root / "first.json", root / "second.json", root / "combined.json"
            source = source_spec("a")
            write_source(first, source)
            other = source_spec("b")
            other["states"][0]["state_id"] = "a"
            write_source(second, other)
            with self.assertRaisesRegex(ValueError, "Duplicate state ID"):
                assemble_specs([first, second], "both", output)
            other["states"] = [{**source["states"][0], "state_id": "new-id", "scenario_id": "new-scenario"}]
            write_source(second, other)
            with self.assertRaisesRegex(ValueError, "Duplicate canonical observation"):
                assemble_specs([first, second], "both", output)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
