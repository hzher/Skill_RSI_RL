"""Exercise real archive recovery and official scoring with recorded task responses."""

import argparse
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import tempfile

from .branch_data import initial_memory
from .branch_score import HARNESS, TASKS
from .contracts import apply_action, digest, propose_options


def verify_window(source, cut, instances, max_queries, *, all_actions=False):
    from .harnesses.dsh_system import DSHSystem
    from .harnesses.poker_resume import _observation_payload, _query_payload
    from .harnesses.replay_branch import execute_branch, prepare_branch
    from src.interface import Response

    start = prepare_branch(source, cut, memory=initial_memory(source, cut),
                           score_instances=instances, max_queries=max_queries)
    offset = len(start.prefix["instance_outcomes"])
    expected = [{key: row[key] for key in ("instance_id", "instance_index", "reward")}
                | {"native_success": row["success"]}
                for row in source.trace["instance_outcomes"][offset:offset + instances]]
    recorded = source.trace["interactions"][cut:cut + max_queries]

    class RecordedSystem(DSHSystem):
        def _start(self):
            raise AssertionError("Archive verification must never start a provider session")

        def respond(self, query):
            self.archived = next(continuation)
            saved_query = self.archived["query"]
            if _query_payload(query, saved_query, None, None) != saved_query:
                raise ValueError("Official continuation query differs from the archive")
            response = self.archived["response"]
            action = query.response_schema.model_validate(response["action"])
            if response["action_type"] != "structured" or action.model_dump() != response["action"]:
                raise ValueError("Official structured action differs from the archive")
            feedback = list(self._feedback)
            if query.feedback and query.feedback.content not in feedback:
                feedback.append(query.feedback.content)
            self._feedback.clear()
            memo, receipt = self._memory_policy.decide(query.prompt, feedback, self.timeout)
            memos.append(memo)
            return Response(action=action, metadata={**deepcopy(response["metadata"]), "memory_policy": receipt})

        def observe(self, observation, next_query=None):
            if _observation_payload(observation) != self.archived["observation"]:
                raise ValueError("Official continuation feedback differs from the archive")
            super().observe(observation, next_query)

    options = (propose_options(start.query, start.feedback, start.memory) if all_actions
               else [{"label": "A", "action": {"op": "noop"}}])
    baseline_memos = []
    tested = []
    with tempfile.TemporaryDirectory(prefix="skill-rsi-history-verify-") as directory:
        for option in options:
            label, action = option["label"], option["action"]
            continuation = iter(recorded)
            memos = []
            result = execute_branch(start, action=action, system_params={},
                                    runtime_root=directory, system_factory=RecordedSystem)
            if result["status"] != "scored" or result["outcomes"] != expected:
                detail = result.get("error") or result.get("reason") or result["status"]
                raise ValueError(f"Archive reward window differs at cut {cut}, candidate {label}: {detail}")
            memory, first_memo = apply_action(action, start.memory)
            after = digest(memory)
            receipts = result["policy_receipts"]
            first = receipts[0]
            if (first["phase"] != "candidate" or first["op"] != action["op"]
                    or first["action_sha256"] != digest(action) or first["input_sha256"] != start.receipt["state_sha256"]
                    or first["memory_before_sha256"] != digest(start.memory)
                    or first["memory_after_sha256"] != after or memos[0] != first_memo):
                raise ValueError(f"First-action receipt differs from the planned candidate {label}")
            for index, (memo, receipt) in enumerate(zip(memos, receipts, strict=True)):
                if (receipt["status"] != "applied" or receipt["injected"] != bool(memo)
                        or receipt["memo_sha256"] != (sha256(memo.encode()).hexdigest() if memo else None)):
                    raise ValueError(f"Memo differs from its receipt at candidate {label}, query {index + 1}")
                if index and (receipt["phase"] != "continuation" or receipt["op"] not in {"recall", "noop"}
                              or receipt["memory_before_sha256"] != after or receipt["memory_after_sha256"] != after):
                    raise ValueError(f"Continuation changed memory for candidate {label}")

            if action["op"] == "noop":
                baseline_memos = memos
            detail = {"label": label, "op": action["op"], "first_query_has_memo": bool(first_memo)}
            if action["op"] == "write":
                _, written_memo = apply_action({"op": "recall", "keys": [action["key"]]}, memory)
                detail["write_readback_queries"] = [
                    index for index, (memo, baseline) in enumerate(zip(memos, baseline_memos, strict=True), start=1)
                    if index > 1 and memo == written_memo and memo != baseline
                ]
            tested.append(detail)
    report = {"cut": cut, "queries_replayed": len(baseline_memos), "expected_outcomes": len(expected),
              "native_memory_files": start.receipt["native"]["memory_file_count"],
              "boundary_sha256": start.receipt["boundary_sha256"],
              "score_window_sha256": start.receipt["score_window_sha256"]}
    if all_actions:
        report["actions"] = tested
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--task", choices=TASKS, default="exploitable_poker")
    parser.add_argument("--run", type=int, default=0)
    parser.add_argument("--cut", type=int, action="append")
    parser.add_argument("--score-instances", type=int, default=2)
    parser.add_argument("--max-queries", type=int, default=24)
    parser.add_argument("--all-actions", action="store_true", help="verify each candidate using fixed recorded responses")
    args = parser.parse_args()
    try:
        from .harnesses.historical_replay import load_completed_source

        source = load_completed_source(args.archive, args.task, args.run)
        windows = [verify_window(source, cut, args.score_instances, args.max_queries, all_actions=args.all_actions)
                   for cut in sorted(set(args.cut or (44, 84)))]
        print(json.dumps({"mode": "replay_recorded_responses", "task": args.task, "rollout": args.run,
                          "provider_calls": 0, "windows": windows}, indent=2))
    except (ImportError, KeyError, OSError, RuntimeError, StopIteration, ValueError) as error:
        parser.exit(2, f"{type(error).__name__}: {error}\n")


if __name__ == "__main__":
    main()
