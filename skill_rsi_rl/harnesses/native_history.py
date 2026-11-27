"""Replay native history boundaries using only the frozen original episode plan."""
from copy import deepcopy

from ..contracts import apply_action, bound_state, digest
from .native_spec import validate_instances


class History:
    def __init__(self, initial_memory):
        self.initial = dict(initial_memory)
        self.memory, self.feedback, self.anchors = dict(initial_memory), [], {}

    def begin(self, episode):
        if not episode["persistence_allowed"]:
            self.memory, self.feedback = {}, []
        elif episode["history_mode"] == "fresh":
            self.memory, self.feedback = dict(self.initial), []
        elif episode["history_mode"] == "from_anchor":
            self.memory, self.feedback = deepcopy(self.anchors[episode["history_load_anchor"]])
        return dict(memory=deepcopy(self.memory), feedback=list(self.feedback))

    def end(self, episode):
        if episode["history_save_anchor"]:
            if not episode["persistence_allowed"]:
                raise ValueError("no-persistence control cannot save an anchor")
            self.anchors[episode["history_save_anchor"]] = deepcopy((self.memory, self.feedback))


def validate_history(row):
    """Return expected memory before each decision; reject fabricated resets/anchors."""
    validate_instances(row)
    config = row["benchmark_config"]
    history = History(config.get("initial_memory", {}))
    interactions = row["native_interactions"]
    decisions = row["decisions"]
    if len(interactions) != len(decisions) or not interactions:
        raise ValueError("native query/decision coverage differs")
    cursor, expected = 0, {}
    records = row["history"]
    if len(records) != len(config["native"]["plan"]):
        raise ValueError("native history boundary missing")
    for episode, boundary in zip(config["native"]["plan"], records, strict=True):
        state = history.begin(episode)
        if boundary["episode"] != episode or boundary["student_before"] != state:
            raise ValueError("native student reset or anchor differs from original history plan")
        count = 0
        while cursor < len(decisions) and decisions[cursor]["instance_index"] == episode["instance_index"]:
            step, interaction = decisions[cursor], interactions[cursor]
            public = interaction["public"]
            if (interaction["query_id"] != step["query_id"] or interaction["receipt"] != step["receipt"]
                    or interaction["instance_id"] != episode["label"]
                    or interaction["public_sha256"] != digest(public)):
                raise ValueError("native interaction is detached from its student decision")
            if not episode["persistence_allowed"]:
                history.memory, history.feedback = {}, []
            feedback = public_feedback(public) if episode["persistence_allowed"] else []
            query, recent, memory = bound_state(public_prompt(public), history.feedback + feedback, history.memory)
            if step["state"] != dict(query=query, feedback=recent, memory=memory):
                raise ValueError("student input contains feedback or memory absent from native execution")
            expected[step["query_id"]] = dict(memory)
            history.memory, _ = apply_action(step["action"], memory)
            history.feedback = recent
            cursor += 1
            count += 1
        if count == 0:
            raise ValueError("native episode has no executor query")
        if boundary["student_after"] != dict(memory=history.memory, feedback=history.feedback):
            raise ValueError("native history anchor does not match applied student actions")
        history.end(episode)
    if cursor != len(decisions):
        raise ValueError("native decisions are out of schedule")
    return expected


def public_prompt(public):
    from ..contracts import canonical_json
    return "Execute the benchmark task using only its available tools.\n" + canonical_json(public)


def public_feedback(public):
    from ..contracts import canonical_json
    return [canonical_json(x) for x in public.get("tool_results", [])]
