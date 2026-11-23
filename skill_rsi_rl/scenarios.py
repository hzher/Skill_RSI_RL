"""Independent stateful fixtures with a deterministic downstream scorer."""

import random
import re

from .contracts import apply_action, digest, propose_options


FAMILIES = ("recall", "compact", "write", "noop")
EVIDENCE_BUDGET = 420
FACT = re.compile(r"\brecord ([a-z0-9_]+) = ([A-Z0-9]+)\b")
PADDING = (
    "The prior inventory ledger was archived after its routine verification. ",
    "The previous desk reconciled the checklist and left these reference notes. ",
    "The service log retained standard observations from the previous handover. ",
)


def _score_candidates(family, query, memory, expected, options):
    scores = []
    for option in options:
        updated, memo = apply_action(option["action"], memory)
        if family == "write":
            # The next request starts with saved memory and uses fixed one-entry recall.
            next_query = f"Retrieve the revised record for {next(iter(expected))}."
            recall = propose_options(next_query, [], updated)[1]["action"]
            _, next_memo = apply_action(recall, updated)
            visible = next_query + "\n" + next_memo[:EVIDENCE_BUDGET]
        else:
            visible = query + "\n" + memo[:EVIDENCE_BUDGET]
        observations = dict(FACT.findall(visible))
        correct = sum(observations.get(key) == answer for key, answer in expected.items())
        scores.append({"reward": correct / len(expected), "native_success": correct == len(expected)})
    return scores


def synthetic_state(split, seed, index):
    """Generate one scenario and independently score every allowed intervention."""
    scenario_id = digest(["context-memory-fixtures-v1", split, seed, index])
    rng = random.Random(int(scenario_id[:16], 16))
    family = FAMILIES[index % len(FAMILIES)]
    names = rng.sample(("birch", "cedar", "elm", "fir", "hazel", "maple", "oak", "pine"), 3)
    lead, second, distractor = [f"{name}_{rng.randrange(10, 99)}" for name in names]
    values = [f"R{value}" for value in rng.sample(range(1000, 10000), 4)]
    lead_value, second_value, distractor_value, stale_value = values
    lead_fact = f"record {lead} = {lead_value}"
    second_fact = f"record {second} = {second_value}"
    distractor_fact = f"record {distractor} = {distractor_value}"
    stale_fact = f"record {lead} = {stale_value}"
    long_note = rng.choice(PADDING) * 8
    memory = {lead: lead_fact, second: second_fact, distractor: distractor_fact}
    expected = {lead: lead_value}
    feedback = []
    if family == "recall":
        query = f"Retrieve the current record identifier for {lead}."
        memory[lead] = long_note[:270] + "\n" + lead_fact
    elif family == "compact":
        query = f"Return both approved record identifiers for {lead} and {second}."
        memory[lead] = lead_fact + "\n" + long_note
        memory[second] = second_fact + "\n" + long_note
        expected[second] = second_value
    elif family == "write":
        query = f"Prepare the desk for the next shift to look up the revised record for {lead}."
        memory[lead] = stale_fact
        memory["recent_feedback"] = stale_fact
        feedback = [f"The revised {lead} record passed verification. {lead_fact}",
                    f"Inventory update received. {distractor_fact}"]
    else:
        query = f"The latest signed update for {lead} is already here. {lead_fact}\nReturn this identifier."
        memory[lead] = stale_fact
    options = propose_options(query, feedback, memory)
    return {
        "state_id": digest([scenario_id, query, feedback, memory, options]),
        "scenario_id": scenario_id,
        "family": family,
        "query": query,
        "feedback": feedback,
        "memory": memory,
        "options": options,
        "scores": _score_candidates(family, query, memory, expected, options),
    }
