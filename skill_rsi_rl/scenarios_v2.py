"""Counterfactual decision states with a shared two-query evidence scorer."""

import random
import re

from .contracts import apply_action, digest, propose_options


SUITE_ID = "context-memory-contrasts-v2"
SCORE_PROTOCOL = "bounded_recent_context_and_delayed_lookup_v2"
FAMILIES = ("noop", "recall", "compact", "write")
STATES_PER_SCENARIO = len(FAMILIES)
RECORD = re.compile(
    r"\brecord ([a-z0-9_]+) = (R\d+)\b|\[([a-z0-9_]+)\] -> (R\d+)\b",
)
TRAIN_NOTES = (
    "Prior rounds of inventory reconciliation concluded without any disputed items. ",
    "Routine temperature observations were archived in chronological sequence. ",
)
DEV_NOTES = (
    "无异常的批次已登记，备用清单经盘点后移交保管。记录顺序按时间排列。 ",
    "Inventaire périodique archivé. Valeurs ordinaires conservées hors cycle quotidien. ",
)


def _records(context):
    return {left or bracket: plain_value or bracket_value
            for left, plain_value, bracket, bracket_value in RECORD.findall(context)}


def _scores(query, feedback, memory, options, required, later_query, later_key, budget):
    # The current executor retains the tail of the previous observations.
    prefix = "\n".join(feedback)[-budget:] + "\n" + query
    scores = []
    for option in options:
        updated, memo = apply_action(option["action"], memory)
        current = _records(prefix + "\n" + memo[:budget])
        following = propose_options(later_query, [], updated)
        next_action = next((row["action"] for row in following if row["action"]["op"] == "recall"), {"op": "noop"})
        _, next_memo = apply_action(next_action, updated)
        future = _records(next_memo[:budget])
        current_hits = sum(current.get(key) == value for key, value in required.items())
        later_hit = future.get(later_key) == required[later_key]
        count = len(required) + 1
        # At least one more verified fact always beats any possible cost difference.
        cost = 0.03 * min(len(memo) / budget, 1) + 0.02 * (updated != memory)
        scores.append({"reward": (current_hits + later_hit) / count - cost,
                       "native_success": current_hits == len(required) and later_hit})
    return scores


def synthetic_state(split, seed, index):
    """Generate one of four states with an identical query and different evidence."""
    if split not in ("train", "dev") or type(index) is not int or index < 0:
        raise ValueError("Invalid split or index for counterfactual scenarios")
    scenario_index, sibling = divmod(index, STATES_PER_SCENARIO)
    scenario_id = digest([SUITE_ID, split, seed, scenario_index])
    rng = random.Random(int(scenario_id[:16], 16))
    # The output labels, visible query, fact values, and budget are fixed within a contrast.
    order = list(FAMILIES)
    rng.shuffle(order)
    layout = order[sibling]
    a, b, distractor = (f"{role}_{rng.getrandbits(36):09x}" for role in ("item", "zone", "aux"))
    current_a, current_b, stale_a, stale_b = (f"R{value}" for value in rng.sample(range(10000, 99999), 4))
    required = {a: current_a, b: current_b}
    if split == "train":
        query = rng.choice((
            f"Handover ledger audit: verify both current records for {a} and {b}. A follow-up will look up {b} again.",
            f"For the handover ledger audit, confirm {a} and {b} now and preserve evidence relevant to {b}.",
        ))
        notes = rng.choice(TRAIN_NOTES)
        budget = rng.choice((366, 402, 438))
        fact = "record {} = {}".format
    else:
        query = rng.choice((
            f"复核 handover ledger audit 中 {a} 和 {b} 的现行编号。稍后单独查 {b}。",
            f"Relevé handover ledger audit — vérifiez les identifiants de {a} et {b} ; {b} sera demandé à nouveau.",
        ))
        notes = rng.choice(DEV_NOTES)
        budget = rng.choice((460, 490, 520))
        fact = "[{}] -> {}".format
    query += f" [memo capacity {budget} chars]"
    future_query = f"Look up {b} in the handover ledger."
    records = {key: fact(key, value) for key, value in required.items()}
    old_a, old_b = fact(a, stale_a), fact(b, stale_b)
    late_padding = (notes * 20)[:rng.randint(211, 221)]
    pressure_padding = (notes * 36)[:budget + 110]
    history = f"Handover ledger preliminary audit for {a} and {b}: {old_a}\n{old_b}"
    missing = rng.choice((a, b))
    visible = b if missing == a else a
    target_first = rng.choice((True, False))
    fresh_first = rng.choice((True, False))
    first = b
    if layout == "recall":
        first = missing if target_first else visible
    elif layout == "compact":
        first = visible
    headers = {a: "audit handover ledger\n" if first == a else "Archive buffer notes\n",
               b: "handover ledger\n"}
    memory = {
        **{key: headers[key] + value for key, value in records.items()},
        distractor: "Previously archived logistics notes. " + notes,
    }
    if split == "dev":
        memory[distractor + "_spare"] = notes + fact(distractor, stale_a)

    if layout == "recall":
        memory[missing] = headers[missing] + late_padding + "\n" + records[missing]
        feedback = [history, f"Signed update: {records[visible]}"]
    elif layout == "compact":
        memory[visible] += "\n" + pressure_padding
        feedback = [history, f"Signed update: {records[visible]}"]
    else:
        if layout == "write":
            memory[b] = headers[b] + old_b
        else:
            memory[distractor] = old_b
        verified = f"Confirmed handover ledger: {records[a]}\n{records[b]}"
        if fresh_first:
            feedback = ["Audit " + verified, f"Supplemental desk tally for {distractor}."]
        else:
            feedback = [history, verified]

    options = propose_options(query, feedback, memory)
    scores = _scores(query, feedback, memory, options, required, future_query, b, budget)
    return {
        "state_id": digest([scenario_id, query, feedback, memory, options]),
        "scenario_id": scenario_id,
        "contrast_id": digest([scenario_id, query]),
        "family": layout,
        "query": query,
        "feedback": feedback,
        "memory": memory,
        "options": options,
        "scores": scores,
    }
