"""Generate state contrasts with varied evidence domains and success-weighted rewards."""

import random

from .contracts import digest, propose_options
from .scenarios_v2 import FAMILIES, STATES_PER_SCENARIO, _scores


SUITE_ID = "context-memory-contrasts-v3"
SCORE_PROTOCOL = "bounded_context_and_delayed_lookup_success_weighted_v3"

TRAIN_DOMAINS = (
    ("warehouse inventory audit", "Preserve the current handover readings for {a} and {b}.",
     "A previous shipment was signed off. The routine inspections were archived in delivery order. "),
    ("software migration record", "Compare the current migration markers against {a} and {b}.",
     "Routine compatibility checks completed earlier. Build caches and test artifacts were archived. "),
    ("sensor calibration report", "Verify the channel readings for {a} and {b} in today's calibration.",
     "The station finished a routine sweep. The control equipment was returned to its storage location. "),
    ("forecast reconciliation log", "Confirm today's reference values for {a} and {b} before publishing.",
     "The previous forecast window closed. Regional distribution files were reviewed and filed. "),
)
DEV_DOMAINS = (
    ("dispatch routing review", "复核调度参考 {a} 和 {b} 的当前值。",
     "已完成的运输交接单按日期归档。日常核验未发现异常，其余条目稍后处理。"),
    ("billing ledger review", "Validez les références actuelles de {a} et {b} dans les justificatifs.",
     "Les pièces du cycle précédent ont été archivées par date. Les contrôles ordinaires sont terminés. "),
)


def synthetic_state(split, seed, index):
    if split not in ("train", "dev") or type(index) is not int or index < 0:
        raise ValueError("Invalid split or index for counterfactual scenarios")
    scenario_index, sibling = divmod(index, STATES_PER_SCENARIO)
    scenario_id = digest([SUITE_ID, split, seed, scenario_index])
    rng = random.Random(int(scenario_id[:16], 16))
    domain, prompt, notes = rng.choice(TRAIN_DOMAINS if split == "train" else DEV_DOMAINS)
    order = list(FAMILIES)
    rng.shuffle(order)
    family = order[sibling]
    a, b = (f"{role}_{rng.getrandbits(32):08x}" for role in ("source", "target"))
    active_a, active_b, stale_a, stale_b = (f"R{value}" for value in rng.sample(range(10000, 99999), 4))
    required = {a: active_a, b: active_b}
    budget = rng.randint(402, 470) if split == "train" else rng.randint(460, 550)
    query = (
        f"{domain}. {prompt.format(a=a, b=b)} Retain evidence for a later lookup of {b}."
        f" [memo capacity {budget} chars]"
    )
    later_query = f"{domain} follow-up lookup: {b}"
    formatters = ("record {} = {}".format, "[{}] -> {}".format)
    fact = rng.choice(formatters) if split == "train" else formatters[1]
    records = {key: fact(key, value) for key, value in required.items()}
    stale_b_record = fact(b, stale_b)
    history = f"{domain} preliminary report: {fact(a, stale_a)}\n{stale_b_record}"
    missing = rng.choice((a, b))
    visible = b if missing == a else a
    recall_first = rng.choice((visible, missing))
    fresh_first = rng.choice((True, False))
    first = {"recall": recall_first, "compact": visible, "noop": b, "write": b}[family]
    headers = {key: f"{domain}{' evidence' if key == first else ''}\n" for key in required}
    memory = {key: headers[key] + value for key, value in records.items()}
    distraction_count = rng.randint(2, 4) if split == "train" else rng.randint(3, 5)
    distractors = [f"aux_{rng.getrandbits(32):08x}" for _ in range(distraction_count)]
    for key in distractors:
        memory[key] = f"Archived background: {fact(key, stale_a)}\n" + notes * rng.randint(1, 3)
    if family == "recall":
        # The missing fact is past compact's cutoff, but within a two-entry recall.
        padding = (notes * 12)[: rng.randint(217, 224)]
        memory[missing] = headers[missing] + padding + "\n" + records[missing]
        feedback = [history, f"Signed update: {records[visible]}"]
    elif family == "compact":
        # Recall spends the budget on padding in the first entry.
        memory[visible] += "\n" + (notes * 30)[: budget + 100]
        feedback = [history, f"Signed update: {records[visible]}"]
    else:
        if family == "write":
            memory[b] = headers[b] + stale_b_record
        recent = f"{domain} confirmed handover: {records[a]}\n{records[b]}"
        irrelevant = f"Archived desk tally for {rng.choice(distractors)}."
        feedback = [history, recent, irrelevant] if fresh_first else [history, irrelevant, recent]
    options = propose_options(query, feedback, memory)
    scores = _scores(query, feedback, memory, options, required, later_query, b, budget)
    for row in scores:
        # Partial evidence remains useful, but completing both queries dominates it.
        row["reward"] = 0.5 * (row["reward"] + row["native_success"])
    return {
        "state_id": digest([scenario_id, query, feedback, memory, options]),
        "scenario_id": scenario_id,
        "contrast_id": digest([scenario_id, query]),
        "family": family,
        "query": query,
        "feedback": feedback,
        "memory": memory,
        "options": options,
        "scores": scores,
    }
