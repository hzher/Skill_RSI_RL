"""Build collector states from public observations at a verified replay boundary."""

from .contracts import apply_action, bound_state, digest, propose_options


def _observation(query, previous, history, memory):
    queued = [previous["content"]] if previous is not None else []
    feedback = query.get("feedback")
    if feedback is not None and feedback["content"] not in queued:
        queued.append(feedback["content"])
    return bound_state(query["prompt"], history + queued, memory)


def observation_from_prefix(prefix, next_query, *, memory=None):
    """Recreate the live policy's bounded history from its public inputs only."""
    memory = {} if memory is None else memory
    previous = None
    history = []
    for row in prefix["interactions"]:
        _, history, _ = _observation(row["query"], previous, history, memory)
        previous = row["observation"]
    query, feedback, memory = _observation(next_query, previous, history, memory)
    return {"query": query, "feedback": feedback, "memory": memory}


def initial_memory(source, cut):
    """Optionally seed memory with deterministic writes during the archived prefix."""
    interactions = source.trace["interactions"]
    if type(cut) is not int or not 0 < cut < len(interactions):
        raise ValueError("Memory seeding requires a nonempty prefix and a following query")
    previous = None
    history = []
    memory = {}
    for row in interactions[:cut]:
        query, history, memory = _observation(row["query"], previous, history, memory)
        actions = (option["action"] for option in propose_options(query, history, memory))
        action = next((action for action in actions if action["op"] == "write"), None)
        if action is not None:
            memory, _ = apply_action(action, memory)
        previous = row["observation"]
    return memory


def build_state(source, start):
    """Freeze the shared action support and seed-family identity for scoring."""
    if start.receipt["trace_sha256"] != source.receipt["trace_sha256"]:
        raise ValueError("Branch state and source trace differ")
    if not isinstance(start.feedback, (list, tuple)):
        raise ValueError("Branch feedback must be a sequence of observations")
    query, feedback, memory = bound_state(start.query, list(start.feedback), start.memory)
    observation = {"query": query, "feedback": feedback, "memory": memory}
    if digest(observation) != start.receipt["state_sha256"]:
        raise ValueError("Branch observation differs from the verified state receipt")
    options = propose_options(query, feedback, memory)
    scenario_id = digest([
        "official-environment-lineage-v1", source.receipt["task"],
        source.receipt["benchmark_resources_sha256"], source.attempt.get("seed"),
        source.attempt.get("stage_seeds"),
    ])
    state_id = digest([
        "verified-context-branch-v1", start.receipt["boundary_sha256"], start.receipt["state_sha256"],
        start.receipt["score_window_sha256"], start.target_outcomes, options,
    ])
    return {
        "state_id": state_id, "scenario_id": scenario_id, **observation, "options": options,
    }
