"""Validate and attach a rollout's acknowledged native session prefix."""

from dataclasses import dataclass
import gzip
from hashlib import sha256
import json
from pathlib import Path
import re
import time
from types import MethodType

from .poker_resume import PREFIX_TASKS


_HISTORICAL_BRANCH_TASKS = {"exploitable_poker", "blind_spectrum_monitoring"}


@dataclass(frozen=True)
class _NativePrefix:
    session_id: str
    header: dict
    events: bytes
    usage: bytes
    memory: dict
    lineage: dict
    uses_absolute_paths: bool


def _require(condition, message):
    if not condition:
        raise ValueError(f"Cannot resume DSH native session: {message}")


def _lines(path):
    raw = path.read_bytes().splitlines(keepends=True)
    _require(all(line.endswith(b"\n") for line in raw), f"incomplete JSONL in {path.name}")
    return raw, [json.loads(line) for line in raw]


def _one(items, message):
    items = list(items)
    _require(len(items) == 1, message)
    return items[0]


def _contained(root, value):
    path = root / value
    _require(path.is_relative_to(root) and path.resolve() == path, "path is outside its scope or symlinked")
    return path


def _memory_name(value):
    _require(isinstance(value, str) and bool(value) and Path(value).parts
             and not Path(value).is_absolute() and ".." not in Path(value).parts, "invalid memory filename")
    return value


def _memory_object(directory, description):
    digest = description.get("sha256")
    _require(description.get("status") == "captured" and description.get("representation") == "file_bytes"
             and isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest), "missing captured memory bytes")
    obj = _contained(directory, f"capture/objects/{digest}").read_bytes()
    _require(len(obj) == description.get("bytes") and sha256(obj).hexdigest() == digest, "memory object hash differs")
    return obj


def _memory_at(directory, events, cutoff, seed, *, paths_at_cut=False):
    path = _contained(directory, "capture/memory_operations.jsonl")
    rows = _lines(path)[1] if path.is_file() else []
    captures = {}
    for row in rows:
        captures.setdefault(row.get("tool_call_id"), []).append(row)
    initial = {
        _memory_name(name): _memory_object(directory, desc)
        for name, desc in (seed.get("memory_files") or {}).items()
    }
    initial_cut = seed.get("session_terminal_seq", -1)
    _require(type(initial_cut) is int and initial_cut < cutoff, "resume seed is beyond the selected boundary")
    memory = initial.copy()
    at_cut = initial.copy()
    uses_absolute_paths = bool(seed.get("uses_absolute_paths"))
    allowed = {"clbench_submit", "read", "write", "edit"}
    unknown = {event["data"].get("callId") for event in events if event.get("type") == "tool/call"
               and event["seq"] > initial_cut and event["data"]["name"] not in allowed}
    rejected_calls = {}
    if unknown:
        patch = json.loads(_contained(directory, "bridge.patch.json").read_bytes())
        guard = _one((plugin["config"] for entry in patch for plugin in entry.get("insert", [])
                      if plugin["id"] == "clbench-submit"), "memory guard is missing")
        _require(guard.get("toolsMode") == "memory" and guard.get("runtimeId") == directory.name,
                 "source runtime has no benchmark memory guard")
        for event in events:
            if event.get("type") == "tool/result":
                result = event["data"]
                call_id = result.get("message", {}).get("toolCallId")
                if call_id in unknown:
                    rejected_calls.setdefault(call_id, []).append(result)
    for row in events:
        if row.get("type") != "tool/call" or row["seq"] <= initial_cut:
            continue
        call = row["data"]
        if call["name"] not in allowed:
            result = _one(rejected_calls.get(call.get("callId"), ()), "unobserved native tool effects")
            message = result["message"]
            _require(result.get("turn") == call.get("turn") and message.get("isError") is True
                     and message.get("content") == [{"type": "text", "text":
                                                     "Error: Use clbench_submit for benchmark environment actions"}]
                     and not captures.get(call.get("callId")), "unobserved native tool effects")
            continue
        if call["name"] == "clbench_submit":
            continue
        operations = captures.get(call.get("callId"), [])
        before = _one((item for item in operations if item["kind"] == "memory_before"),
                      "native memory call lacks its preimage")
        after = _one((item for item in operations if item["kind"] == "memory_after"),
                     "native memory call lacks its postimage")
        name = _memory_name(before["scope_path"])
        _require(after.get("scope_path") == name and after.get("operation_id") == before.get("operation_id")
                 and after.get("session_id") == before.get("session_id")
                 and before.get("tool_name") == after.get("tool_name") == call["name"]
                 and after.get("concurrent") is False, "memory operations are ambiguous or overlapped")
        if not paths_at_cut or row["seq"] <= cutoff:
            uses_absolute_paths |= Path(before["arguments"]["file_path"]).is_absolute()
        state = before["before"]
        expected = None if state.get("status") == "absent" else _memory_object(directory, state)
        _require(memory.get(name) == expected, "recorded memory transitions do not form a chain")
        state = after["after"]
        if state.get("status") == "absent":
            memory.pop(name, None)
        else:
            memory[name] = _memory_object(directory, state)
        if row["seq"] <= cutoff:
            if name in memory:
                at_cut[name] = memory[name]
            else:
                at_cut.pop(name, None)
    actual = {}
    workspace = _contained(directory, "workspace")
    for entry in workspace.rglob("*"):
        _require(not entry.is_symlink() and (entry.is_dir() or entry.is_file()),
                 "untracked workspace object")
        if entry.is_file():
            _require(entry.is_relative_to(workspace / "memory"), "untracked file outside memory")
            actual[entry.relative_to(workspace / "memory").as_posix()] = entry.read_bytes()
    _require(actual == memory, "final workspace does not match recorded memory transitions")
    return at_cut, uses_absolute_paths


def prepare_native_resume(batch, attempt, prefix):
    """Prove the native state of an interrupted or failed attempt for ordinary recovery."""
    _require(attempt.get("task") in PREFIX_TASKS and attempt.get("phase") == "run"
             and attempt.get("status") in ("failed", "interrupted"), "source is not a supported failed rollout")
    return _prepare_native_prefix(Path(batch).resolve(strict=True), attempt, prefix)


def prepare_native_branch(batch, attempt, prefix):
    """Read a verified nonterminal native state from a completed Poker or Spectrum trace."""
    from .run import verify_resume_ancestry

    batch = Path(batch).resolve(strict=True)
    task = attempt.get("task")
    _require(task in _HISTORICAL_BRANCH_TASKS and attempt.get("phase") == "run"
             and attempt.get("status") == "completed", "source is not a supported completed rollout")
    path, source, _ = _sealed_source(batch, attempt)
    _require(path.name == "trace.json.gz" and attempt == json.loads(_contained(batch, path.with_name("manifest.json")).read_bytes()),
             "source attempt is not the sealed completed manifest")
    identity = source.get("capture_identity") or {}
    _require(source.get("status") == "completed" and source.get("phase") == "run"
             and all(identity.get(key) == attempt.get(key) for key in ("run_id", "attempt_id", "fingerprint"))
             and len(source["instance_outcomes"]) == attempt.get("expected_instance_count")
             and source["result"]["instance_outcomes"] == source["instance_outcomes"], "source rollout is incomplete")
    cut = len(prefix["interactions"])
    _require(0 < cut < len(source["interactions"]), "historical cut must leave a next official Query")
    query = source["interactions"][cut]["query"]
    matching_outcomes = [index for index, row in enumerate(source["instance_outcomes"])
                         if (row.get("instance_id"), row.get("instance_index"))
                         == (query.get("instance_id"), query.get("instance_index"))]
    outcome_count = _one(matching_outcomes, "next Query has no unique official outcome")
    _require(prefix["instance_outcomes"] == source["instance_outcomes"][:outcome_count],
             "historical prefix omits or includes the wrong official outcomes")

    ancestry = {}
    current_path, current_attempt, current_trace = path, attempt, source
    while True:
        identifier = current_attempt["attempt_id"]
        _require(identifier not in ancestry, "native attempt ancestry has a cycle")
        ancestry[identifier] = (current_attempt, current_trace)
        parent = verify_resume_ancestry(batch, current_path, current_attempt, current_trace)
        if parent is None:
            break
        current_path, current_attempt, current_trace = parent
    return _prepare_native_prefix(batch, attempt, prefix, ancestry=ancestry, outcome_count=outcome_count)


def _sealed_source(batch, attempt):
    task = attempt["task"]
    trace_path = _contained(batch, attempt["trace_path"])
    _require(trace_path.parent.parent == batch / "checkpoints" / task / "attempts" / f"run-{attempt['index']}"
             and trace_path.parent.name == attempt["attempt_id"], "trace is outside its attempt directory")
    compressed = trace_path.read_bytes()
    trace_digest = sha256(compressed).hexdigest()
    _require(trace_digest == attempt.get("trace_sha256"), "source trace hash differs")
    return trace_path, json.loads(gzip.decompress(compressed)), trace_digest


def _historical_runtime(batch, task, meta):
    origin = Path(meta["runtime_directory"])
    relative = Path(*origin.parts[-3:])
    _require(origin.is_absolute() and relative.parent == Path("native") / task
             and re.fullmatch(r"[0-9a-f]{32}", relative.name) is not None
             and meta.get("runtime_id") == relative.name, "historical runtime identity is invalid")
    return _contained(batch, relative), origin


def _verify_historical_seed(batch, directory, owner, origin, seed, raw, ledger, ancestry):
    relative = directory.relative_to(batch).as_posix()
    receipt = owner.get("native_resume") or {}
    if not seed:
        _require(receipt.get("restored_workspace") != str(origin / "workspace"), "native seed receipt is absent")
        return
    _require(receipt and all(seed.get(key) == value for key, value in receipt.items())
             and seed.get("restored_workspace") == str(origin / "workspace")
             and relative in owner["runtime_directories"], "native seed disagrees with its owning attempt")
    parent = owner.get("resume_from") or {}
    previous = ancestry.get(seed.get("source_attempt_id"))
    _require(previous is not None and seed.get("source_attempt_id") == parent.get("attempt_id"),
             "native seed does not belong to a verified parent attempt")
    parent_attempt, parent_trace = previous
    _require(parent_attempt["status"] in {"failed", "interrupted"}
             and seed.get("source_trace") == parent_attempt.get("trace_path")
             and seed.get("source_trace_sha256") == parent_attempt.get("trace_sha256")
             and seed.get("retained_interactions") == parent.get("interactions") == len(parent_trace["interactions"])
             and seed.get("retained_episodes") == parent.get("instances") == len(parent_trace["instance_outcomes"]),
             "native seed does not match the parent's sealed task prefix")
    previous_bundle = _prepare_native_prefix(batch, parent_attempt, parent_trace, ancestry=ancestry)
    _require(previous_bundle.session_id == seed.get("session_id")
             and previous_bundle.header["cwd"] == seed.get("source_workspace")
             and previous_bundle.uses_absolute_paths == seed.get("uses_absolute_paths")
             and all(seed.get(key) == value for key, value in previous_bundle.lineage.items())
             and b"".join(raw[1:seed["session_terminal_seq"] + 2]) == previous_bundle.events
             and b"".join(ledger[:seed["retained_usage_rows"]]) == previous_bundle.usage,
             "restored session or usage prefix differs from its sealed parent")
    memory = {_memory_name(name): _memory_object(directory, item) for name, item in seed["memory_files"].items()}
    _require(memory == previous_bundle.memory, "restored memory seed differs from its sealed parent")


def _prepare_native_prefix(batch, attempt, prefix, *, ancestry=None, outcome_count=None):
    task = attempt["task"]
    _, source, trace_digest = _sealed_source(batch, attempt)
    interactions = prefix["interactions"]
    retained_outcomes = source["instance_outcomes"] if outcome_count is None else source["instance_outcomes"][:outcome_count]
    _require(interactions and source["interactions"][:len(interactions)] == interactions
             and retained_outcomes == prefix["instance_outcomes"]
             and source["task"] == prefix["task"] and source["task"]["name"] == task
             and source["execution"]["run_index"] == attempt["index"],
             "provided prefix differs from source trace")
    boundary = interactions[-1]
    response = boundary["response"]
    meta = response["metadata"]
    if ancestry is None:
        directory = origin = _contained(batch, Path(meta["runtime_directory"]))
    else:
        directory, origin = _historical_runtime(batch, task, meta)
    relative = directory.relative_to(batch).as_posix()
    _require(directory.parent == batch / "native" / task
             and relative in attempt["runtime_directories"], "runtime is outside the source attempt")
    session = meta.get("session_id")
    _require(isinstance(session, str) and re.fullmatch(r"clbench-[0-9a-f]{32}", session), "invalid session ID")
    _, boundaries = _lines(_contained(directory, "capture/boundaries.jsonl"))
    _require(any(row.get("kind") == "process_exit" and row.get("session_id") == session
                 and type(row.get("exit_code")) is int for row in boundaries), "source DSH process lacks an exit record")
    seeds = [row for row in boundaries if row.get("kind") == "native_resume" and row.get("session_id") == session]
    seed = seeds[0] if seeds else {}
    owner = None
    if ancestry is not None:
        reset = _one((row for row in boundaries if row.get("kind") == "reset"), "source runtime reset is ambiguous")
        source_owner = ancestry.get(reset.get("attempt_id"))
        _require(source_owner is not None and len(seeds) <= 1, "source runtime owner or seed is not verified")
        owner = source_owner[0]
        ancestor_order = tuple(ancestry)
        _require(ancestor_order.index(owner["attempt_id"]) >= ancestor_order.index(attempt["attempt_id"])
                 and reset.get("runtime_id") == directory.name and relative in owner["runtime_directories"]
                 and all(reset.get(key) == owner.get(key) for key in ("run_id", "job_id", "fingerprint", "execution_fingerprint"))
                 and (seed or reset.get("session_id") == session), "source runtime reset differs from owning attempt")
    path = _one(directory.glob(f"sessions/*/{session}/session.v4.jsonl"), "session log must be unique")
    raw, rows = _lines(_contained(directory, path))
    _require(rows and rows[0].get("type") == "session" and rows[0].get("id") == session
             and rows[0].get("version") == 4 and Path(rows[0]["cwd"]) == origin / "workspace"
             and path.parent.parent.name == _project_key(origin / "workspace")
             and all(row.get("seq") == index for index, row in enumerate(rows[1:])),
             "session header or sequence does not match the native runtime")
    call_id = meta["tool_call_id"]
    marker = _one((row for row in rows if row.get("type") == "tool/call"
                   and row["data"].get("name") == "clbench_submit" and row["data"].get("callId") == call_id),
                  "trace action is absent or ambiguous in the session")
    turn = marker["data"]["turn"]
    terminal = _one((row for row in rows if row.get("type") == "turn/end" and row["data"].get("turn") == turn),
                    "submission turn is absent or ambiguous")
    cutoff = terminal["seq"]
    result = _one((row for row in rows[marker["seq"] + 2:cutoff + 2] if row.get("type") == "tool/result"
                   and row["data"].get("message", {}).get("toolCallId") == call_id),
                  "submission has no unique native tool result")
    _require(terminal["data"]["reason"]["kind"] == "completed"
             and not result["data"]["message"].get("isError"), "submission did not complete")
    _, queries = _lines(_contained(directory, "capture/queries.jsonl"))
    query_id = meta["query_request_id"]
    confirmed = _one((row for row in queries if row.get("query_request_id") == query_id and row.get("kind") == "query_end"),
                     "source submission has no unique driver acknowledgement")
    observed = _one((row for row in queries if row.get("query_request_id") == query_id and row.get("kind") == "observation"),
                    "source action has no unique benchmark observation")
    raw_observation = observed["observation"]
    traced_observation = boundary["observation"]
    _require(confirmed.get("status") == "completed" and confirmed.get("tool_call_id") == call_id
             and confirmed.get("normalized_action") == response["action"]
             and all(raw_observation.get(key) == traced_observation.get(key) for key in ("content", "instance_complete"))
             and (raw_observation.get("metadata") or {}) == (traced_observation.get("metadata") or {}),
             "driver or benchmark acknowledgement differs")
    ledger, usage = _lines(_contained(directory, "usage.jsonl"))
    required = [
        event["metadata"]["call_id"]
        for step in interactions for phase in ("respond", "observe")
        for event in step.get("usage", {}).get(phase, {}).get("events", [])
        if event.get("metadata", {}).get("call_id")
    ]
    _require(required and len(set(required)) == len(required)
             and len(usage) >= len(required)
             and {row.get("callId") for row in usage[:len(required)]} == set(required),
             "usage rows do not match the acknowledged actions")
    if ancestry is not None:
        _require([row.get("callId") for row in usage[:len(required)]] == required,
                 "historical usage order differs from the acknowledged actions")
        _verify_historical_seed(batch, directory, owner, origin, seed, raw, ledger, ancestry)
    memory, uses_absolute_paths = _memory_at(directory, rows[1:], cutoff, seed, paths_at_cut=outcome_count is not None)
    session_bytes = b"".join(raw[1:cutoff + 2])
    usage_bytes = b"".join(ledger[:len(required)])
    return _NativePrefix(session, rows[0], session_bytes, usage_bytes, memory, {
        "source_attempt_id": attempt["attempt_id"], "source_trace": attempt["trace_path"],
        "source_trace_sha256": trace_digest, "source_runtime": relative,
        "source_session_sha256": sha256(b"".join(raw)).hexdigest(), "session_terminal_seq": cutoff,
        "retained_session_events_sha256": sha256(session_bytes).hexdigest(),
        "retained_usage_sha256": sha256(usage_bytes).hexdigest(), "retained_usage_rows": len(required),
        "retained_episodes": len(prefix["instance_outcomes"]), "retained_interactions": len(interactions),
    }, uses_absolute_paths)


def _project_key(workspace):
    encoded = str(workspace).encode("utf-16-le")
    parts = []
    for offset in range(0, len(encoded), 2):
        unit = int.from_bytes(encoded[offset:offset + 2], "little")
        char = chr(unit)
        if char in "/\\:":
            if not offset or chr(int.from_bytes(encoded[offset - 2:offset], "little")) not in "/\\:":
                parts.append("-")
        else:
            parts.append(char if re.fullmatch(r"[A-Za-z0-9._-]", char) else f"~{unit:04X}")
    return "--" + ("".join(parts).lstrip("-") or "root")[:251] + "--"


def restore_native_resume(system, bundle, prefix):
    """Copy the acknowledged state into the new runtime and attach it at first SDK start."""
    target = system.directory
    workspace = system.workspace
    _require(system._process is None and not system._used and not (target / "sessions").exists()
             and not system.usage_file.exists() and not any((workspace / "memory").iterdir()), "target runtime is not fresh")
    header = {**bundle.header, "cwd": str(workspace)}
    path = target / "sessions" / _project_key(workspace) / bundle.session_id / "session.v4.jsonl"
    path.parent.mkdir(parents=True, mode=0o700)
    path.write_bytes((json.dumps(header, ensure_ascii=False) + "\n").encode() + bundle.events)
    system.usage_file.write_bytes(bundle.usage)
    objects = target / "capture/objects"
    objects.mkdir(mode=0o700)
    memory_files = {}
    for name, data in bundle.memory.items():
        dest = _contained(workspace / "memory", name)
        dest.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        dest.write_bytes(data)
        digest = sha256(data).hexdigest()
        (objects / digest).write_bytes(data)
        memory_files[name] = {"status": "captured", "sha256": digest, "bytes": len(data), "representation": "file_bytes"}
    system.session_id = bundle.session_id
    system._usage_offset = len(bundle.usage)
    system._interaction_step = len(prefix["interactions"])
    last = prefix["interactions"][-1]
    system._current_query = {k: last["query"][k] for k in ("instance_id", "instance_index")}
    system._feedback = [last["observation"]["content"]]
    if bundle.uses_absolute_paths:
        system._feedback.append(f"The private workspace was relocated. Continue using the memory files in {workspace / 'memory'}.")
    lineage = {**bundle.lineage, "session_id": bundle.session_id, "memory_files": memory_files,
               "uses_absolute_paths": bundle.uses_absolute_paths, "source_workspace": bundle.header["cwd"],
               "restored_workspace": str(workspace), "relocation_feedback_added": bundle.uses_absolute_paths}
    system._record("boundaries", "native_resume", **lineage)
    start = system._start

    def attach(self):
        was_stopped = self._process is None
        start()
        if was_stopped and self.directory == target:
            try:
                self._request("session/resume", {"sessionId": bundle.session_id}, time.monotonic() + min(self.timeout, 60))
            except BaseException:
                self._stop_process("native_resume_failure")
                raise

    system._start = MethodType(attach, system)
    return lineage
