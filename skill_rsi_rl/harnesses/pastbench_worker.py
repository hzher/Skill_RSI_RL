"""Native PAST tool execution and original grading in a dedicated worker.

The public-action client follows Skill_RSI's MIT-licensed pastbench_worker;
benchmark files, tool dispatch, task budgets and grading are not rewritten.
"""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import queue
import random
import socket
import sys
import threading
import traceback
from uuid import uuid4

from .native_spec import contained


def verify_installed_package(root):
    import past_bench
    installed = Path(past_bench.__file__).resolve().parent
    expected = root / "src/past_bench"
    for path in installed.rglob("*.py"):
        original = expected / path.relative_to(installed)
        if not original.is_file() or hashlib.sha256(path.read_bytes()).digest() != hashlib.sha256(original.read_bytes()).digest():
            raise ValueError("installed PAST package differs from the pinned dataset source")


class Session:
    def __init__(self, benchmark, episode, output):
        self.benchmark, self.episode, self.output = benchmark, episode, output
        self.events, self.actions = queue.Queue(), queue.Queue()
        self.thread = None
        self.result = None

    def reset(self):
        if self.thread is not None:
            raise ValueError("a native PAST worker owns one original task")
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        return self._next()

    def _next(self):
        event, value = self.events.get(timeout=self.benchmark["native"].get("rpc_timeout", 900))
        if event == "failed":
            raise RuntimeError(value)
        if event == "finished":
            self.result = value
            return dict(done=True, current=None)
        return dict(done=False, current=value)

    def _run(self):
        try:
            root = Path(self.benchmark["native"]["repository"]).resolve()
            verify_installed_package(root)
            from past_bench import cli
            from past_bench.models.task import TaskDefinition
            from past_bench.runner import agent_orchestrator, services
            from past_bench.runtime.protocol import BootstrapResponse, StartSessionResponse, StepResponse, ToolCallAction
            from past_bench.models.content import TextBlock, ToolUseBlock
            from past_bench.models.message import Message
            import httpx

            task_dir = contained(root, self.episode["task"])
            task = TaskDefinition.from_yaml(task_dir / "task.yaml")
            lock_root = Path("/tmp/skill-rsi-native-service-locks")
            lock_root.mkdir(mode=0o700, exist_ok=True)
            lock = None
            for offset in random.SystemRandom().sample(range(11000, 45000), 512):
                if any(s.port + offset > 65535 for s in task.services):
                    continue
                handle = (lock_root / str(offset)).open("a")
                try:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    probes = []
                    try:
                        for service in task.services:
                            probe = socket.socket()
                            probes.append(probe)
                            probe.bind(("127.0.0.1", service.port + offset))
                    finally:
                        for probe in probes:
                            probe.close()
                except (OSError, BlockingIOError):
                    handle.close()
                    continue
                lock = handle
                break
            if lock is None:
                raise RuntimeError("no independent native service port set available")

            class OwnedServices(services.ServiceManager):
                def __enter__(manager):
                    for service in manager._services:
                        with socket.socket() as probe:
                            if probe.connect_ex(("127.0.0.1", service.port)) == 0:
                                raise RuntimeError("refusing to share or reset another run's native service")
                    return super().__enter__()

                def reset_all(manager):
                    with httpx.Client(trust_env=False, timeout=5) as client:
                        for service in manager._services:
                            if service.reset_endpoint:
                                client.post(service.reset_endpoint).raise_for_status()

            services.ServiceManager = OwnedServices
            outer = self

            class Client:
                def start(self, **kwargs):
                    pass

                def stop(self):
                    pass

                def bootstrap(self, request):
                    return BootstrapResponse(agent_name=request.agent_name, already_present=True)

                def start_session(self, request):
                    self.request = request
                    return StartSessionResponse(session_id=request.session_id, agent_name=request.agent_name,
                                                adapter="skill-rsi-rl-native-actions")

                def step(self, request):
                    start = self.request
                    public = dict(messages=[m.model_dump(mode="json") for m in start.initial_messages] if request.step_id == 0 else [],
                                  tools=[t.model_dump(mode="json") for t in start.tools] if request.step_id == 0 else [],
                                  tool_results=[t.model_dump(mode="json") for t in request.tool_results])
                    outer.events.put(("query", public))
                    action = outer.actions.get(timeout=start.timeout_seconds)
                    if action is None:
                        raise RuntimeError("native host disconnected")
                    calls = [ToolCallAction(tool_use_id=uuid4().hex, **value) for value in action["tool_calls"]]
                    if any(call.name not in {tool.name for tool in start.tools} for call in calls):
                        raise ValueError("unavailable original PAST tool")
                    blocks = [ToolUseBlock(id=c.tool_use_id, name=c.name, input=c.arguments) for c in calls]
                    if action["final_output"] is not None:
                        blocks.append(TextBlock(text=action["final_output"]))
                    return StepResponse(status="acting" if calls else "finished", tool_calls=calls,
                                        assistant_message=Message(role="assistant", content=blocks), final_output=action["final_output"])

                def close_session(self, request):
                    pass

            agent_orchestrator.create_runtime_client = lambda *args, **kwargs: Client()
            registry = self.output / "registry.json"
            registry.write_text(json.dumps({"agents": {"skill-rsi-rl": {"adapter": "openai_compat_chat",
                                "default_model": {"model_id": "framework-managed"}}}}))
            judge = self.benchmark["native"].get("judge", {})
            if not judge or not all(judge.get(k) for k in ("model", "upstream", "credential_env")):
                raise ValueError("native PAST requires an explicit original judge configuration")
            if not os.environ.get(judge["credential_env"]):
                raise ValueError("native judge credential environment is empty")
            config = self.output / "native-config.json"
            config.write_text(json.dumps({"judge": {"enabled": True, "model_id": judge["model"],
                "base_url": judge["upstream"], "api_key": "${" + judge["credential_env"] + "}"}}))
            try:
                result = cli._run_single_task(task_dir=str(task_dir), config_path=str(config), model="framework-managed",
                    api_key=None, base_url=None, trace_dir=str(self.output / "native"), port_offset=offset,
                    no_judge=False, judge_model=judge["model"], trials=1, sandbox=False,
                    agent="skill-rsi-rl", runtime="local", registry=str(registry))
            finally:
                lock.close()
            (self.output / "native-result.json").write_text(json.dumps(result, ensure_ascii=False))
            if result.get("error") or len(result.get("trials", [])) != 1 or result["trials"][0].get("error"):
                raise RuntimeError("original PAST runtime or grader failed")
            trial = result["trials"][0]
            self.events.put(("finished", dict(reward=trial["task_score"], success=trial["passed"],
                                            raw_metric_name="task_score", original_task_id=result["task_id"])))
        except BaseException as exc:
            traceback.print_exc(file=sys.stderr)
            self.events.put(("failed", type(exc).__name__))

    def step(self, action):
        if not isinstance(action, dict) or set(action) != {"tool_calls", "final_output"}:
            raise ValueError("expected original PAST tool_calls and final_output")
        if not isinstance(action["tool_calls"], list) or not (action["final_output"] is None or isinstance(action["final_output"], str)):
            raise ValueError("invalid PAST action")
        if not action["tool_calls"] and action["final_output"] is None:
            raise ValueError("empty PAST action")
        self.actions.put(action)
        return self._next()

    def score(self):
        if self.result is None:
            raise ValueError("PAST task has not finished")
        return self.result

    def close(self):
        self.actions.put(None)
        if self.thread:
            self.thread.join(timeout=5)
