"""Original WorkBuddy Docker task and CompositeVerifier, without harness addons."""
import asyncio
import json
import math
from pathlib import Path
import time
from uuid import uuid4

from .native_spec import contained


class Session:
    def __init__(self, benchmark, episode, output):
        self.benchmark, self.episode, self.output = benchmark, episode, output
        self.loop = asyncio.Runner()
        self.environment = None
        self.done = False
        self.started = None
        self.result = None

    def reset(self):
        if self.environment is not None:
            raise ValueError("WorkBuddy workers own one original task")
        from harbor.models.task.task import Task
        from harbor.models.task.config import NetworkPolicy
        from harbor.models.trial.paths import TrialPaths
        from workbuddy_bench.runner.docker_environment import WorkBuddyDockerEnvironment

        task_dir = contained(Path(self.benchmark["native"]["repository"]).resolve(), self.episode["task"])
        self.task = Task(task_dir=task_dir)
        self.paths = TrialPaths(trial_dir=self.output / "trial")
        self.paths.mkdir()
        self.paths.chmod_dir()
        config = self.task.config.environment
        self.environment = WorkBuddyDockerEnvironment(environment_dir=self.task.paths.environment_dir,
            environment_name=self.task.short_name, session_id="rsi-" + uuid4().hex,
            trial_paths=self.paths, task_env_config=config, mounts=[],
            network_policy=NetworkPolicy(network_mode=config.network_mode, allowed_hosts=config.allowed_hosts or []))
        self.loop.run(self.environment.start(force_build=False))
        self.started = time.monotonic()
        return dict(done=False, current=dict(instruction=self.task.instruction,
            tools=[dict(name="shell", description="Run a shell command inside the original task container. "
                        "Return command=null and final_output when finished.")]))

    def step(self, action):
        if self.done or self.started is None:
            raise ValueError("WorkBuddy task is not active")
        if set(action) != {"command", "final_output"}:
            raise ValueError("expected terminal command or final_output")
        command, final = action["command"], action["final_output"]
        if command is not None and (not isinstance(command, str) or not command.strip() or final is not None):
            raise ValueError("invalid terminal command")
        if command is None:
            if not isinstance(final, str):
                raise ValueError("terminal finish requires final_output")
            self.done = True
            return dict(done=True, current=None)
        remaining = self.task.config.agent.timeout_sec - (time.monotonic() - self.started)
        if remaining <= 0:
            raise TimeoutError("native WorkBuddy agent time budget exhausted")
        result = self.loop.run(self.environment.exec(command=command, timeout_sec=max(1, min(120, int(remaining)))))
        return dict(done=False, current=dict(tool_results=[dict(name="shell", stdout=result.stdout or "",
            stderr=result.stderr or "", return_code=result.return_code)]))

    def score(self):
        if not self.done:
            raise ValueError("WorkBuddy must finish before hidden tests are made available")
        if self.result is None:
            from workbuddy_bench.judge.profiles.dispatcher import CompositeVerifier

            verifier = CompositeVerifier(task=self.task, trial_paths=self.paths, environment=self.environment)
            result = self.loop.run(asyncio.wait_for(verifier.verify(), timeout=self.task.config.verifier.timeout_sec))
            payload = result.rewards
            if not payload or type(payload.get("reward")) not in (int, float) or not math.isfinite(payload["reward"]):
                raise ValueError("original WorkBuddy verifier produced no finite reward")
            (self.output / "native-result.json").write_text(json.dumps(result.model_dump(mode="json")))
            # WorkBuddy's standard reward is a fraction of passed checks.
            self.result = dict(reward=payload["reward"], success=payload["reward"] == 1,
                raw_metric_name="reward", original_task_id=self.episode["label"], rewards=payload)
        return self.result

    def close(self):
        try:
            if self.environment is not None:
                self.loop.run(asyncio.wait_for(self.environment.stop(delete=True), timeout=90))
        finally:
            self.loop.close()
