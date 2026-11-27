"""Bounded JSONL RPC to isolated, dataset-owned execution and scoring workers."""
import json
import os
from pathlib import Path
import queue
import signal
import subprocess
from threading import Thread


class NativeSession:
    def __init__(self, benchmark, episode, output):
        self.output = Path(output).resolve()
        self.output.mkdir(parents=True, exist_ok=False)
        self.timeout = benchmark["native"].get("rpc_timeout", 900)
        if type(self.timeout) not in (int, float) or not 0 < self.timeout <= 3600:
            raise ValueError("invalid native RPC timeout")
        self.replies = queue.Queue()
        request = dict(benchmark=benchmark, episode=episode, output=str(self.output))
        config = self.output / "worker.json"
        config.write_text(json.dumps(request))
        with (self.output / "worker.stderr").open("wb") as stderr:
            self.process = subprocess.Popen([benchmark["native"]["python"], "-u", "-m",
                "skill_rsi_rl.harnesses.native_worker", str(config)], cwd=benchmark["native"]["repository"],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr, text=True,
                bufsize=1, start_new_session=True)
        self.reader = Thread(target=self._read, daemon=True)
        self.reader.start()

    def _read(self):
        try:
            for line in self.process.stdout:
                self.replies.put(line)
        finally:
            self.replies.put(None)

    def request(self, method, **params):
        try:
            self.process.stdin.write(json.dumps({"method": method, "params": params}) + "\n")
            self.process.stdin.flush()
            raw = self.replies.get(timeout=self.timeout)
        except (OSError, queue.Empty) as exc:
            raise RuntimeError(f"native {method} did not respond; see {self.output / 'worker.stderr'}") from exc
        if raw is None:
            raise RuntimeError(f"native worker exited; see {self.output / 'worker.stderr'}")
        row = json.loads(raw)
        if "error" in row:
            raise RuntimeError(f"native {method} failed ({row['error']}); see {self.output / 'worker.stderr'}")
        return row["result"]

    def close(self):
        if self.process.stdin and not self.process.stdin.closed:
            self.process.stdin.close()
        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGKILL)
                self.process.wait()
            except ProcessLookupError:
                pass
        self.reader.join(timeout=2)
        self.process.stdout.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
