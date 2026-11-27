from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import importlib.util
import math
from pathlib import Path
import random
import sys
from tempfile import TemporaryDirectory
from threading import Thread
import unittest
from unittest.mock import patch

from pydantic import BaseModel

if sys.version_info < (3, 13) or importlib.util.find_spec("litellm") is None:
    raise unittest.SkipTest("native Codex tests require the documented Python 3.13 CL-Bench environment")

from skill_rsi_rl.harnesses.codex_system import CodexSystem
from src.interface import Observation, Query


class Answer(BaseModel):
    answer: str


class CodexSystemTests(unittest.TestCase):
    def test_isolation_resume_feedback_usage_and_public_evidence(self):
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                self.rfile.read(int(self.headers["Content-Length"]))
                raw = json.dumps({"id": "synthetic-response", "status": "completed", "output": []}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def log_message(self, *args):
                pass

        with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server, TemporaryDirectory() as temporary:
            thread = Thread(target=server.serve_forever, daemon=True)
            thread.start()
            self.addCleanup(server.shutdown)
            with patch.dict("os.environ", {"CODEX_TEST_KEY": "synthetic-only"}):
                native = CodexSystem("synthetic", runtime_root=temporary, timeout=30,
                                     command=str(Path(__file__).parent / "fixtures/codex_cli.py"),
                                     upstream=f"http://127.0.0.1:{server.server_port}/v1",
                                     credential_env="CODEX_TEST_KEY")
                self.addCleanup(native.close)
                original_workspace = native.workspace
                first = native.respond(Query("Return ok", Answer, instance_id="one", instance_index=0))
                observation = Observation("Public previous result")
                native.observe(observation)
                second = native.respond(Query("Return ok again", Answer, feedback=observation,
                                              instance_id="two", instance_index=1))
                self.assertEqual(first.action.answer, "ok")
                self.assertTrue(second.metadata["resumed"])
                self.assertEqual(second.metadata["token_usage"], {"input_tokens": 3, "output_tokens": 2})
                self.assertEqual(second.metadata["plugins"], [])
                self.assertEqual(second.metadata["extensions"], [])
                launch = json.loads((Path(second.metadata["capture_path"]) / "prompt.json").read_text())
                self.assertEqual(launch["prompt"].count("Public previous result"), 1)
                from skill_rsi.evidence.verify import verify
                verify(Path(second.metadata["evidence_path"]))
                self.assertTrue(list(Path(temporary).rglob("request.raw")))
                native.reset()
                self.assertNotEqual(native.workspace, original_workspace)
                self.assertFalse(original_workspace.exists())
                self.assertIsNone(native.session)
                self.assertEqual(len(native.get_run_artifacts()["captures"]), 2)

                # A synthetic student traverses the real wrapper, native task,
                # application receipts and official scorer without training.
                from skill_rsi_rl.contracts import propose_options
                from skill_rsi_rl.harnesses.runner import run_rollout
                from skill_rsi_rl.serve import make_server

                def sample(query, feedback, memory, query_id, seed):
                    options = propose_options(query, feedback, memory)
                    index = random.Random(seed).randrange(len(options))
                    return {"action": options[index]["action"], "option_index": index,
                            "prompt_token_ids": [1], "option_token_ids": list(range(len(options))),
                            "behavior_logprobs": [-math.log(len(options))] * len(options)}

                with make_server("127.0.0.1", 0, "synthetic-student", "synthetic-policy",
                                 lambda *args: {"op": "noop"}, sample=sample) as policy:
                    worker = Thread(target=policy.serve_forever, daemon=True)
                    worker.start()
                    try:
                        report = run_rollout({"harness": "codex", "task": "exploitable_poker", "split": "train",
                            "endpoint": f"http://127.0.0.1:{policy.server_port}", "policy_model": "synthetic-student",
                            "initial_memory": {"public_note": "Synthetic prior public observation"},
                            "task_params": {"num_instances": 1, "opponent_policy": "calling_station"},
                            "system_params": {"backend": "skill_rsi", "model": "synthetic", "timeout": 30,
                                "command": str(Path(__file__).parent / "fixtures/codex_cli.py"),
                                "upstream": f"http://127.0.0.1:{server.server_port}/v1", "credential_env": "CODEX_TEST_KEY"}},
                            environment_seed=71011, sampling_seed=17, policy_id="synthetic-policy",
                            output=Path(temporary) / "official", sample=True)
                        self.assertEqual(report["status"], "completed")
                        self.assertEqual(len(report["outcomes"]), 1)
                        self.assertTrue(report["decisions"])
                        self.assertTrue(all(d["applied"] for d in report["decisions"]))
                        self.assertTrue((Path(temporary) / "official/trace.json.gz").is_file())
                    finally:
                        policy.shutdown()
                        worker.join()

    def test_changed_framework_identity_rejected(self):
        with self.assertRaisesRegex(ValueError, "changed"):
            CodexSystem("synthetic", source_id="sha256:" + "0" * 64)


if __name__ == "__main__":
    unittest.main()
