"""Exercise algorithm discovery and argument forwarding without model dependencies."""

import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


class AlgorithmCliTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        package = self.root / "skill_rsi_rl"
        self.algorithms = package / "algorithms"
        self.algorithms.mkdir(parents=True)
        (package / "__init__.py").touch()

        # Copy the real dispatcher and GRPO entry points. Replace only the model trainers.
        source = Path(__file__).resolve().parent
        for path in source.glob("*.py"):
            if not path.name.startswith("test_"):
                shutil.copyfile(path, self.algorithms / path.name)
        shutil.copytree(source / "grpo", self.algorithms / "grpo", ignore=shutil.ignore_patterns("__pycache__"))
        for module in ("train", "online"):
            self.write_entry(package / f"{module}.py")

    @staticmethod
    def write_entry(path):
        path.write_text(
            "import json, sys\n"
            "def main():\n"
            "    print(json.dumps([__spec__.name, *sys.argv[1:]]))\n"
            "if __name__ == '__main__':\n"
            "    main()\n",
            encoding="utf-8",
        )

    def plugin(self, name):
        directory = self.algorithms / name
        directory.mkdir()
        (directory / "__init__.py").touch()
        return directory

    def cli(self, *args):
        return subprocess.run(
            [sys.executable, "-E", "-B", "-m", "skill_rsi_rl.algorithms", *args],
            cwd=self.root, capture_output=True, text=True, timeout=10,
        )

    def test_new_plugin_discovery_lists_modes_without_importing_training_modules(self):
        plugin = self.plugin("fixture_policy")
        (plugin / "offline.py").write_text("raise ImportError('optional model package absent')\n")
        result = self.cli("--list")
        self.assertEqual(result.returncode, 0, result.stderr)
        rows = result.stdout.splitlines()
        plugin_row = next(row for row in rows if "fixture_policy" in row)
        self.assertIn("offline", plugin_row)
        self.assertNotIn("online", plugin_row)
        grpo_row = next(row for row in rows if "grpo" in row)
        for mode in ("offline", "online"):
            self.assertIn(mode, grpo_row)

    def test_new_plugin_can_run_from_its_own_directory(self):
        self.write_entry(self.plugin("fixture_policy") / "offline.py")
        for args in (["--tag", "run with spaces", "--help", "--", ""], ["--", "-q"]):
            with self.subTest(args=args):
                result = self.cli("fixture_policy", "offline", *args)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(json.loads(result.stdout), ["skill_rsi_rl.algorithms.fixture_policy.offline", *args])

    def test_grpo_forwards_cli_arguments_to_original_trainers(self):
        for mode, trainer in (("offline", "train"), ("online", "online")):
            with self.subTest(mode=mode):
                result = self.cli("grpo", mode, "--help")
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(json.loads(result.stdout), [f"skill_rsi_rl.{trainer}", "--help"])

    def test_unknown_names_and_unavailable_modes_are_rejected(self):
        self.write_entry(self.plugin("fixture_policy") / "offline.py")
        self.plugin("empty_policy")
        for args in (
            ("fixture_policy",), ("fixture_policy", "online"), ("empty_policy", "offline"),
            ("unknown_policy", "offline"), ("../grpo", "offline"), ("grpo.offline", "offline"),
            ("--list", "grpo"),
        ):
            with self.subTest(args=args):
                result = self.cli(*args)
                self.assertNotEqual(result.returncode, 0)
                self.assertNotIn("Traceback", result.stderr)


if __name__ == "__main__":
    unittest.main()
