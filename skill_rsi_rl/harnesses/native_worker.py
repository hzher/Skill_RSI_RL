"""JSONL boundary; stdout is exclusively RPC, native runtime output goes to stderr."""
from contextlib import redirect_stdout
import json
from pathlib import Path
import sys
import traceback

from .native_spec import KINDS, kind, prepare


def main():
    config = json.loads(Path(sys.argv[1]).read_text())
    channel = sys.stdout
    session = None
    try:
        with redirect_stdout(sys.stderr):
            benchmark = prepare(config["benchmark"])
            if config["episode"] not in benchmark["native"]["plan"]:
                raise ValueError("worker episode differs from the pinned native plan")
            if kind(benchmark) == "past-bench":
                from .pastbench_worker import Session
            elif kind(benchmark) == "workbuddy":
                from .workbuddy_worker import Session
            else:
                raise ValueError(f"expected native benchmark in {KINDS}")
            session = Session(benchmark, config["episode"], Path(config["output"]))
        for line in sys.stdin:
            try:
                request = json.loads(line)
                if request["method"] not in {"reset", "step", "score"}:
                    raise ValueError("unknown native RPC method")
                with redirect_stdout(sys.stderr):
                    result = getattr(session, request["method"])(**request.get("params", {}))
                response = {"result": result}
            except Exception as exc:
                traceback.print_exc(file=sys.stderr)
                response = {"error": type(exc).__name__}
            channel.write(json.dumps(response, ensure_ascii=False, allow_nan=False) + "\n")
            channel.flush()
            if "error" in response:
                break
    finally:
        if session is not None:
            with redirect_stdout(sys.stderr):
                session.close()


if __name__ == "__main__":
    main()
