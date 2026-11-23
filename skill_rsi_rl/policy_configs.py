"""Bind an evaluation arm to its service and its selected checkpoint."""

import argparse
import json
from pathlib import Path
import shlex
import sys

from .contracts import MODEL_PATH
from .serve import checkpoint_identities, checkpoint_identity


def make_configs(report, model, adapter, ports, timeout=10):
    if len(set(ports)) != 2 or any(not 0 < port < 65536 for port in ports):
        raise ValueError("Provide separate TCP ports for the two policies")
    if not 0 < timeout <= 60:
        raise ValueError("Policy timeout must be in (0, 60] seconds")
    identities = checkpoint_identities(model, adapter)
    if identities[0] == identities[1] or identities != (report["base_policy_id"], report["policy_id"]):
        raise ValueError("The supplied base and adapter do not match the scored policy identities")
    return {arm: {"endpoint": f"http://127.0.0.1:{port}/decide", "model": model.name,
                  "checkpoint": identity, "timeout": timeout}
            for arm, port, identity in zip(("base", "trained"), ports, identities, strict=True)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path(MODEL_PATH))
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-port", type=int, default=8711)
    parser.add_argument("--trained-port", type=int, default=8712)
    parser.add_argument("--online-result", type=Path, help="published round's result.json for a fourth arm")
    parser.add_argument("--online-port", type=int, default=8713)
    parser.add_argument("--timeout", type=float, default=10)
    args = parser.parse_args()
    try:
        report = json.loads(args.report.read_text(encoding="utf-8"))
        ports = [args.base_port, args.trained_port]
        configs = make_configs(report, args.model, args.adapter, ports, args.timeout)
        adapters = {"trained": args.adapter}
        if args.online_result is not None:
            if args.online_port in ports or not 0 < args.online_port < 65536:
                raise ValueError("Use a distinct TCP port for the online policy")
            publication = json.loads(args.online_result.read_text(encoding="utf-8"))
            if publication["checkpoint"] != "checkpoint":
                raise ValueError("Unexpected checkpoint location in online publication")
            adapter = args.online_result.parent / "checkpoint"
            online_id = checkpoint_identity(args.model, adapter)
            if online_id != publication["policy_id"] or online_id in {c["checkpoint"] for c in configs.values()}:
                raise ValueError("Online checkpoint differs from its publication or duplicates another arm")
            ports.append(args.online_port)
            adapters["online"] = adapter
            configs["online"] = {**configs["base"], "endpoint": f"http://127.0.0.1:{args.online_port}/decide",
                                 "checkpoint": online_id}
        args.output.mkdir(parents=True, exist_ok=True)
        targets = {args.output / f"{arm}.json": value for arm, value in configs.items()}
        for path, value in targets.items():
            if path.exists() and json.loads(path.read_text(encoding="utf-8")) != value:
                raise ValueError(f"Existing policy config differs: {path}")
        for path, value in targets.items():
            if not path.exists():
                with path.open("x", encoding="utf-8") as stream:
                    json.dump(value, stream, indent=2)
                    stream.write("\n")
        for rank, (arm, port) in enumerate(zip(configs, ports, strict=True)):
            command = [sys.executable, "-B", "-m", "skill_rsi_rl.serve", "--model", str(args.model),
                       "--model-id", configs[arm]["model"], "--device", f"cuda:{rank}", "--port", str(port)]
            if arm in adapters:
                command.extend(("--adapter", str(adapters[arm])))
            print(shlex.join(command))
    except (OSError, ValueError, KeyError) as error:
        parser.exit(2, f"{type(error).__name__}: {error}\n")


if __name__ == "__main__":
    main()
