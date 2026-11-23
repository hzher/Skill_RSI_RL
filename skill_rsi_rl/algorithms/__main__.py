"""Run a post-training algorithm from its own package."""

import argparse
from importlib import import_module
from pathlib import Path
import re
import sys


def available_algorithms():
    """Discover installed modes without importing trainers or their dependencies."""
    algorithms = {}
    for child in sorted(Path(__file__).parent.iterdir()):
        if not re.fullmatch(r"[a-z][a-z0-9_]*", child.name) or not (child / "__init__.py").is_file():
            continue
        modes = [mode for mode in ("offline", "online") if (child / f"{mode}.py").is_file()]
        if modes:
            algorithms[child.name] = modes
    return algorithms


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m skill_rsi_rl.algorithms", description=__doc__,
                                     usage="%(prog)s --list | %(prog)s ALGORITHM {offline,online} [trainer args ...]",
                                     allow_abbrev=False)
    parser.add_argument("--list", action="store_true", help="list algorithms and their supported modes")
    parser.add_argument("algorithm", nargs="?", help="an installed algorithm name")
    parser.add_argument("mode", nargs="?", choices=("offline", "online"))
    argv = list(sys.argv[1:] if argv is None else argv)
    # Only the first two tokens belong to the launcher. Keep the trainer's own
    # flags and its literal "--" delimiter intact.
    args = parser.parse_args(argv[:2])
    available = available_algorithms()
    if args.list:
        if len(argv) != 1:
            parser.error("use --list on its own")
        for name, modes in available.items():
            print(f"{name}: {', '.join(modes)}")
        return
    if args.algorithm not in available:
        parser.error(f"select an installed algorithm with --list (got {args.algorithm!r})")
    if args.mode not in available[args.algorithm]:
        modes = ", ".join(available[args.algorithm])
        parser.error(f"{args.algorithm} supports: {modes} (got {args.mode!r})")

    module = import_module(f"{__package__}.{args.algorithm}.{args.mode}")
    original_argv = sys.argv
    try:
        sys.argv = [f"{parser.prog} {args.algorithm} {args.mode}", *argv[2:]]
        return module.main()
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    raise SystemExit(main())
