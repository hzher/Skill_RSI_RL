"""Locate and verify the source snapshots shipped beside this package."""

import hashlib
import json
from functools import lru_cache
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent


@lru_cache(maxsize=None)
def verify_source(name: str) -> tuple[Path, str]:
    record = json.loads((PROJECT / "sources.lock.json").read_text())[name]
    root = (PROJECT / record["path"]).resolve(strict=True)
    root.relative_to(PROJECT)
    files = record["files"]
    for relative, expected in files.items():
        path = root / relative
        if path.resolve() != path or not path.is_file():
            raise ValueError(f"{name}: missing or redirected source {relative}")
        with path.open("rb") as stream:
            sha = hashlib.sha256()
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                sha.update(block)
        if sha.hexdigest() != expected:
            raise ValueError(f"{name}: SHA-256 mismatch for {relative}")
    package = "verl" if name == "easyr1" else "src"
    for path in (root / package).rglob("*"):
        if path.is_symlink() or (path.suffix in {".py", ".so", ".pyd"} and path.relative_to(root).as_posix() not in files):
            raise ValueError(f"{name}: unexpected source {path.relative_to(root)}")
    return root, record["commit"]
