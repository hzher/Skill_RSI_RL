"""Streaming access to the published JSONL replay contract."""

from contextlib import contextmanager
from hashlib import sha256
import json
from pathlib import Path

from .contracts import _validate_manifest, _validated_groups


def _parse_replay(lines):
    first = next(lines, None)
    if first is None:
        raise ValueError("empty replay")
    manifest = json.loads(first)
    _validate_manifest(manifest)
    return manifest, _validated_groups(json.loads(line) for line in lines if line.strip())


@contextmanager
def open_replay(path):
    with Path(path).open(encoding="utf-8") as stream:
        yield _parse_replay(stream)


@contextmanager
def open_hashed_replay(path):
    """Read one snapshot and expose its byte digest after consuming all groups."""
    with Path(path).open("rb") as stream:
        digest = sha256()

        def decoded_lines():
            for line in stream:
                digest.update(line)
                yield line.decode("utf-8")

        manifest, groups = _parse_replay(decoded_lines())
        yield manifest, groups, digest.hexdigest
