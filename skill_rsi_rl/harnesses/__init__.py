"""Native CL-Bench systems and historical DSH replay support."""

from functools import lru_cache
from pathlib import Path
import sys

from skill_rsi_rl.sources import verify_source


@lru_cache(maxsize=1)
def activate_benchmark():
    """Resolve the vendored benchmark before importing its top-level src package."""
    root, commit = verify_source("clbench")
    loaded = sys.modules.get("src")
    if loaded is not None and Path(loaded.__file__).resolve() != root / "src/__init__.py":
        raise ImportError("The active src module does not belong to the pinned CL-Bench checkout")
    sys.path.insert(0, str(root))
    return root, commit
