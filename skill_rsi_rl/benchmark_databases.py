"""Stage the two pinned official CL Bench SQLite assets with resumable ranges."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import os
from pathlib import Path
import subprocess
import tempfile
import time


REPOSITORY = "datasets/continual-learning-bench/database-exploration/resolve/main"
ASSETS = {
    "products.db": (397910016, "edf8ee80ff125de0bfd6c37a1d185efa9e3037ce28eb1bd1d32ae0829bd264a6"),
    "products_drifted.db": (432148480, "a53d523f70604be0e4328f3722417895250ac35e1127cc504b609576aee70fad"),
}
BLOCK_SIZE = 8 * 2**20


def _digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _download_block(mirrors, directory, name, start, end):
    output = directory / f"{name}.{start}"
    expected = end - start + 1
    if output.is_file() and output.stat().st_size == expected:
        return output
    temporary = output.with_suffix(f".{start}.downloading")
    for attempt in range(6):
        endpoint = mirrors[attempt % len(mirrors)]
        result = subprocess.run([
            "curl", "-fsSL", "--connect-timeout", "12", "--max-time", "600",
            "--range", f"{start}-{end}", "--output", str(temporary), "--write-out", "%{response_code}",
            f"{endpoint.rstrip('/')}/{REPOSITORY}/{name}",
        ], capture_output=True, text=True, check=False)
        if (result.returncode == 0 and result.stdout == "206" and temporary.is_file()
                and temporary.stat().st_size == expected):
            temporary.replace(output)
            return output
        time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"{name}: could not download verified byte range {start}-{end}")


def stage_databases(directory, cache, mirrors, workers):
    directory.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    missing = {}
    for name, (size, digest) in ASSETS.items():
        path = directory / name
        if path.exists():
            if path.stat().st_size != size or _digest(path) != digest:
                raise ValueError(f"{name}: existing file does not match the pinned asset")
        else:
            missing[name] = list(range(0, size, BLOCK_SIZE))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        jobs = []
        for start in sorted({start for plan in missing.values() for start in plan}):
            for name, plan in missing.items():
                if start in plan:
                    size, _ = ASSETS[name]
                    jobs.append(executor.submit(_download_block, mirrors, cache, name,
                                                start, min(start + BLOCK_SIZE, size) - 1))
        for index, job in enumerate(as_completed(jobs), start=1):
            job.result()
            if index == 1 or index % 8 == 0 or index == len(jobs):
                print(f"Official SQLite blocks ready {index}/{len(jobs)}", flush=True)
    for name, plan in missing.items():
        with tempfile.NamedTemporaryFile(dir=directory, prefix=f".{name}.", delete=False) as tmp:
            temporary = Path(tmp.name)
            try:
                for start in plan:
                    tmp.write((cache / f"{name}.{start}").read_bytes())
                tmp.close()
                if _digest(temporary) != ASSETS[name][1]:
                    raise ValueError(f"{name}: assembled file checksum differs from the pinned asset")
                temporary.replace(directory / name)
            finally:
                temporary.unlink(missing_ok=True)
        for start in plan:
            (cache / f"{name}.{start}").unlink()
        print(f"Verified {name}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("cache", type=Path)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    if not 1 <= args.workers <= 16:
        parser.error("--workers must be between 1 and 16")
    mirrors = [os.environ.get("SKILL_RSI_HF_ENDPOINT", "https://hf-mirror.com"), "https://huggingface.co"]
    stage_databases(args.directory, args.cache, mirrors, args.workers)


if __name__ == "__main__":
    main()
