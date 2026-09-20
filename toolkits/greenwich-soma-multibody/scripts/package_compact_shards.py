"""Pack verified compact robot motions into deterministic tar.zst shards."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tarfile
import time
from pathlib import Path


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--files-per-shard", type=int, default=5000)
    parser.add_argument("--zstd", default="zstd")
    args = parser.parse_args()

    files = sorted(args.source.glob("*.npz"))
    args.output.mkdir(parents=True, exist_ok=True)
    state: dict[str, object] = {
        "schema": "greenwich.compact50.delivery.state.v1",
        "source": str(args.source),
        "prefix": args.prefix,
        "file_count": len(files),
        "files_per_shard": args.files_per_shard,
        "parts": {},
        "status": "packing",
    }
    state_path = args.output / f"{args.name}-state.json"
    snapshot_path = args.output / f"{args.name}-snapshot.json"
    snapshot_path.write_text(
        json.dumps(
            {
                "schema": "greenwich.compact50.delivery.snapshot.v1",
                "count": len(files),
                "files_per_shard": args.files_per_shard,
                "paths": [f"{args.prefix}/{path.name}" for path in files],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    for index in range(0, len(files), args.files_per_shard):
        group = files[index : index + args.files_per_shard]
        part = index // args.files_per_shard
        stem = f"{args.name}-{part:04d}"
        archive = args.output / f"{stem}.tar.zst"
        temporary_tar = args.output / f".{stem}.tar.partial"
        temporary_zst = args.output / f".{stem}.tar.zst.partial"
        started = time.time()
        with tarfile.open(temporary_tar, "w") as tar:
            for path in group:
                tar.add(path, arcname=f"{args.prefix}/{path.name}", recursive=False)
        subprocess.run(
            [args.zstd, "-1", "-T0", "-q", "-f", str(temporary_tar), "-o", str(temporary_zst)],
            check=True,
        )
        subprocess.run([args.zstd, "-q", "-t", str(temporary_zst)], check=True)
        temporary_zst.replace(archive)
        temporary_tar.unlink()
        state["parts"][archive.name] = {
            "file_count": len(group),
            "bytes": archive.stat().st_size,
            "sha256": digest(archive),
            "started_unix_s": started,
            "completed_unix_s": time.time(),
        }
        state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
        print(json.dumps({"part": archive.name, **state["parts"][archive.name]}), flush=True)

    state["status"] = "complete"
    state["finished_unix_s"] = time.time()
    state["total_bytes"] = sum(part["bytes"] for part in state["parts"].values())
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
