"""Stream compact robot NPZ delivery shards from a remote conversion node."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import time
from pathlib import Path


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", required=True)
    parser.add_argument("--node", type=int, required=True)
    parser.add_argument("--key", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--remote-user", default="ubuntu")
    parser.add_argument("--remote-root", required=True)
    parser.add_argument(
        "--compact-dir",
        help="Path relative to remote root (default: production-node-N/compact50)",
    )
    parser.add_argument("--files-per-shard", type=int, default=5000)
    args = parser.parse_args()
    args.destination.mkdir(parents=True, exist_ok=True)
    ssh = [
        "ssh", "-o", "BatchMode=yes", "-o", "ServerAliveInterval=30",
        "-o", "ServerAliveCountMax=6", "-i", str(args.key),
        f"{args.remote_user}@{args.host}",
    ]
    compact_dir = args.compact_dir or f"production-node-{args.node}/compact50"
    remote_root = shlex.quote(args.remote_root.rstrip("/"))
    compact_dir_quoted = shlex.quote(compact_dir)
    command = (
        f"cd {remote_root} && find {compact_dir_quoted} -maxdepth 1 -type f "
        "-name '*.npz' -printf '%p\\0' | sort -z"
    )
    listing = subprocess.run(ssh + [command], capture_output=True, check=True).stdout
    paths = [item.decode("utf-8") for item in listing.split(b"\0") if item]
    snapshot_path = args.destination / f"node-{args.node}-compact50-snapshot.json"
    snapshot = {
        "schema": "greenwich.compact50.delivery.snapshot.v1",
        "host": args.host,
        "node": args.node,
        "created_unix_s": time.time(),
        "count": len(paths),
        "files_per_shard": args.files_per_shard,
        "paths": paths,
    }
    atomic_json(snapshot_path, snapshot)
    state_path = args.destination / f"node-{args.node}-compact50-state.json"
    state = (
        json.loads(state_path.read_text(encoding="utf-8"))
        if state_path.exists()
        else {"schema": "greenwich.compact50.delivery.state.v1", "parts": {}, "status": "running"}
    )
    zstd = shutil.which("zstd")
    if not zstd:
        raise RuntimeError("zstd executable not found")
    groups = [paths[i:i + args.files_per_shard] for i in range(0, len(paths), args.files_per_shard)]
    for index, group in enumerate(groups):
        name = f"h2-compact50-node-{args.node}-{index:04d}.tar.zst"
        output = args.destination / name
        previous = state["parts"].get(name)
        if previous and output.exists() and output.stat().st_size == previous["bytes"] and digest(output) == previous["sha256"]:
            continue
        partial = output.with_suffix(output.suffix + ".partial")
        partial.unlink(missing_ok=True)
        remote = (
            f"set -o pipefail; cd {remote_root} && "
            "tar --null --verbatim-files-from -T - -cf - | zstd -1 -T2 -q"
        )
        started = time.time()
        with partial.open("wb") as target:
            process = subprocess.Popen(ssh + [remote], stdin=subprocess.PIPE, stdout=target)
            assert process.stdin is not None
            process.stdin.write(b"\0".join(item.encode("utf-8") for item in group) + b"\0")
            process.stdin.close()
            if process.wait():
                raise RuntimeError(f"Remote archive failed: {name}")
        subprocess.run([zstd, "-q", "-t", str(partial)], check=True)
        os.replace(partial, output)
        state["parts"][name] = {
            "file_count": len(group),
            "bytes": output.stat().st_size,
            "sha256": digest(output),
            "started_unix_s": started,
            "completed_unix_s": time.time(),
        }
        atomic_json(state_path, state)
    state["status"] = "complete"
    state["finished_unix_s"] = time.time()
    state["file_count"] = len(paths)
    state["total_bytes"] = sum(part["bytes"] for part in state["parts"].values())
    atomic_json(state_path, state)


if __name__ == "__main__":
    main()
