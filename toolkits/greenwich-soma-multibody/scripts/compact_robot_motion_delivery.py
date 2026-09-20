"""Create minimal per-motion robot NPZ files for delivery.

The payload intentionally matches the existing Greenwich compact reader and
publisher.  It excludes source SOMA arrays, decoder intermediates, FK/world
position copies, and physics/evaluation caches.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path

import numpy as np


SCHEMA = "greenwich.soma18.compact.v1"
REQUIRED = (
    "q",
    "rot6d",
    "root_t",
    "joint_names",
    "fps",
    "sole_surface_height_cm",
    "model_contact",
)


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def validate(path: Path, robot: str) -> tuple[int, float]:
    with np.load(path, allow_pickle=False) as data:
        if str(data["schema"].item()) != SCHEMA:
            raise ValueError(f"Unexpected schema in {path}")
        q = data[f"{robot}__q"]
        root_r = data[f"{robot}__root_rot6d"]
        root_t = data[f"{robot}__root_t_cm"]
        contact = data[f"{robot}__model_contact"]
        sole = data[f"{robot}__sole_height_cm"]
        fps = float(data["fps"])
        frames = int(q.shape[0])
        if q.dtype != np.float32 or q.ndim != 3 or q.shape[2] != 3:
            raise ValueError(f"Invalid q in {path}")
        if root_r.shape != (frames, 6) or root_t.shape != (frames, 3):
            raise ValueError(f"Invalid root trajectory in {path}")
        if contact.shape != (frames, 2) or sole.shape != (frames, 2):
            raise ValueError(f"Invalid foot payload in {path}")
        if not np.isfinite(q).all() or not np.isfinite(root_r).all() or not np.isfinite(root_t).all():
            raise ValueError(f"Non-finite trajectory in {path}")
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError(f"Invalid fps in {path}")
    return frames, fps


def convert(source: Path, destination: Path, robot: str) -> dict[str, object]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        frames, fps = validate(destination, robot)
        return {
            "clip": destination.stem,
            "status": "existing_verified",
            "frames": frames,
            "fps": fps,
            "bytes": destination.stat().st_size,
            "sha256": sha256(destination),
        }
    with np.load(source, allow_pickle=False) as data:
        missing = sorted(set(REQUIRED) - set(data.files))
        if missing:
            raise ValueError(f"Missing {missing} in {source}")
        q = np.asarray(data["q"], dtype=np.float32)
        root_rot6d = np.asarray(data["rot6d"][:, 0], dtype=np.float32)
        root_t = np.asarray(data["root_t"], dtype=np.float32)
        joint_names = np.asarray(data["joint_names"], dtype=str)
        fps = np.asarray(data["fps"], dtype=np.float32)
        sole = np.asarray(data["sole_surface_height_cm"], dtype=np.float32)
        contact = np.asarray(data["model_contact"], dtype=np.bool_)
    temporary = destination.with_suffix(".npz.partial")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            schema=np.asarray(SCHEMA),
            stem=np.asarray(destination.stem),
            fps=fps,
            **{
                f"{robot}__q": q,
                f"{robot}__root_rot6d": root_rot6d,
                f"{robot}__root_t_cm": root_t,
                f"{robot}__joint_names": joint_names,
                f"{robot}__sole_height_cm": sole,
                f"{robot}__model_contact": contact,
            },
        )
    os.replace(temporary, destination)
    frames, fps_value = validate(destination, robot)
    return {
        "clip": destination.stem,
        "status": "converted",
        "frames": frames,
        "fps": fps_value,
        "bytes": destination.stat().st_size,
        "sha256": sha256(destination),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--robot", default="h2")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    # Local backups and cloud workers do not have the same directory depth.
    # Discover recursively, then collapse pilot/full duplicates by clip id.
    # Prefer the full run when both are present; these duplicates are expected
    # for pilot motions that were subsequently included in the full batch.
    discovered = sorted(args.source.rglob("motion.npz"))
    by_clip: dict[str, Path] = {}
    for source in discovered:
        if source.parent.name != args.robot:
            continue
        clip = source.parents[1].name
        previous = by_clip.get(clip)
        if previous is None or ("full" in source.parts and "full" not in previous.parts):
            by_clip[clip] = source
    motions = [by_clip[clip] for clip in sorted(by_clip)]
    args.output.mkdir(parents=True, exist_ok=True)
    state_path = args.output / "compact-state.jsonl"
    with state_path.open("a", encoding="utf-8", buffering=1) as ledger:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(convert, source, args.output / f"{source.parents[1].name}.npz", args.robot): source
                for source in motions
            }
            for future in concurrent.futures.as_completed(futures):
                source = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    result = {"clip": source.parents[1].name, "status": "failed", "error": repr(exc)}
                ledger.write(json.dumps(result, sort_keys=True) + "\n")
                print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
