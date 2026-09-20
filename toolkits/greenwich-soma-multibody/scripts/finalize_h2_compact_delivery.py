"""Build and verify the unified H2 50 Hz delivery metadata."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def clip_from_member(member: str) -> str:
    return Path(member).stem


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--delivery", type=Path, required=True)
    parser.add_argument("--remaining-selection", type=Path, required=True)
    parser.add_argument("--old-cloud-manifest", type=Path, required=True)
    parser.add_argument("--old-compact-ledger", type=Path, required=True)
    args = parser.parse_args()

    metadata = args.delivery / "metadata"
    shards = args.delivery / "shards"
    metadata.mkdir(parents=True, exist_ok=True)
    states = sorted(metadata.glob("*-state.json"))
    snapshots = sorted(metadata.glob("*-snapshot.json"))
    if len(states) != 3 or len(snapshots) != 3:
        raise RuntimeError(f"Expected three states and snapshots, got {len(states)} / {len(snapshots)}")

    shard_records: list[dict[str, object]] = []
    expected_count = 0
    for state_path in states:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("status") != "complete":
            raise RuntimeError(f"Incomplete state: {state_path}")
        expected_count += int(state["file_count"])
        for name, record in state["parts"].items():
            path = shards / name
            if not path.is_file():
                raise FileNotFoundError(path)
            if path.stat().st_size != int(record["bytes"]):
                raise RuntimeError(f"Size mismatch: {path}")
            actual = sha256(path)
            if actual != record["sha256"]:
                raise RuntimeError(f"SHA256 mismatch: {path}")
            shard_records.append(
                {
                    "name": name,
                    "file_count": int(record["file_count"]),
                    "bytes": int(record["bytes"]),
                    "sha256": actual,
                }
            )

    paths: list[str] = []
    for snapshot_path in snapshots:
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
        if int(snapshot["count"]) != len(snapshot["paths"]):
            raise RuntimeError(f"Snapshot count mismatch: {snapshot_path}")
        paths.extend(snapshot["paths"])
    clips = [clip_from_member(path) for path in paths]
    unique = set(clips)
    if len(paths) != expected_count or len(unique) != expected_count:
        raise RuntimeError(
            f"Combined count/duplicate failure: paths={len(paths)} unique={len(unique)} expected={expected_count}"
        )

    remaining_metadata: dict[str, dict[str, object]] = {}
    with args.remaining_selection.open("r", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            remaining_metadata[row["clip_id"]] = row
    new_clips = {clip for clip, path in zip(clips, paths) if path.startswith("production-node-")}
    unresolved_new = sorted(set(remaining_metadata) - new_clips)

    old_cloud = json.loads(args.old_cloud_manifest.read_text(encoding="utf-8"))
    old_failed = sorted(old_cloud["failed_clips"])
    failure_path = metadata / "failures.jsonl"
    with failure_path.open("w", encoding="utf-8") as stream:
        for clip in old_failed:
            stream.write(json.dumps({"clip_id": clip, "batch": "old30000", "status": "conversion_failed"}) + "\n")
        for clip in unresolved_new:
            row = remaining_metadata[clip]
            stream.write(
                json.dumps(
                    {
                        "clip_id": clip,
                        "batch": "remaining",
                        "status": "conversion_failed_or_unresolved",
                        "source_archive_member": row.get("source_archive_member"),
                        "duration_s_metadata": row.get("duration_s_metadata"),
                    }
                )
                + "\n"
            )

    old_duration_s = 0.0
    with args.old_compact_ledger.open("r", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("status") in {"converted", "existing_verified"}:
                old_duration_s += float(row["frames"]) / float(row["fps"])
    new_duration_s = sum(
        float(remaining_metadata[clip]["duration_s_metadata"])
        for clip in new_clips
        if clip in remaining_metadata
    )

    manifest = {
        "schema": "greenwich.h2.compact50.complete.v1",
        "robot": "h2",
        "fps": 50,
        "successful_motion_count": expected_count,
        "old_batch_success_count": int(old_cloud["success_count"]),
        "remaining_success_count": len(new_clips),
        "failure_count": len(old_failed) + len(unresolved_new),
        "old_batch_failure_count": len(old_failed),
        "remaining_failure_or_unresolved_count": len(unresolved_new),
        "duration_hours_estimate": (old_duration_s + new_duration_s) / 3600.0,
        "duration_basis": "exact frames/fps for old batch; source metadata duration for remaining batch",
        "shard_count": len(shard_records),
        "total_compressed_bytes": sum(int(row["bytes"]) for row in shard_records),
        "payload": [
            "h2__q float32",
            "h2__root_rot6d float32",
            "h2__root_t_cm float32",
            "h2__joint_names",
            "h2__sole_height_cm float32",
            "h2__model_contact bool",
            "fps float32",
        ],
        "shards": sorted(shard_records, key=lambda row: str(row["name"])),
    }
    (args.delivery / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    readme = f"""Greenwich H2 50 Hz Complete Compact Delivery

Successful motions: {expected_count:,}
Known failed or unresolved motions: {manifest['failure_count']:,}
Estimated duration: {manifest['duration_hours_estimate']:.2f} hours
Compressed payload: {manifest['total_compressed_bytes'] / 1_000_000_000:.2f} GB
Shards: {manifest['shard_count']}

All motions use one schema and one 50 Hz rate. The old 30k batch and the later
remaining batch are not delivered in different formats. Source SOMA arrays,
decoder intermediates, duplicated FK/world-position arrays, physics caches,
videos, and temporary files are intentionally excluded.

Every shard's byte size and SHA-256 digest are recorded in MANIFEST.json.
Known conversion failures are listed in metadata/failures.jsonl rather than
being silently omitted.

Extraction example:
  zstd -d -c shards/<name>.tar.zst | tar -xf -
"""
    (args.delivery / "README.txt").write_text(readme, encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
