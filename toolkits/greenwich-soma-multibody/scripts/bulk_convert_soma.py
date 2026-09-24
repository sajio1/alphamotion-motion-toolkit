"""Resumable stream conversion of a SOMA archive into compact 18-body NPZ shards.

The archive is scanned once.  Only the current batch is extracted, converted via
the public invoke.ps1 entrypoint, and removed.  Progress is file-backed so no
port, scheduler, or chat notification is required.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tarfile
import time

import pandas as pd


GROUND_PATTERN = re.compile(
    r"crawl|all[_ ]fours|lying|laying|prone|supine|roll(?:ing)?|kneel|"
    r"handstand|meditat|sit[_ ]on[_ ]floor|floor[_ ]sit|balled[_ ]up|"
    r"death[_ ]stomach|faint",
    re.IGNORECASE,
)
GROUND_FIELDS = (
    "move_name", "category", "content_name", "content_short_description_2",
    "content_type_of_movement", "content_body_position",
)


def atomic_json(path: Path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def selection_rows(metadata: Path):
    frame = pd.read_parquet(metadata)
    structured = frame[list(GROUND_FIELDS)].fillna("").astype(str).agg(" ".join, axis=1)
    excluded = structured.str.contains(GROUND_PATTERN)
    rows = []
    for source_index, (_, item) in enumerate(frame.loc[~excluded].iterrows(), 1):
        path = str(item["move_soma_uniform_path"])
        rows.append({
            "index": source_index,
            "move_name": str(item["move_name"]),
            "path": path,
            "frames": int(item["move_duration_frames"]),
            "duration_s": int(item["move_duration_frames"]) / 120.0048,
            "package": str(item["package"]),
            "category": str(item["category"]),
            "movement_type": str(item["content_type_of_movement"]),
            "semantic_type": str(item["content_short_description_2"] or item["content_name"]),
            "description": str(item["content_natural_desc_4"] or item["content_natural_desc_1"]),
            "stem": Path(path).stem,
        })
    return len(frame), int(excluded.sum()), rows


def load_completed(ledger: Path):
    completed = set()
    if ledger.exists():
        for line in ledger.read_text(encoding="utf-8").splitlines():
            try:
                item = json.loads(line)
                if item.get("status") == "complete": completed.add(item["stem"])
            except json.JSONDecodeError:
                continue
    return completed


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--metadata", type=Path, required=True)
    p.add_argument("--archive", type=Path, required=True)
    p.add_argument("--bind", type=Path, required=True)
    p.add_argument("--toolkit", type=Path, required=True)
    p.add_argument("--robots", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--powershell", help="Use invoke.ps1 on Windows; omit for direct portable Python execution")
    p.add_argument("--python", type=Path, required=True)
    p.add_argument("--repo", type=Path, help="AlphaMotion repository for direct Python execution")
    p.add_argument("--pipeline", type=Path, help="Pipeline runtime for direct Python execution")
    p.add_argument("--ffmpeg", help="FFmpeg executable for direct Python execution")
    p.add_argument("--cache", type=Path,
                   help="Shared AlphaMotion semantic cache; defaults to OUTPUT/cache")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--fps", type=float, default=30)
    p.add_argument("--contact-config", type=Path)
    p.add_argument("--minimum-free-gb", type=float, default=20)
    p.add_argument("--shard-count", type=int, default=1)
    p.add_argument("--shard-index", type=int, default=0)
    a = p.parse_args()
    if a.cache is None: a.cache = a.output / "cache"
    if a.batch_size < 1: p.error("batch-size must be positive")
    if a.shard_count < 1: p.error("shard-count must be positive")
    if not 0 <= a.shard_index < a.shard_count: p.error("shard-index must be in [0, shard-count)")
    a.output.mkdir(parents=True, exist_ok=True)
    stage = a.output / "_staging"
    source = stage / "outputs" / "seed-100-contact" / "source"
    source.mkdir(parents=True, exist_ok=True)
    shutil.copy2(a.bind, source / "soma_base_skel_minimal.bvh")
    shutil.copy2(a.robots, a.output / "robots.json")
    total, excluded_count, rows = selection_rows(a.metadata)
    global_selected_count = len(rows)
    rows = [row for row in rows if row["index"] % a.shard_count == a.shard_index]
    robot_count = len(json.loads(a.robots.read_text(encoding="utf-8-sig")))
    atomic_json(a.output / "manifest.json", {
        "schema": "greenwich.soma18.bulk.v1", "source_rows": total,
        "excluded_ground_system_rows": excluded_count,
        "global_selected_rows": global_selected_count, "selected_rows": len(rows),
        "shard_count": a.shard_count, "shard_index": a.shard_index,
        "robot_count": robot_count,
        "fps": a.fps, "contact_iterations": 100,
        "upper_body_iterations": 100,
        "contact_config": json.loads(a.contact_config.read_text(encoding="utf-8-sig")) if a.contact_config else None,
        "ground_filter_regex": GROUND_PATTERN.pattern,
        "compact_npz": "q float32 + root rot6d float32 + root XYZ float32 + contacts; FK reconstructable",
    })
    ledger = a.output / "ledger.jsonl"
    completed = load_completed(ledger)
    pending = {row["path"]: row for row in rows if row["stem"] not in completed}
    started = time.time(); processed_at_start = len(completed); batch = []

    def progress(state, current=None, error=None):
        done = len(completed); elapsed = time.time() - started
        rate = (done - processed_at_start) / elapsed if elapsed > 0 else 0
        # A resumed item can have an old failure followed by a successful retry.
        # Count only each stem's latest state so stale startup failures do not
        # inflate the live failure total.
        latest_status = {}
        if ledger.exists():
            for line in ledger.read_text(encoding="utf-8").splitlines():
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if record.get("stem") and record.get("status"):
                    latest_status[record["stem"]] = record["status"]
        atomic_json(a.output / "progress.json", {
            "state": state, "total_source_rows": total, "excluded_ground_system_rows": excluded_count,
            "selected_motions": len(rows), "completed_motions": done,
            "completed_robot_conversions": done * robot_count,
            "total_robot_conversions": len(rows) * robot_count,
            "failed_motions": sum(status == "failed" for status in latest_status.values()),
            "current": current, "elapsed_this_run_s": elapsed, "motions_per_hour": rate * 3600,
            "eta_hours_at_current_rate": (len(rows) - done) / rate / 3600 if rate else None,
            "free_disk_gb": shutil.disk_usage(a.output).free / 1e9, "last_error": error,
            "updated_unix_s": time.time(),
        })

    def invoke(group):
        batch_id = min(x["index"] for x in group)
        shard = a.output / "shards" / f"{batch_id:06d}"
        shard.mkdir(parents=True, exist_ok=True)
        atomic_json(stage / "outputs" / "seed-100-contact" / "selection.json", group)
        indices=",".join(str(x["index"]) for x in group)
        if a.powershell:
            command = [a.powershell, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
                str(a.toolkit / "scripts" / "invoke.ps1"), "-Stage", "Generate",
                "-Indices", indices, "-DatasetWorkspace", str(stage), "-Robots", str(a.robots),
                "-Output", str(shard), "-Fps", str(a.fps), "-MaxSeconds", "3600",
                "-SkipPreview", "-Compact",
                "-SkipSummary", "-Cache", str(a.cache)]
            if a.contact_config:command.extend(["-ContactConfig", str(a.contact_config)])
        else:
            if not all((a.repo,a.pipeline,a.ffmpeg)):
                raise ValueError("Direct execution requires --repo, --pipeline and --ffmpeg")
            command = [str(a.python),str(a.toolkit/'scripts'/'run_soma_locomotion.py'),
                '--repo',str(a.repo),'--pipeline',str(a.pipeline),'--dataset-workspace',str(stage),
                '--robots',str(a.robots),'--output',str(shard),'--indices',indices,'--fps',str(a.fps),
                '--max-seconds','3600','--ffmpeg',a.ffmpeg,
                '--input-representation','soma77','--skip-preview','--compact','--cache',str(a.cache)]
            if a.contact_config:command.extend(['--contact-config',str(a.contact_config)])
        with (a.output / "worker.log").open("a", encoding="utf-8") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        produced = shard / "motions"
        missing = [item for item in group if not (produced / (item["stem"] + ".npz")).exists()]
        # A malformed source must not discard the rest of an amortized model batch.
        # Bisect failed groups until only the actual bad motion is recorded failed.
        if (result.returncode or missing) and len(group) > 1:
            midpoint = len(group) // 2
            invoke(group[:midpoint])
            invoke(group[midpoint:])
            return result.returncode
        with ledger.open("a", encoding="utf-8") as stream:
            for item in group:
                artifact = produced / (item["stem"] + ".npz")
                status = "complete" if artifact.exists() else "failed"
                record = {"status": status, "stem": item["stem"], "index": item["index"],
                          "artifact": str(artifact) if artifact.exists() else None,
                          "worker_exit_code": result.returncode, "time_unix_s": time.time()}
                stream.write(json.dumps(record) + "\n");stream.flush()
                if status == "complete": completed.add(item["stem"])
        return result.returncode

    progress("scanning_archive")
    try:
        with tarfile.open(a.archive, "r|gz") as archive:
            for member in archive:
                row = pending.get(member.name)
                if row is None or not member.isfile(): continue
                extracted = archive.extractfile(member)
                if extracted is None: continue
                target = source / (row["stem"] + ".bvh")
                with target.open("wb") as stream: shutil.copyfileobj(extracted, stream, 1024 * 1024)
                batch.append(row)
                if len(batch) < a.batch_size: continue
                if shutil.disk_usage(a.output).free / 1e9 < a.minimum_free_gb:
                    progress("stopped_low_disk", batch[0]["stem"]);return 2
                progress("converting", batch[0]["stem"])
                invoke(batch)
                for item in batch: (source / (item["stem"] + ".bvh")).unlink(missing_ok=True)
                batch.clear();progress("scanning_archive")
        if batch:
            progress("converting", batch[0]["stem"]);invoke(batch)
            for item in batch: (source / (item["stem"] + ".bvh")).unlink(missing_ok=True)
        missing=len(rows)-len(completed)
        progress("complete" if not missing else "archive_exhausted_with_missing")
        return int(bool(missing))
    except Exception as exc:
        progress("failed", error=repr(exc));raise


if __name__ == "__main__": raise SystemExit(main())
