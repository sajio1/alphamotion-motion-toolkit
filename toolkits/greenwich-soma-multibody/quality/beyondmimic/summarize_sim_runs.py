"""Summarize one or more simulation queues without machine-specific paths."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path


def read_json(path: Path) -> dict | list:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def job_name(job: dict) -> str:
    if "name" in job:
        return str(job["name"])
    if "clip" in job and "robot" in job:
        return f"{job['clip']}__{job['robot']}"
    raise KeyError(f"queue job has no name or clip+robot fields: {job}")


def summarize(label: str, results: Path, queue_pattern: str) -> dict:
    jobs: dict[str, dict] = {}
    queue_files = [Path(path) for path in sorted(glob.glob(queue_pattern))]
    if not queue_files:
        raise RuntimeError(f"queue glob matched nothing: {queue_pattern}")
    for queue in queue_files:
        content = read_json(queue)
        if not isinstance(content, list):
            raise RuntimeError(f"queue is not a JSON list: {queue}")
        for job in content:
            jobs[job_name(job)] = job

    rows = []
    for name in sorted(jobs):
        result = read_json(results / name / "result.json")
        if not isinstance(result, dict):
            result = {}
        status = result.get("status", "pending")
        verified = result.get("video_verified") is True
        if status == "training_finished" and verified:
            state = "complete_verified"
        elif status in {"training", "initializing"}:
            state = "running"
        else:
            state = "pending_or_failed"
        rows.append(
            {
                "name": name,
                "state": state,
                "status": status,
                "iterations": result.get("completed_iterations"),
                "video_verified": result.get("video_verified"),
                "passed": result.get("passed"),
            }
        )
    return {
        "label": label,
        "jobs": len(rows),
        "complete_verified": sum(row["state"] == "complete_verified" for row in rows),
        "running": sum(row["state"] == "running" for row in rows),
        "pending_or_failed": sum(row["state"] == "pending_or_failed" for row in rows),
        "passed": sum(row["passed"] is True for row in rows if row["state"] == "complete_verified"),
        "running_jobs": [row for row in rows if row["state"] == "running"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--group", action="append", nargs=3, required=True,
        metavar=("LABEL", "RESULTS_ROOT", "QUEUE_GLOB"),
        help="Queue group; repeat as needed.",
    )
    parser.add_argument("--json", type=Path, help="Optional path for machine-readable output")
    args = parser.parse_args()

    summaries = [summarize(label, Path(results).resolve(), pattern) for label, results, pattern in args.group]
    for item in summaries:
        print(
            f"{item['label']}: total={item['jobs']} complete_verified={item['complete_verified']} "
            f"running={item['running']} pending_or_failed={item['pending_or_failed']} passed={item['passed']}"
        )
        for job in item["running_jobs"]:
            print(f"  RUNNING {job['name']} iterations={job['iterations']}")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(summaries, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
