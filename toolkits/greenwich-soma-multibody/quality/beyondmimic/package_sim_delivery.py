"""Build a compact, auditable delivery from completed simulation runs."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def copy_file(source: Path, destination: Path) -> None:
    if source.is_file():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def copy_job(source: Path, destination: Path, outer_log: Path) -> dict:
    result = json.loads((source / "result.json").read_text(encoding="utf-8"))
    for name in (
        "result.json", "train-config.json", "environment-config.json",
        "best.pt", "last.pt", "video.log",
    ):
        copy_file(source / name, destination / name)
    copy_file(outer_log, destination / "training.stdout.log")
    if (source / "video").is_dir():
        shutil.copytree(source / "video", destination / "video", dirs_exist_ok=True)
    for event in (source / "logs").glob("events.out.tfevents.*"):
        copy_file(event, destination / "logs" / event.name)
    for evaluation in (source / "evaluations").glob("iteration-*/result.json"):
        copy_file(evaluation, destination / "evaluations" / evaluation.parent.name / "result.json")
    return {
        "name": source.name,
        "status": result.get("status"),
        "passed": result.get("passed"),
        "video_verified": result.get("video_verified"),
        "terminal_reason": result.get("terminal_reason"),
        "completed_iterations": result.get("completed_iterations"),
        "best_metrics": (result.get("best_evaluation") or {}).get("metrics"),
    }


def package_group(destination: Path, label: str, results: Path, motions: Path) -> list[dict]:
    if not results.is_dir():
        raise RuntimeError(f"results directory does not exist: {results}")
    if not motions.is_dir():
        raise RuntimeError(f"motions directory does not exist: {motions}")
    group = destination / label
    jobs = []
    for source in sorted(results.iterdir()):
        if not source.is_dir() or ".failed-attempt-" in source.name or not (source / "result.json").is_file():
            continue
        jobs.append(copy_job(source, group / "jobs" / source.name, results / f"{source.name}.log"))
    shutil.copytree(motions, group / "motions", dirs_exist_ok=True)
    for file in results.glob("queue-state*.json"):
        copy_file(file, group / "queue-state" / file.name)
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--group", action="append", nargs=3, required=True,
        metavar=("LABEL", "RESULTS_ROOT", "MOTIONS_ROOT"),
        help="Delivery group; repeat for each result set.",
    )
    args = parser.parse_args()

    destination = args.output.resolve()
    if destination.exists():
        raise RuntimeError(f"refusing to overwrite existing delivery: {destination}")
    destination.mkdir(parents=True)
    manifest: dict = {
        "schema": "greenwich.beyondmimic.delivery.v1",
        "contents": (
            "Best and last policies, verified videos, full result/evaluation JSON, TensorBoard curves, "
            "stdout logs, configs, and input motions. Intermediate checkpoints omitted."
        ),
        "groups": {},
    }
    for label, results, motions in args.group:
        if label in manifest["groups"]:
            raise RuntimeError(f"duplicate group label: {label}")
        manifest["groups"][label] = package_group(
            destination, label, Path(results).resolve(), Path(motions).resolve()
        )
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    summary = {
        group: {
            "jobs": len(jobs),
            "passed": sum(job["passed"] is True for job in jobs),
            "videos_verified": sum(job["video_verified"] is True for job in jobs),
        }
        for group, jobs in manifest["groups"].items()
    }
    (destination / "SUMMARY.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
