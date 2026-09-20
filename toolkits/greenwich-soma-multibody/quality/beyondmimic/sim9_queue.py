"""Run an ordered, restart-safe set of native BeyondMimic jobs on one GPU."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def atomic_json(path: Path, value: object) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temp.replace(path)


def completed(output: Path) -> bool:
    result = output / "result.json"
    if not result.exists():
        return False
    try:
        value = json.loads(result.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return value.get("status") == "training_finished" and value.get("video_verified") is True


def resume_point(output: Path) -> tuple[Path, int] | None:
    checkpoints = []
    for path in output.glob("checkpoint-*.pt"):
        try:
            iteration = int(path.stem.split("-")[-1])
        except ValueError:
            continue
        checkpoints.append((iteration, path))
    if not checkpoints:
        return None
    iteration, path = max(checkpoints)
    return path, iteration


def failed_before_training(output: Path) -> bool:
    path = output / "result.json"
    if not path.exists():
        return False
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return result.get("status") == "failed" and not result.get("timings")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", required=True)
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--envs", type=int, default=4096)
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--state-file")
    parser.add_argument("--sim9-root", type=Path, default=Path.home() / "sim9")
    parser.add_argument("--mjlab-root", type=Path, default=Path.home() / "mjlab")
    parser.add_argument("--video-width", type=int, default=1280)
    parser.add_argument("--video-height", type=int, default=720)
    parser.add_argument("--video-crf", type=int, default=18)
    args = parser.parse_args()
    sim9_root = args.sim9_root.resolve()
    mjlab_root = args.mjlab_root.resolve()
    python = mjlab_root / ".venv" / "bin" / "python"
    if not python.is_file():
        parser.error(f"mjlab Python does not exist: {python}")
    if not (sim9_root / "sim9_worker.py").is_file():
        parser.error(f"sim9 worker does not exist: {sim9_root / 'sim9_worker.py'}")

    jobs = json.loads(Path(args.jobs).read_text(encoding="utf-8"))
    root = Path(args.results_root)
    root.mkdir(parents=True, exist_ok=True)
    state_path = Path(args.state_file) if args.state_file else root / "queue-state.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "started_at": time.time(),
        "status": "running",
        "jobs_file": args.jobs,
        "envs": args.envs,
        "iterations": args.iterations,
        "jobs": [],
    }
    atomic_json(state_path, state)

    for index, job in enumerate(jobs):
        name = f"{job['clip']}__{job['robot']}"
        output = root / name
        entry = {
            "index": index,
            "name": name,
            "robot": job["robot"],
            "motion": job["motion"],
            "output": str(output),
            "status": "pending",
        }
        state["jobs"].append(entry)
        atomic_json(state_path, state)

        if completed(output):
            entry.update(status="skipped_complete", finished_at=time.time())
            atomic_json(state_path, state)
            continue
        resume = resume_point(output) if output.exists() else None
        if output.exists() and resume is None and failed_before_training(output):
            archived = output.with_name(f"{output.name}.failed-attempt-{int(time.time())}")
            output.rename(archived)
            entry["archived_failed_attempt"] = str(archived)
        if output.exists() and resume is None:
            entry.update(status="blocked_existing_incomplete", finished_at=time.time())
            atomic_json(state_path, state)
            continue

        entry.update(status="running", started_at=time.time())
        atomic_json(state_path, state)
        log_path = root / f"{name}.log"
        command = [
            str(python),
            str(sim9_root / "sim9_worker.py"),
            "--robot", job["robot"],
            "--motion", job["motion"],
            "--output", str(output),
            "--envs", str(job.get("envs", args.envs)),
            "--iterations", str(args.iterations),
            "--formal",
            "--video-width", str(args.video_width),
            "--video-height", str(args.video_height),
            "--video-crf", str(args.video_crf),
        ]
        if resume is not None:
            checkpoint, resume_iteration = resume
            command.extend([
                "--resume-checkpoint", str(checkpoint),
                "--resume-iteration", str(resume_iteration),
            ])
            entry.update(
                status="resuming",
                resume_checkpoint=str(checkpoint),
                resume_iteration=resume_iteration,
            )
            atomic_json(state_path, state)
        env = os.environ.copy()
        env.update(
            PYTHONPATH=f"{sim9_root}:{mjlab_root / 'src'}",
            SIM9_ROOT=str(sim9_root),
            MJLAB_ROOT=str(mjlab_root),
            MUJOCO_GL="egl",
            TF_CPP_MIN_LOG_LEVEL="3",
        )
        with log_path.open("a" if resume else "w", encoding="utf-8") as handle:
            returncode = subprocess.run(
                command,
                cwd=mjlab_root,
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
            ).returncode
        entry.update(
            status="complete" if returncode == 0 and completed(output) else "failed",
            returncode=returncode,
            finished_at=time.time(),
        )
        atomic_json(state_path, state)

    failures = [job for job in state["jobs"] if job["status"] in {"failed", "blocked_existing_incomplete"}]
    state.update(status="complete_with_failures" if failures else "complete", finished_at=time.time())
    atomic_json(state_path, state)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
