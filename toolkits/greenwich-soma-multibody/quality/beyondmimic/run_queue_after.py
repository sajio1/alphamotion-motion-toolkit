"""Start a queue only after an existing queue has reached a terminal state."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def write(path: Path, value: dict) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temp.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait-state", type=Path, required=True)
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--envs", type=int, default=4096)
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--sim9-root", type=Path, default=Path.home() / "sim9")
    parser.add_argument("--mjlab-root", type=Path, default=Path.home() / "mjlab")
    parser.add_argument("--video-width", type=int, default=1280)
    parser.add_argument("--video-height", type=int, default=720)
    parser.add_argument("--video-crf", type=int, default=18)
    args = parser.parse_args()
    sim9_root = args.sim9_root.resolve()
    mjlab_root = args.mjlab_root.resolve()
    args.results_root.mkdir(parents=True, exist_ok=True)
    state_path = args.results_root / "after-state.json"
    state = {
        "status": "waiting_for_current_queue",
        "started_at": time.time(),
        "wait_state": str(args.wait_state),
        "jobs": str(args.jobs),
    }
    write(state_path, state)
    while True:
        try:
            current = json.loads(args.wait_state.read_text(encoding="utf-8"))
            status = current.get("status")
        except (OSError, json.JSONDecodeError):
            status = None
        if status in {"complete", "complete_with_failures"}:
            break
        time.sleep(args.poll_seconds)

    state.update(status="running", launched_at=time.time(), prior_queue_status=status)
    write(state_path, state)
    command = [
        sys.executable,
        str(sim9_root / "sim9_queue.py"),
        "--jobs", str(args.jobs),
        "--results-root", str(args.results_root),
        "--envs", str(args.envs),
        "--iterations", str(args.iterations),
        "--sim9-root", str(sim9_root),
        "--mjlab-root", str(mjlab_root),
        "--video-width", str(args.video_width),
        "--video-height", str(args.video_height),
        "--video-crf", str(args.video_crf),
    ]
    env = os.environ.copy()
    env.update(
        PYTHONPATH=f"{sim9_root}:{mjlab_root / 'src'}",
        SIM9_ROOT=str(sim9_root),
        MJLAB_ROOT=str(mjlab_root),
        MUJOCO_GL="egl",
    )
    returncode = subprocess.run(command, cwd=mjlab_root, env=env).returncode
    state.update(status="complete" if returncode == 0 else "failed", returncode=returncode, finished_at=time.time())
    write(state_path, state)
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
