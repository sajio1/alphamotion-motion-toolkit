from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=Path, required=True)
    args = parser.parse_args()
    jobs = json.loads(args.jobs.read_text(encoding="utf-8"))
    bad = []
    for job in jobs:
        path = Path(job["motion"])
        if not path.is_file():
            bad.append({"clip": job["clip"], "reason": "missing"})
            continue
        with np.load(path, allow_pickle=False) as archive:
            if float(archive["fps"]) != 50.0:
                bad.append({"clip": job["clip"], "reason": "fps"})
            if archive["joint_pos"].ndim != 2 or archive["joint_pos"].shape[1] != 29:
                bad.append({"clip": job["clip"], "reason": "joint_shape"})
            if not all(np.isfinite(archive[key]).all() for key in (
                "joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"
            )):
                bad.append({"clip": job["clip"], "reason": "non_finite"})
    print(json.dumps({
        "jobs": len(jobs),
        "variants": dict(Counter(job["benchmark_variant"] for job in jobs)),
        "bad": bad,
    }))
    raise SystemExit(bool(bad))


if __name__ == "__main__":
    main()
