"""Instantiate a native robot tracking environment and advance a few policy steps."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", choices=("a3", "g1", "booster_t1_29", "all"), required=True)
    parser.add_argument("--motion")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--envs", type=int, default=1)
    parser.add_argument("--steps", type=int, default=2)
    args = parser.parse_args()

    import numpy as np
    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from native_robot_cfg import native_tracking_env_cfg

    robots = ("a3", "g1", "booster_t1_29") if args.robot == "all" else (args.robot,)
    results = []
    for robot in robots:
        motion = args.motion or str(
            ROOT / "motions" / "04_coordinated_dance" / robot / "motion_50hz.npz"
        )
        cfg = native_tracking_env_cfg(robot)
        cfg.scene.num_envs = args.envs
        cfg.commands["motion"].motion_file = motion
        cfg.events.pop("push_robot", None)
        cfg.observations["actor"].enable_corruption = False
        with np.load(motion) as archive:
            duration = float(archive["source_duration_s"])
        cfg.episode_length_s = max(cfg.episode_length_s, duration + 0.1)
        started = time.perf_counter()
        env = ManagerBasedRlEnv(cfg=cfg, device=args.device)
        build_s = time.perf_counter() - started
        obs = env.reset()[0]
        step_started = time.perf_counter()
        with torch.inference_mode():
            for _ in range(args.steps):
                action = torch.zeros(
                    (args.envs, env.action_manager.total_action_dim), device=args.device
                )
                obs, _, _, _, _ = env.step(action)
        elapsed = time.perf_counter() - step_started
        results.append({
            "robot": robot,
            "envs": args.envs,
            "steps": args.steps,
            "device": args.device,
            "build_s": build_s,
            "step_s": elapsed,
            "obs_shape": list(obs["actor"].shape),
            "action_dim": env.action_manager.total_action_dim,
            "model": {
                name: int(getattr(env.sim.mj_model, name))
                for name in ("nq", "nv", "nu", "nbody", "ngeom")
            },
        })
        env.close()
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
