"""Fixed-seed full-clip/suffix evaluation for native A3/G1/Booster policies."""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", choices=("a3", "g1", "booster_t1_29"), required=True)
    parser.add_argument("--motion", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--episodes", type=int, default=64)
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--video-width", type=int, default=1280)
    parser.add_argument("--video-height", type=int, default=720)
    parser.add_argument("--video-crf", type=int, default=18)
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    result = {"status": "evaluating", "passed": False, "arguments": vars(args), "started_at": time.time()}
    env = None
    try:
        import numpy as np
        import torch
        from mjlab.envs import ManagerBasedRlEnv
        from mjlab.rl import RslRlVecEnvWrapper
        from mjlab.tasks.registry import load_runner_cls
        from mjlab.tasks.tracking.config.g1.rl_cfg import unitree_g1_tracking_ppo_runner_cfg
        from mjlab.utils.lab_api.math import quat_error_magnitude
        from native_robot_cfg import native_tracking_env_cfg, robot_eval_names

        torch.set_num_threads(2)
        cfg = native_tracking_env_cfg(args.robot)
        agent = unitree_g1_tracking_ppo_runner_cfg()
        cfg.seed = 20260919
        agent.seed = 20260919
        cfg.scene.num_envs = args.episodes
        cfg.auto_reset = True
        cfg.observations["actor"].enable_corruption = False
        cfg.events.pop("push_robot", None)
        cfg.commands["motion"].motion_file = args.motion
        cfg.commands["motion"].sampling_mode = "start"
        with np.load(args.motion) as archive:
            duration = float(archive["source_duration_s"])
            total = len(archive["joint_pos"])
        cfg.episode_length_s = duration + 1.0
        cfg.viewer.width = args.video_width
        cfg.viewer.height = args.video_height
        cfg.viewer.max_extra_envs = 0
        env = ManagerBasedRlEnv(
            cfg=cfg, device="cuda:0", render_mode="rgb_array" if args.video else None
        )
        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
        runner_cls = load_runner_cls("Mjlab-Tracking-Flat-Unitree-G1")
        runner = runner_cls(wrapped, asdict(agent), device="cuda:0")
        runner.load(args.checkpoint, load_cfg={"actor": True}, strict=True, map_location="cuda:0")
        policy = runner.get_inference_policy(device="cuda:0")
        command = env.command_manager.get_term("motion")
        robot = env.scene["robot"]
        eval_names = robot_eval_names(args.robot)
        wrists = [command.cfg.body_names.index(name) for name in eval_names["wrists"]]
        feet = [command.cfg.body_names.index(name) for name in eval_names["feet"]]
        supplied = list(range(len(command.robot.joint_names)))

        n = args.episodes
        phases = [0.0] * 32 + [v for v in (0.2, 0.4, 0.6, 0.8) for _ in range(8)] if n == 64 else [0.0] * n
        starts = torch.tensor([int(v * (total - 1)) for v in phases], device="cuda:0")
        with torch.inference_mode():
            for frame in sorted(set(starts.cpu().tolist())):
                command.reset_to_frame(torch.where(starts == frame)[0], frame)
            env.sim.forward()
            command.update_relative_body_poses()
            obs = wrapped.get_observations()
            active = torch.ones(n, dtype=torch.bool, device="cuda:0")
            success = torch.zeros_like(active)
            counts = torch.zeros(n, device="cuda:0")
            sums = {key: torch.zeros(n, device="cuda:0") for key in ("body_m", "joint_rmse_rad", "anchor_ori_rad", "wrist_m", "wrist_ori_rad")}
            wrist_samples = []
            torque_samples = []
            support_count = torch.zeros(n, device="cuda:0")
            slip_count = torch.zeros(n, device="cuda:0")
            writer = None
            video_frames = 0
            if args.video:
                import imageio.v2 as imageio
                writer = imageio.get_writer(
                    str(out / "policy.mp4"),
                    fps=50,
                    codec="libx264",
                    output_params=["-preset", "medium", "-crf", str(args.video_crf), "-pix_fmt", "yuv420p"],
                )
            for step in range(total):
                if not bool(active.any()):
                    break
                body_error = torch.linalg.vector_norm(command.body_pos_relative_w - command.robot_body_pos_w, dim=-1)
                wrist_error = body_error[:, wrists]
                values = {
                    "body_m": body_error.mean(-1),
                    "joint_rmse_rad": ((command.joint_pos[:, supplied] - command.robot_joint_pos[:, supplied]) ** 2).mean(-1).sqrt(),
                    "anchor_ori_rad": quat_error_magnitude(command.anchor_quat_w, command.robot_anchor_quat_w),
                    "wrist_m": wrist_error.mean(-1),
                    "wrist_ori_rad": quat_error_magnitude(command.body_quat_relative_w[:, wrists], command.robot_body_quat_w[:, wrists]).mean(-1),
                }
                counts += active
                for key, value in values.items():
                    sums[key] += torch.where(active, value, 0)
                wrist_samples.append(wrist_error[active].flatten().cpu())
                torque_samples.append(robot.data.actuator_force[active].abs().flatten().cpu())
                ref_speed = torch.linalg.vector_norm(command.body_lin_vel_w[:, feet, :2], dim=-1)
                support = ref_speed < 0.15
                robot_speed = torch.linalg.vector_norm(command.robot_body_lin_vel_w[:, feet, :2], dim=-1)
                support_count += (support & active[:, None]).sum(-1)
                slip_count += (support & (robot_speed > 0.15) & active[:, None]).sum(-1)
                if writer and bool(active[0]):
                    writer.append_data(env.render())
                    video_frames += 1
                reached = active & (step >= total - 1 - starts)
                success |= reached
                active &= ~reached
                if not bool(active.any()):
                    break
                obs, _, dones, _ = wrapped.step(policy(obs))
                active &= ~dones.bool()
            if writer:
                writer.close()
                with imageio.get_reader(str(out / "policy.mp4")) as reader:
                    decoded = sum(1 for _ in reader)
                if decoded != video_frames:
                    raise RuntimeError("video decode frame count mismatch")
                result.update(video_verified=True, decoded_frames=decoded)

            mean = {key: float((value / counts.clamp(min=1)).mean()) for key, value in sums.items()}
            mean["wrist_p95_m"] = float(torch.quantile(torch.cat(wrist_samples), 0.95))
            mean["support_proxy_slip_fraction"] = float(slip_count.sum() / support_count.sum().clamp(min=1))
            mean["success_rate"] = float(success.float().mean())
            mean["mean_duration_coverage"] = float((counts / (total - starts)).clamp(max=1).mean())
            torques = torch.cat(torque_samples)
            mean["actuator_force_abs_p95_nm"] = float(torch.quantile(torques, 0.95))
            mean["actuator_force_abs_max_nm"] = float(torques.max())
            limits = torch.as_tensor(env.sim.mj_model.actuator_forcerange[:, 1], dtype=torch.float32)
            finite_limits = limits[limits > 0]
            mean["actuator_force_limit_min_nm"] = float(finite_limits.min()) if len(finite_limits) else None

            passed = mean["success_rate"] >= 0.95 and mean["body_m"] <= 0.12
            passed = passed and mean["joint_rmse_rad"] <= 0.35 and mean["anchor_ori_rad"] <= 0.4
            passed = passed and mean["support_proxy_slip_fraction"] <= 0.1
            if n == 64:
                passed = passed and int(success[:32].sum()) >= 31
                passed = passed and all(int(success[32 + i * 8:40 + i * 8].sum()) >= 7 for i in range(4))
            result.update(
                status="complete",
                passed=bool(passed),
                metrics=mean,
                source_duration_s=duration,
                video_frames=video_frames,
                evaluation_semantics="full clip and fixed suffix phases; first failure ends a trial; no reset splicing",
                torque_semantics="MuJoCo position-actuator force in actuation space",
                quality_score=mean["body_m"] / 0.12 + mean["joint_rmse_rad"] / 0.35 + mean["anchor_ori_rad"] / 0.4,
            )
    except Exception as exc:
        result.update(status="failed", error=repr(exc), traceback=traceback.format_exc())
    finally:
        result["finished_at"] = time.time()
        (out / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        if env is not None:
            env.close()
    print(json.dumps({k: v for k, v in result.items() if k != "traceback"}), flush=True)


if __name__ == "__main__":
    main()
