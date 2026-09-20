"""Train one native A3/G1/Booster BeyondMimic policy with audited early stop."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
import traceback
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("WANDB_MODE", "disabled")


def runtime_roots() -> tuple[Path, Path]:
    sim9_root = Path(os.environ.get("SIM9_ROOT", Path.home() / "sim9")).resolve()
    mjlab_root = Path(os.environ.get("MJLAB_ROOT", Path.home() / "mjlab")).resolve()
    return sim9_root, mjlab_root


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", choices=("a3", "g1", "booster_t1_29"), required=True)
    parser.add_argument("--motion", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--envs", type=int, required=True)
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--formal", action="store_true")
    parser.add_argument("--resume-checkpoint")
    parser.add_argument("--resume-iteration", type=int, default=0)
    parser.add_argument("--video-width", type=int, default=1280)
    parser.add_argument("--video-height", type=int, default=720)
    parser.add_argument("--video-crf", type=int, default=18)
    args = parser.parse_args()
    sim9_root, mjlab_root = runtime_roots()
    out = Path(args.output)
    if args.resume_checkpoint:
        out.mkdir(parents=True, exist_ok=True)
    else:
        out.mkdir(parents=True, exist_ok=False)
    prior_result = {}
    if args.resume_checkpoint and (out / "result.json").exists():
        prior_result = json.loads((out / "result.json").read_text(encoding="utf-8"))
        shutil.copy2(out / "result.json", out / f"result.before-resume-{int(time.time())}.json")
    result = dict(prior_result)
    result.update(
        arguments=vars(args),
        status="initializing",
        resumed_at=time.time() if args.resume_checkpoint else None,
    )
    result.setdefault("started_at", time.time())
    result.setdefault("timings", [])
    if (out / "stop.request").exists():
        (out / "stop.request").unlink()
    env = None

    def save() -> None:
        temp = out / "result.tmp"
        temp.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
        temp.replace(out / "result.json")

    save()
    try:
        import numpy as np
        import torch
        from mjlab.envs import ManagerBasedRlEnv
        from mjlab.rl import RslRlVecEnvWrapper
        from mjlab.tasks.registry import load_runner_cls
        from mjlab.tasks.tracking.config.g1.rl_cfg import unitree_g1_tracking_ppo_runner_cfg
        from native_robot_cfg import native_tracking_env_cfg

        torch.set_num_threads(4)
        cfg = native_tracking_env_cfg(args.robot)
        agent = unitree_g1_tracking_ppo_runner_cfg()
        cfg.seed = 20260919
        agent.seed = 20260919
        cfg.scene.num_envs = args.envs
        cfg.commands["motion"].motion_file = args.motion
        with np.load(args.motion) as archive:
            duration = float(archive["source_duration_s"])
        cfg.episode_length_s = max(cfg.episode_length_s, duration + 0.1)
        agent.max_iterations = args.iterations
        agent.save_interval = 500
        agent.logger = "tensorboard"
        agent.upload_model = False
        result.update(
            physics_dt=cfg.sim.mujoco.timestep,
            policy_dt=cfg.sim.mujoco.timestep * cfg.decimation,
            samples_per_iteration=args.envs * agent.num_steps_per_env,
            iteration_semantics="one 24-policy-step rollout per environment plus five PPO epochs",
            reward_policy="unmodified shared BeyondMimic tracking reward; hardware interface adaptation only",
            python_torch=str(torch.__version__),
        )
        (out / "train-config.json").write_text(json.dumps(asdict(agent), indent=2, default=str), encoding="utf-8")
        (out / "environment-config.json").write_text(json.dumps(asdict(cfg), indent=2, default=str), encoding="utf-8")

        env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
        runner_cls = load_runner_cls("Mjlab-Tracking-Flat-Unitree-G1")
        runner = runner_cls(wrapped, asdict(agent), log_dir=str(out / "logs"), device="cuda:0")
        if args.resume_checkpoint:
            runner.load(args.resume_checkpoint, map_location="cuda:0")
            # The milestone filename records completed iterations, while RSL-RL
            # stores a zero-based loop index. Continue at the next iteration.
            runner.current_learning_iteration = args.resume_iteration
            result.update(
                resumed_from=str(args.resume_checkpoint),
                resume_iteration=args.resume_iteration,
            )
        result["model_dimensions"] = {
            name: int(getattr(env.sim.mj_model, name))
            for name in ("nq", "nv", "nu", "nbody", "ngeom")
        }
        original_log = runner.logger.log
        pending = None
        evaluations = list(prior_result.get("evaluations", []))
        best = None
        stable = False
        prior_best = prior_result.get("best_evaluation")
        if prior_best and (out / "best.pt").exists():
            prior_metrics = prior_best.get("metrics", {})
            best = ((
                prior_metrics.get("success_rate", 0),
                prior_metrics.get("mean_duration_coverage", 0),
                -prior_best.get("quality_score", float("inf")),
            ), prior_best)

        def consume_evaluation(wait: bool = False) -> None:
            nonlocal pending, best, stable
            if pending is None:
                return
            process, path, checkpoint, iteration, handle = pending
            if wait:
                process.wait()
            if process.poll() is None:
                return
            handle.close()
            pending = None
            target = path / "result.json"
            value = json.loads(target.read_text()) if target.exists() else {"status": "failed", "error": "evaluation result missing"}
            value = {key: val for key, val in value.items() if key not in ("traceback",)}
            value.update(completed_iterations=iteration, checkpoint=str(checkpoint))
            evaluations.append(value)
            if value.get("status") == "complete":
                rank = (value["metrics"]["success_rate"], value["metrics"].get("mean_duration_coverage", 0), -value["quality_score"])
                if best is None or rank > best[0]:
                    shutil.copy2(checkpoint, out / "best.pt")
                    best = (rank, value)
            qualifying = evaluations[-3:]
            if (
                len(qualifying) == 3
                and qualifying[-1].get("completed_iterations", 0) >= 1500
                and all(item.get("passed") for item in qualifying)
            ):
                scores = [item["quality_score"] for item in qualifying]
                relative_range = (max(scores) - min(scores)) / max(abs(scores[0]), 1e-8)
                stable = qualifying[-1]["completed_iterations"] - qualifying[0]["completed_iterations"] >= 500 and relative_range < 0.02
            result["evaluations"] = evaluations
            result["stable_success_plateau"] = stable
            if stable:
                (out / "stop.request").touch()

        def launch_evaluation(iteration: int, checkpoint: Path) -> None:
            nonlocal pending
            path = out / "evaluations" / f"iteration-{iteration}"
            path.parent.mkdir(exist_ok=True)
            handle = (out / f"eval-{iteration}.log").open("w")
            cmd = [
                str(mjlab_root / ".venv" / "bin" / "python"),
                str(sim9_root / "sim9_evaluate.py"), "--robot", args.robot,
                "--motion", args.motion, "--checkpoint", str(checkpoint), "--output", str(path),
            ]
            pending = (subprocess.Popen(cmd, stdout=handle, stderr=subprocess.STDOUT), path, checkpoint, iteration, handle)

        if args.formal:
            os.environ["MJLAB_STOP_REQUEST"] = str(out / "stop.request")
        torch.cuda.synchronize()
        last = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()

        def log(*log_args, **kwargs):
            nonlocal last
            torch.cuda.synchronize()
            now = time.perf_counter()
            entry = {
                "completed_iterations": int(kwargs["it"]) + 1,
                "rollout_s": kwargs["collect_time"],
                "ppo_update_s": kwargs["learn_time"],
                "synchronized_iteration_s": now - last,
                "losses": {key: float(value) for key, value in kwargs["loss_dict"].items()},
            }
            result["timings"].append(entry)
            result["timings"] = result["timings"][-64:]
            result.update(
                status="training",
                completed_iterations=entry["completed_iterations"],
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            )
            if args.formal:
                consume_evaluation()
                iteration = entry["completed_iterations"]
                if iteration % 250 == 0 or iteration == args.iterations:
                    checkpoint = out / f"checkpoint-{iteration}.pt"
                    runner.save(str(checkpoint))
                    if pending is None:
                        launch_evaluation(iteration, checkpoint)
            save()
            original_log(*log_args, **kwargs)
            last = time.perf_counter()

        runner.logger.log = log
        remaining_iterations = max(0, args.iterations - args.resume_iteration)
        result["remaining_iterations_at_resume"] = remaining_iterations
        runner.learn(num_learning_iterations=remaining_iterations, init_at_random_ep_len=False)
        runner.save(str(out / "last.pt"))
        if args.formal:
            consume_evaluation(wait=True)
        if best is None:
            shutil.copy2(out / "last.pt", out / "best.pt")
        result.update(
            status="training_finished",
            terminal_reason="stable_success_plateau" if stable else "iteration_cap",
            best_evaluation=best[1] if best else None,
            passed=bool(best and best[1].get("passed")),
            finished_at=time.time(),
        )
        env.close()
        env = None
        video_out = out / "video"
        handle = (out / "video.log").open("w")
        cmd = [
            str(mjlab_root / ".venv" / "bin" / "python"),
            str(sim9_root / "sim9_evaluate.py"), "--robot", args.robot,
            "--motion", args.motion, "--checkpoint", str(out / "best.pt"), "--output", str(video_out),
            "--episodes", "1", "--video",
            "--video-width", str(args.video_width),
            "--video-height", str(args.video_height),
            "--video-crf", str(args.video_crf),
        ]
        video = subprocess.run(cmd, stdout=handle, stderr=subprocess.STDOUT)
        handle.close()
        video_result = video_out / "result.json"
        result["video_process_returncode"] = video.returncode
        result["video_verified"] = video_result.exists() and bool(json.loads(video_result.read_text()).get("video_verified"))
        if not result["video_verified"]:
            result.update(status="artifact_failed", error="policy video missing or decode verification failed")
    except Exception as exc:
        result.update(status="failed", error=repr(exc), traceback=traceback.format_exc(), finished_at=time.time())
    finally:
        save()
        if env is not None:
            env.close()
    print(json.dumps({key: value for key, value in result.items() if key != "traceback"}), flush=True)
    return 0 if result["status"] == "training_finished" else 1


if __name__ == "__main__":
    raise SystemExit(main())
