"""Convert BONES-SEED G1 CSV retargeting output to mjlab's motion schema.

The public CSV contract is 120 Hz, centimetres, degrees, and extrinsic XYZ
root Euler angles.  Resampling preserves physical time; it never changes the
playback speed.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation, Slerp


ROOT_TRANSLATION = ["root_translateX", "root_translateY", "root_translateZ"]
ROOT_ROTATION = ["root_rotateX", "root_rotateY", "root_rotateZ"]


def read_csv(path: Path) -> tuple[list[str], np.ndarray]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        rows = [[float(value) for value in row] for row in reader if row]
    values = np.asarray(rows, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] != len(header):
        raise ValueError(f"Invalid CSV shape for {path}: {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError(f"Non-finite values in {path}")
    return header, values


def resample_qpos(
    model: mujoco.MjModel,
    qpos: np.ndarray,
    free_qpos: int,
    source_fps: float,
    target_fps: float,
) -> tuple[np.ndarray, dict[str, float | int]]:
    source_count = len(qpos)
    source_duration = source_count / source_fps
    source_times = np.arange(source_count, dtype=np.float64) / source_fps
    target_count = int(np.ceil(source_duration * target_fps - 1e-9))
    target_times = np.arange(target_count, dtype=np.float64) / target_fps
    sample_times = np.minimum(target_times, source_times[-1])
    result = np.tile(model.qpos0, (target_count, 1))
    for axis in range(3):
        address = free_qpos + axis
        result[:, address] = np.interp(sample_times, source_times, qpos[:, address])
    quat = slice(free_qpos + 3, free_qpos + 7)
    result[:, quat] = Slerp(
        source_times, Rotation.from_quat(qpos[:, quat], scalar_first=True)
    )(sample_times).as_quat(scalar_first=True)
    for jid in range(model.njnt):
        if model.jnt_type[jid] != mujoco.mjtJoint.mjJNT_HINGE:
            continue
        address = int(model.jnt_qposadr[jid])
        angles = qpos[:, address]
        if not model.jnt_limited[jid]:
            angles = np.unwrap(angles)
        result[:, address] = np.interp(sample_times, source_times, angles)
    return result, {
        "source_frame_count": source_count,
        "source_duration_s": source_duration,
        "source_last_pose_time_s": float(source_times[-1]),
        "target_frame_count": target_count,
        "target_duration_s": target_count / target_fps,
        "duration_rounding_s": target_count / target_fps - source_duration,
    }


def convert(source: Path, xml: Path, output: Path, source_fps: float, target_fps: float) -> dict:
    header, values = read_csv(source)
    columns = {name: index for index, name in enumerate(header)}
    required = ["Frame", *ROOT_TRANSLATION, *ROOT_ROTATION]
    missing = [name for name in required if name not in columns]
    if missing:
        raise ValueError(f"Missing root columns in {source}: {missing}")

    frame = values[:, columns["Frame"]]
    if not np.array_equal(frame, np.arange(len(frame), dtype=np.float64)):
        raise ValueError(f"Non-contiguous Frame column in {source}")

    model = mujoco.MjModel.from_xml_path(str(xml))
    data = mujoco.MjData(model)
    free = [jid for jid in range(model.njnt) if model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE]
    if len(free) != 1:
        raise ValueError("Expected exactly one floating root")
    free_qpos = int(model.jnt_qposadr[free[0]])
    hinges = [jid for jid in range(model.njnt) if model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_HINGE]
    expected = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid) + "_dof" for jid in hinges]
    absent = [name for name in expected if name not in columns]
    extras = [name for name in header if name.endswith("_dof") and name not in expected]
    if absent or extras:
        raise ValueError(f"G1 joint interface mismatch: absent={absent}, extras={extras}")

    qpos = np.tile(model.qpos0, (len(values), 1))
    qpos[:, free_qpos : free_qpos + 3] = values[:, [columns[name] for name in ROOT_TRANSLATION]] / 100.0
    # BONES-SEED declares extrinsic XYZ Euler angles in degrees. SciPy's
    # lowercase 'xyz' convention is extrinsic; MuJoCo stores quaternions wxyz.
    qpos[:, free_qpos + 3 : free_qpos + 7] = Rotation.from_euler(
        "xyz", values[:, [columns[name] for name in ROOT_ROTATION]], degrees=True
    ).as_quat(scalar_first=True)
    for jid, column in zip(hinges, expected):
        qpos[:, int(model.jnt_qposadr[jid])] = np.deg2rad(values[:, columns[column]])

    limit_excess = []
    for jid in hinges:
        if not model.jnt_limited[jid]:
            continue
        address = int(model.jnt_qposadr[jid])
        low, high = model.jnt_range[jid]
        limit_excess.append(np.maximum(low - qpos[:, address], qpos[:, address] - high).clip(min=0))
    max_limit_excess = float(max((item.max() for item in limit_excess), default=0.0))
    if max_limit_excess > 1e-5:
        raise ValueError(
            f"CSV exceeds the benchmark G1 joint limits by {max_limit_excess:.6g} rad; "
            "refusing to silently clamp a benchmark reference"
        )

    qpos, timing = resample_qpos(model, qpos, free_qpos, source_fps, target_fps)
    qvel = np.zeros((len(qpos), model.nv), dtype=np.float64)
    for index in range(1, len(qpos)):
        mujoco.mj_differentiatePos(model, qvel[index], 1.0 / target_fps, qpos[index - 1], qpos[index])
    qvel[0] = qvel[1]
    if len(qvel) > 2:
        for index in range(1, len(qpos) - 1):
            mujoco.mj_differentiatePos(model, qvel[index], 2.0 / target_fps, qpos[index - 1], qpos[index + 1])

    joint_pos = np.column_stack([qpos[:, int(model.jnt_qposadr[jid])] for jid in hinges])
    joint_vel = np.column_stack([qvel[:, int(model.jnt_dofadr[jid])] for jid in hinges])
    body_pos, body_quat, body_lin_vel, body_ang_vel = [], [], [], []
    jacp = np.zeros((3, model.nv)); jacr = np.zeros_like(jacp)
    for index, position in enumerate(qpos):
        data.qpos[:] = position
        data.qvel[:] = qvel[index]
        mujoco.mj_forward(model, data)
        body_pos.append(data.xpos[1:].copy())
        body_quat.append(data.xquat[1:].copy())
        linear, angular = [], []
        for bid in range(1, model.nbody):
            mujoco.mj_jacBody(model, data, jacp, jacr, bid)
            linear.append(jacp @ qvel[index])
            angular.append(jacr @ qvel[index])
        body_lin_vel.append(linear)
        body_ang_vel.append(angular)

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        body_pos_w=np.asarray(body_pos),
        body_quat_w=np.asarray(body_quat),
        body_lin_vel_w=np.asarray(body_lin_vel),
        body_ang_vel_w=np.asarray(body_ang_vel),
        fps=np.asarray(target_fps),
        source_fps=np.asarray(source_fps),
        target_dt_s=np.asarray(1.0 / target_fps),
        source_frame_count=np.asarray(timing["source_frame_count"]),
        source_duration_s=np.asarray(timing["source_duration_s"]),
        source_last_pose_time_s=np.asarray(timing["source_last_pose_time_s"]),
        target_duration_s=np.asarray(timing["target_duration_s"]),
        duration_rounding_s=np.asarray(timing["duration_rounding_s"]),
        joint_limit_excess_max=np.asarray(max_limit_excess),
        source_schema=np.asarray("bones_seed_g1_csv_120hz_cm_deg_extrinsic_xyz"),
        joint_names=np.asarray([mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid) for jid in hinges]),
        body_names=np.asarray([mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) for bid in range(1, model.nbody)]),
    )
    return {"source": str(source), "output": str(output), **timing, "joint_limit_excess_max": max_limit_excess}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--xml", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-fps", type=float, default=120.0)
    parser.add_argument("--target-fps", type=float, default=50.0)
    args = parser.parse_args()
    if (args.source is None) == (args.source_dir is None):
        parser.error("Provide exactly one of --source or --source-dir")
    sources = [args.source] if args.source else sorted(args.source_dir.glob("*.csv"))
    results = []
    for source in sources:
        target = args.output if args.source else args.output / source.stem / "motion_50hz.npz"
        result = convert(source, args.xml, target, args.source_fps, args.target_fps)
        results.append(result)
        print(json.dumps(result))
    if args.source_dir:
        (args.output / "conversion-manifest.json").write_text(
            json.dumps({"schema": "bones_seed_g1_to_mjlab_v1", "motions": results}, indent=2),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
