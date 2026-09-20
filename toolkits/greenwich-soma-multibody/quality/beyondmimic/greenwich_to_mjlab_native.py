"""Convert a Greenwich H2 trace to the local-motion schema used by mjlab."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
import torch
from scipy.optimize import least_squares, minimize
from scipy.spatial.transform import Rotation, Slerp


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--robots", type=Path, required=True)
    parser.add_argument("--robot", default="h2")
    parser.add_argument("--target-xml", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-fps", type=float, default=50.0)
    args = parser.parse_args()

    from alphamotion.embodiment.mjcf_build import build_merge
    from alphamotion.engine import constraints as constraints
    from alphamotion.engine.descriptor import build_from_mjcf

    robots = json.loads(args.robots.read_text(encoding="utf-8-sig"))
    robot = next(item for item in robots if item["name"] == args.robot)
    with np.load(args.source, allow_pickle=False) as archive:
        stored = {key: archive[key] for key in archive.files}
    spec, _, _, qnames, *_ = build_from_mjcf(robot["xml"], robot["body"])
    if not np.array_equal(stored["joint_names"], spec.joint_names):
        raise ValueError("Robot/trace joint order mismatch")
    q = np.asarray(stored["q"], float)
    root = np.asarray(stored["root_t"], float) / 100.0
    fps = float(stored["fps"])
    rotation = constraints.rot6d_to_matrix(torch.tensor(stored["rot6d"], dtype=torch.float64)).numpy()

    source_model = mujoco.MjModel.from_xml_path(robot["xml"]); source_data = mujoco.MjData(source_model)
    keep, frames = build_merge(source_model); base = keep[0]; frame = frames[base]
    free = [jid for jid in range(source_model.njnt) if source_model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE and source_model.jnt_bodyid[jid] == base]
    if len(free) != 1:
        raise ValueError("Expected one floating root")
    source_free_qpos = int(source_model.jnt_qposadr[free[0]])
    bindings = []
    for descriptor_joint, names in enumerate(qnames):
        for axis, name in enumerate(names):
            jid = mujoco.mj_name2id(source_model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise ValueError(f"Missing native joint {name}")
            bindings.append((int(source_model.jnt_qposadr[jid]), descriptor_joint, axis))
    basis = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    qpos = []
    for index in range(len(q)):
        source_data.qpos[:] = source_model.qpos0
        for address, descriptor_joint, axis in bindings:
            source_data.qpos[address] = q[index, descriptor_joint, axis]
        mujoco.mj_forward(source_model, source_data)
        delta = (basis @ rotation[index, 0] @ basis.T) @ source_data.xmat[frame].reshape(3, 3).T
        native_base = delta @ source_data.xmat[base].reshape(3, 3)
        source_data.qpos[source_free_qpos:source_free_qpos + 3] = basis @ root[index]
        source_data.qpos[source_free_qpos + 3:source_free_qpos + 7] = Rotation.from_matrix(native_base).as_quat(scalar_first=True)
        qpos.append(source_data.qpos.copy())
    source_qpos = np.asarray(qpos)
    # mjlab advances exactly one reference frame per control step. Its shared
    # H2 control period is 0.02 s. Resample the complete original trajectory,
    # not merely its fps label; keep a final sample hold through the last source
    # frame's support interval. N/fps defines clip support, (N-1)/fps the last
    # measured pose timestamp. Ceil gives no truncation and <one target-step
    # duration rounding. Original Greenwich archives remain untouched.
    source_fps = fps
    source_frame_count = len(source_qpos)
    source_duration_s = source_frame_count / source_fps
    source_times = np.arange(source_frame_count) / source_fps
    fps = args.target_fps
    if fps <= 0 or source_fps <= 0 or source_frame_count < 2:
        raise ValueError('Positive fps and at least two complete source frames required')
    target_frame_count = int(np.ceil(source_duration_s * fps - 1e-9))
    target_times = np.arange(target_frame_count) / fps
    sample_times = np.minimum(target_times, source_times[-1])
    resampled_qpos = np.tile(source_model.qpos0, (target_frame_count, 1))
    for axis in range(3):
        address = source_free_qpos + axis
        resampled_qpos[:, address] = np.interp(sample_times, source_times, source_qpos[:, address])
    quat_slice = slice(source_free_qpos + 3, source_free_qpos + 7)
    resampled_qpos[:, quat_slice] = Slerp(source_times, Rotation.from_quat(source_qpos[:, quat_slice], scalar_first=True))(sample_times).as_quat(scalar_first=True)
    for jid in range(source_model.njnt):
        if source_model.jnt_type[jid] != mujoco.mjtJoint.mjJNT_HINGE:
            continue
        address = int(source_model.jnt_qposadr[jid])
        angles = source_qpos[:, address]
        if not source_model.jnt_limited[jid]:
            angles = np.unwrap(angles)
        resampled_qpos[:, address] = np.interp(sample_times, source_times, angles)
    source_qpos = resampled_qpos

    # mjlab assumes one scalar coordinate per non-free articulation joint. The
    # H2 runtime XML therefore expands each passive loop ball into three hinge
    # coordinates while retaining the six closed-chain equality constraints.
    # Greenwich does not prescribe those passive loop coordinates, so they
    # start at zero and are solved by the simulator constraints.
    model = mujoco.MjModel.from_xml_path(str(args.target_xml or robot["xml"])); data = mujoco.MjData(model)
    target_free = [jid for jid in range(model.njnt) if model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE]
    if len(target_free) != 1:
        raise ValueError("Expected one target floating root")
    target_free_qpos = int(model.jnt_qposadr[target_free[0]])
    actuated_joints = {int(model.actuator_trnid[aid, 0]) for aid in range(model.nu)}
    # Ordinary serial-chain robots can have no XML actuators because mjlab adds
    # position actuators programmatically.  Such joints are not passive.  Passive
    # projection is only meaningful for closed-chain models with equality constraints.
    passive_joints = ([jid for jid in range(model.njnt)
                       if model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_HINGE
                       and jid not in actuated_joints]
                      if model.neq else [])
    passive_qpos = np.asarray([int(model.jnt_qposadr[jid]) for jid in passive_joints], dtype=int)
    passive_seed = model.qpos0[passive_qpos].copy()
    passive_lower = np.asarray([
        float(model.jnt_range[jid, 0]) if model.jnt_limited[jid] else -np.inf
        for jid in passive_joints
    ])
    passive_upper = np.asarray([
        float(model.jnt_range[jid, 1]) if model.jnt_limited[jid] else np.inf
        for jid in passive_joints
    ])
    passive_seed = np.clip(passive_seed, passive_lower, passive_upper)
    hinge_joints = [jid for jid in range(model.njnt) if model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_HINGE]
    equality_errors = []
    limit_errors = []
    numerical_limit_clips = []
    qpos = []
    for frame_index, source_pos in enumerate(source_qpos):
        target_pos = model.qpos0.copy()
        target_pos[target_free_qpos:target_free_qpos + 7] = source_pos[source_free_qpos:source_free_qpos + 7]
        for jid in range(model.njnt):
            if model.jnt_type[jid] != mujoco.mjtJoint.mjJNT_HINGE:
                continue
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
            source_jid = mujoco.mj_name2id(source_model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if source_jid >= 0 and source_model.jnt_type[source_jid] == mujoco.mjtJoint.mjJNT_HINGE:
                target_pos[int(model.jnt_qposadr[jid])] = source_pos[int(source_model.jnt_qposadr[source_jid])]
        if len(passive_qpos):
            def closure_residual(values: np.ndarray) -> np.ndarray:
                target_pos[passive_qpos] = values
                data.qpos[:] = target_pos
                mujoco.mj_forward(model, data)
                equality = data.efc_type[:data.nefc] == mujoco.mjtConstraint.mjCNSTR_EQUALITY
                return data.efc_pos[:data.nefc][equality].copy()
            # Choose the nearest feasible passive branch; residual-only solves
            # leave rod twist unconstrained and can introduce artificial turns.
            continuation = passive_seed.copy()
            if frame_index == 0:
                initial = least_squares(closure_residual, passive_seed,
                    bounds=(passive_lower,passive_upper),max_nfev=150,ftol=1e-12,xtol=1e-12,gtol=1e-12)
                passive_seed=initial.x
                continuation=initial.x.copy()
            solved = minimize(lambda x: .5 * np.sum((x-continuation)**2),
                              passive_seed, jac=lambda x: x-continuation,
                              method='SLSQP', bounds=list(zip(passive_lower,passive_upper)),
                              constraints={'type':'eq','fun':closure_residual},
                              options={'maxiter':150,'ftol':1e-12})
            if not solved.success and np.max(np.abs(closure_residual(solved.x))) > 1e-7:
                raise ValueError(f'Continuous passive projection failed frame {frame_index}: {solved.message}')
            passive_seed = solved.x
            target_pos[passive_qpos] = passive_seed
            equality_errors.append(float(np.max(np.abs(closure_residual(passive_seed)))))
        frame_limit_error = 0.0
        for jid in hinge_joints:
            if not model.jnt_limited[jid]:
                continue
            value = float(target_pos[int(model.jnt_qposadr[jid])])
            low, high = (float(item) for item in model.jnt_range[jid])
            excess = max(low - value, value - high, 0.0)
            frame_limit_error = max(frame_limit_error, excess)
            if excess > 1e-6:
                name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
                raise ValueError(
                    f"H2 joint-limit violation at frame {frame_index}, {name}: "
                    f"value={value:.9g}, range=[{low:.9g}, {high:.9g}], excess={excess:.9g}"
                )
            if excess:
                target_pos[int(model.jnt_qposadr[jid])] = np.clip(value, low, high)
                numerical_limit_clips.append(excess)
        limit_errors.append(frame_limit_error)
        qpos.append(target_pos)
    qpos = np.asarray(qpos)
    if equality_errors and max(equality_errors) > 1e-4:
        raise ValueError(f"Closed-chain projection failed: {max(equality_errors):.6g}")
    qvel = np.zeros((len(qpos), model.nv))
    for index in range(1, len(qpos)):
        mujoco.mj_differentiatePos(model, qvel[index], 1.0 / fps, qpos[index - 1], qpos[index])
    qvel[0] = qvel[1]
    if len(qvel) > 2:
        for index in range(1, len(qpos) - 1):
            mujoco.mj_differentiatePos(model, qvel[index], 2.0 / fps, qpos[index - 1], qpos[index + 1])

    # Correct generated passive velocities onto the equality tangent space,
    # keeping original motor and root velocities intact.
    passive_dofs=np.asarray([int(model.jnt_dofadr[j]) for j in passive_joints], dtype=int)
    velocity_residuals=[]
    if len(passive_dofs):
        for index in range(len(qpos)):
            data.qpos[:]=qpos[index]; data.qvel[:]=qvel[index]; mujoco.mj_forward(model,data)
            equality=data.efc_type[:data.nefc]==mujoco.mjtConstraint.mjCNSTR_EQUALITY
            if mujoco.mj_isSparse(model):
                dense=np.zeros((data.nefc,model.nv))
                for row in range(data.nefc):
                    start=int(data.efc_J_rowadr[row]);count=int(data.efc_J_rownnz[row])
                    dense[row,data.efc_J_colind[start:start+count]]=data.efc_J[start:start+count]
                jac=dense[equality]
            else:
                jac=data.efc_J.reshape(data.nefc,model.nv)[equality]
            qvel[index,passive_dofs] += np.linalg.lstsq(jac[:,passive_dofs],-jac@qvel[index],rcond=1e-10)[0]
            velocity_residuals.append(float(np.max(np.abs(jac@qvel[index]))))
    if velocity_residuals and max(velocity_residuals)>1e-7:
        raise ValueError(f'Closed-chain tangent velocity inconsistent: {max(velocity_residuals)}')

    joint_pos = np.column_stack([qpos[:, int(model.jnt_qposadr[jid])] for jid in hinge_joints])
    joint_vel = np.column_stack([qvel[:, int(model.jnt_dofadr[jid])] for jid in hinge_joints])
    body_pos = []; body_quat = []
    body_lin_vel=[]; body_ang_vel=[]
    jacp=np.zeros((3,model.nv)); jacr=np.zeros_like(jacp)
    for index,pos in enumerate(qpos):
        data.qpos[:] = pos; data.qvel[:]=qvel[index]; mujoco.mj_forward(model, data)
        body_pos.append(data.xpos[1:].copy()); body_quat.append(data.xquat[1:].copy())
        linear=[];angular=[]
        for bid in range(1,model.nbody):
            mujoco.mj_jacBody(model,data,jacp,jacr,bid)
            linear.append(jacp@qvel[index]);angular.append(jacr@qvel[index])
        body_lin_vel.append(linear);body_ang_vel.append(angular)
    body_pos = np.asarray(body_pos); body_quat = np.asarray(body_quat)
    body_lin_vel=np.asarray(body_lin_vel);body_ang_vel=np.asarray(body_ang_vel)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, joint_pos=joint_pos, joint_vel=joint_vel,
        body_pos_w=body_pos, body_quat_w=body_quat, body_lin_vel_w=body_lin_vel,
        body_ang_vel_w=body_ang_vel, fps=np.asarray(fps),
        source_fps=np.asarray(source_fps), source_frame_count=np.asarray(source_frame_count),
        source_duration_s=np.asarray(source_duration_s),
        source_last_pose_time_s=np.asarray(source_times[-1]),
        target_dt_s=np.asarray(1.0 / fps), target_duration_s=np.asarray(len(qpos) / fps),
        duration_rounding_s=np.asarray(len(qpos) / fps - source_duration_s),
        target_timestamps_s=target_times, source_sample_timestamps_s=sample_times,
        closed_chain_equality_max=np.asarray(max(equality_errors, default=0.0)),
        joint_limit_excess_max=np.asarray(max(limit_errors, default=0.0)),
        closed_chain_velocity_residual_max=np.asarray(max(velocity_residuals, default=0.0)),
        passive_projection_schema=np.asarray('nearest_feasible_continuation_tangent_v1'),
        joint_limit_numerical_clip_max=np.asarray(max(numerical_limit_clips, default=0.0)),
        joint_names=np.asarray([mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid) for jid in hinge_joints]),
        body_names=np.asarray([mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) for bid in range(1, model.nbody)]))
    print(json.dumps({"frames": len(qpos), "joints": joint_pos.shape[1], "bodies": body_pos.shape[1], "fps": fps, "output": str(args.output)}))


if __name__ == "__main__":
    main()
