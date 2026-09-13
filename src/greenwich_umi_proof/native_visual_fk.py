"""Render realized motor trajectories with the original MuJoCo kinematic tree.

Greenwich may merge co-located joints. A merged endpoint rotation cannot
drive all intermediate meshes: each motor still rotates its own vendor link.
This module unpacks named motor angles and never fits or changes the motion.
"""
import numpy as np


def sample_native_geoms(model, q, qnames, geom_ids, *, root_body,
                        root_frame, root_position_m, root_rotation):
    import mujoco as mj

    q = np.asarray(q, np.float64)
    p = np.asarray(root_position_m, np.float64)
    r = np.asarray(root_rotation, np.float64)
    if (q.ndim != 3 or q.shape[1:] != (len(qnames), 3)
            or p.shape != (len(q), 3) or r.shape != (len(q), 3, 3)
            or not all(np.isfinite(x).all() for x in (q, p, r))):
        raise ValueError("Invalid native visual FK trajectory")
    slots = []
    seen = set()
    for j, names in enumerate(qnames):
        for s, name in enumerate(names):
            jid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, name)
            if jid < 0 or jid in seen or s >= 3:
                raise ValueError(f"Invalid/duplicate motor binding: {name}")
            if model.jnt_type[jid] != mj.mjtJoint.mjJNT_HINGE:
                raise ValueError(f"Unsupported descriptor motor type: {name}")
            seen.add(jid)
            slots.append((j, s, int(model.jnt_qposadr[jid])))
    data = mj.MjData(model)
    ids = np.asarray(geom_ids, np.int64)
    positions = np.empty((len(q), len(ids), 3), np.float64)
    rotations = np.empty((len(q), len(ids), 3, 3), np.float64)
    for t in range(len(q)):
        data.qpos[:] = model.qpos0
        for j, s, adr in slots:
            data.qpos[adr] = q[t, j, s]
        mj.mj_forward(model, data)
        delta = r[t] @ data.xmat[root_frame].reshape(3, 3).T
        positions[t] = p[t] + (data.geom_xpos[ids] - data.xpos[root_body]) @ delta.T
        rotations[t] = delta @ data.geom_xmat[ids].reshape(-1, 3, 3)
    return positions, rotations


def descriptor_visual_fk(xml, body, joint_names, q, geom_ids,
                         root_position_m, root_rotation):
    import mujoco as mj
    from alphamotion.engine.descriptor import build_from_mjcf
    from alphamotion.embodiment.mjcf_build import build_merge

    spec, _dof, _rest, qnames, _xml = build_from_mjcf(xml, body)
    if list(spec.joint_names) != list(joint_names):
        raise ValueError("Visual model and saved motor descriptor differ; rebuild required")
    model = mj.MjModel.from_xml_path(str(xml))
    keep, frames = build_merge(model)
    return sample_native_geoms(
        model, q, qnames, geom_ids, root_body=keep[0],
        root_frame=frames[keep[0]], root_position_m=root_position_m,
        root_rotation=root_rotation)
