"""Robot-independent endpoint geometry for load-bearing floor support.

Feet use their native visual meshes.  Distal endpoints that do not own a
visual mesh (for example a fixed wrist TCP on a handless robot) fall back to
the declared semantic endpoint itself.  The fallback is an interface fact,
not a robot- or action-specific offset.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np

from greenwich_umi_proof.sole_geometry import (
    FootSole,
    FootSoleModel,
    _MUJOCO_TO_GREENWICH,
    _neutral_descriptor_fk,
)


def build_endpoint_surface_model(xml: str | Path, spec, rest, roles, joints) -> FootSoleModel:
    """Express native endpoint surfaces in descriptor joint frames.

    This is the same geometry contract as ``build_foot_sole_model`` but it is
    valid for hands as well as feet.  A fixed TCP with no owned visual mesh is
    represented by its origin, which is the only geometry-neutral contact
    point available from that robot interface.
    """
    import mujoco as mj
    from alphamotion.viz.kinematics import visual_mesh_geom_ids

    model = mj.MjModel.from_xml_path(str(Path(xml).resolve()))
    data = mj.MjData(model)
    data.qpos[:] = model.qpos0
    mj.mj_forward(model, data)

    joint_names = [str(name) for name in spec.joint_names]
    name_to_index = {name: index for index, name in enumerate(joint_names)}
    body_parent = np.asarray(model.body_parentid, np.int64)

    def owner_joint(body: int) -> int | None:
        cursor = int(body)
        while cursor > 0:
            name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, cursor)
            if name in name_to_index:
                return name_to_index[name]
            cursor = int(body_parent[cursor])
        return None

    root_m = np.asarray(data.qpos[:3], np.float64) if model.nq >= 7 else np.zeros(3)
    root_world_cm = _MUJOCO_TO_GREENWICH @ root_m * 100.0
    neutral_position, neutral_rotation = _neutral_descriptor_fk(spec, rest)
    endpoints = {}
    report = {
        "algorithm": "mjcf_endpoint_surface_in_descriptor_frame_v1",
        "robot_or_action_specific_parameters": False,
        "surfaces": {},
    }
    for role, joint in zip(roles, np.asarray(joints, np.int64)):
        joint = int(joint)
        vertices_world = []
        bodies = set()
        for geom_id in visual_mesh_geom_ids(model):
            body = int(model.geom_bodyid[geom_id])
            if owner_joint(body) != joint:
                continue
            mesh_id = int(model.geom_dataid[geom_id])
            if mesh_id < 0:
                continue
            start = int(model.mesh_vertadr[mesh_id])
            count = int(model.mesh_vertnum[mesh_id])
            local = np.asarray(model.mesh_vert[start:start + count], np.float64)
            rotation = np.asarray(data.geom_xmat[geom_id], np.float64).reshape(3, 3)
            vertices_world.append(local @ rotation.T + np.asarray(data.geom_xpos[geom_id]))
            bodies.add(mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, body))
        if vertices_world:
            world_cm = np.concatenate(vertices_world, axis=0) @ _MUJOCO_TO_GREENWICH.T * 100.0
            joint_world_cm = root_world_cm + neutral_position[joint]
            local_cm = (world_cm - joint_world_cm) @ neutral_rotation[joint]
            lowest = float(local_cm[:, 1].min())
            patch = local_cm[local_cm[:, 1] <= lowest + .3]
            point = patch.mean(axis=0)
            point[1] = lowest
            source = "owned_visual_mesh"
        else:
            local_cm = np.zeros((1, 3), np.float64)
            patch = local_cm
            point = local_cm[0].copy()
            lowest = 0.0
            source = "semantic_endpoint_origin"
        endpoints[str(role)] = FootSole(
            role=str(role), joint=joint, vertices_local_cm=local_cm,
            sole_point_local_cm=point, neutral_drop_cm=-lowest,
        )
        report["surfaces"][str(role)] = {
            "endpoint_joint": joint_names[joint],
            "geometry_source": source,
            "mjcf_bodies": sorted(bodies),
            "vertex_count": int(len(local_cm)),
            "neutral_drop_cm": -lowest,
        }
    return FootSoleModel(endpoints, report)
