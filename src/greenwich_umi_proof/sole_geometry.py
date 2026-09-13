from __future__ import annotations

"""Foot-sole geometry derived once from the vendor MJCF visual meshes.

The lower-body solver needs, for every frame, the world position of the part
of each foot that actually touches the floor.  Anchoring the ankle joint is not
enough: a pitched foot moves its sole while the ankle stays put, so any sliding
metric computed from ankle positions is wrong by construction.

This module expresses the visual mesh vertices owned by each foot endpoint in
that endpoint's own descriptor frame (Greenwich Y-up basis, centimetres).  From
then on the lowest sole point, the sole reference point and the sole clearance
are cheap numpy operations on the descriptor FK output, without touching
MuJoCo again and without any robot-specific constants.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np


# MuJoCo Z-up metres -> Greenwich Y-up centimetres: (x, y, z) -> (y, z, x).
_MUJOCO_TO_GREENWICH = np.asarray(
    [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]], np.float64)


@dataclass(frozen=True)
class FootSole:
    role: str
    joint: int
    vertices_local_cm: np.ndarray   # [V,3] in the endpoint joint frame
    sole_point_local_cm: np.ndarray  # [3] centroid of the lowest vertex patch
    neutral_drop_cm: float           # endpoint joint height above its sole


@dataclass(frozen=True)
class SoleSample:
    lowest_y_cm: np.ndarray      # [T] lowest vertex height per frame
    sole_point_cm: np.ndarray    # [T,3] world sole reference point per frame


class FootSoleModel:
    """Per-foot sole geometry in descriptor joint frames."""

    def __init__(self, feet: dict[str, FootSole], report: dict):
        self.feet = feet
        self.report = report

    @property
    def roles(self) -> tuple[str, ...]:
        return tuple(self.feet)

    @property
    def joints(self) -> np.ndarray:
        return np.asarray([self.feet[r].joint for r in self.feet], np.int64)

    def sample(
        self,
        position_world_cm: np.ndarray,
        rotation_world: np.ndarray,
    ) -> dict[str, SoleSample]:
        """Lowest sole height and sole reference point per foot and frame."""

        position = np.asarray(position_world_cm, np.float64)
        rotation = np.asarray(rotation_world, np.float64)
        result: dict[str, SoleSample] = {}
        for role, foot in self.feet.items():
            joint_position = position[:, foot.joint]
            joint_rotation = rotation[:, foot.joint]
            world_y = (
                np.einsum("tj,vj->tv", joint_rotation[:, 1, :],
                          foot.vertices_local_cm)
                + joint_position[:, 1:2]
            )
            sole_point = (
                np.einsum("tij,j->ti", joint_rotation,
                          foot.sole_point_local_cm)
                + joint_position
            )
            result[role] = SoleSample(world_y.min(axis=1), sole_point)
        return result

    def endpoint_target_from_sole(
        self,
        role: str,
        sole_point_world_cm: np.ndarray,
        rotation_world: np.ndarray,
    ) -> np.ndarray:
        """Endpoint joint position that places the sole point at a target."""

        foot = self.feet[role]
        return (
            np.asarray(sole_point_world_cm, np.float64)
            - np.einsum("tij,j->ti", np.asarray(rotation_world, np.float64),
                        foot.sole_point_local_cm)
        )


def _neutral_descriptor_fk(spec, rest) -> tuple[np.ndarray, np.ndarray]:
    """Root-relative neutral joint positions/rotations from the descriptor."""

    parents = np.asarray(spec.parents, np.int64)
    offsets = np.asarray(spec.rest_offsets, np.float64)
    rest_rotation = np.asarray(rest, np.float64)
    position = np.zeros_like(offsets)
    rotation = np.zeros((len(parents), 3, 3), np.float64)
    for joint in range(len(parents)):
        parent = int(parents[joint])
        if parent < 0:
            rotation[joint] = np.eye(3)
            position[joint] = 0.0
        else:
            rotation[joint] = rotation[parent] @ rest_rotation[joint]
            position[joint] = position[parent] + rotation[parent] @ offsets[joint]
    return position, rotation


def build_foot_sole_model(
    xml: str | Path,
    spec,
    rest,
    roles,
    joints,
    *,
    patch_tolerance_cm: float = 0.3,
) -> FootSoleModel:
    """Read each foot's visual meshes into its descriptor endpoint frame.

    Only MJCF topology and mesh vertices are used.  The joint that owns each
    visual mesh is resolved by walking the MuJoCo body tree up to the nearest
    body that is also a descriptor joint, exactly like the viewer does.
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

    feet: dict[str, FootSole] = {}
    report: dict[str, object] = {
        "algorithm": "mjcf_visual_mesh_sole_in_descriptor_frame_v1",
        "robot_or_action_specific_parameters": False,
        "feet": {},
    }
    for role, joint in zip(roles, np.asarray(joints, np.int64)):
        joint = int(joint)
        vertices_world: list[np.ndarray] = []
        bodies: set[str] = set()
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
            world_m = local @ rotation.T + np.asarray(data.geom_xpos[geom_id])
            vertices_world.append(world_m)
            bodies.add(mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, body))
        if not vertices_world:
            raise ValueError(
                f"no visual mesh is owned by foot endpoint {joint_names[joint]!r}")
        world_m = np.concatenate(vertices_world, axis=0)
        world_cm = world_m @ _MUJOCO_TO_GREENWICH.T * 100.0
        joint_world_cm = root_world_cm + neutral_position[joint]
        local_cm = (world_cm - joint_world_cm) @ neutral_rotation[joint]
        lowest = float(local_cm[:, 1].min())
        patch = local_cm[local_cm[:, 1] <= lowest + float(patch_tolerance_cm)]
        sole_point = patch.mean(axis=0)
        sole_point[1] = lowest
        feet[str(role)] = FootSole(
            role=str(role),
            joint=joint,
            vertices_local_cm=local_cm,
            sole_point_local_cm=sole_point,
            neutral_drop_cm=-lowest,
        )
        report["feet"][str(role)] = {
            "endpoint_joint": joint_names[joint],
            "mjcf_bodies": sorted(bodies),
            "vertex_count": int(len(local_cm)),
            "neutral_drop_cm": -lowest,
            "sole_patch_vertex_count": int(len(patch)),
            "sole_point_local_cm": sole_point.tolist(),
            "sole_extent_local_cm": np.ptp(patch, axis=0).tolist(),
        }
    return FootSoleModel(feet, report)
