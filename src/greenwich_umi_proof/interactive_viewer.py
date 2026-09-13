from __future__ import annotations
from pathlib import Path
import numpy as np
YUP_TO_ZUP = np.asarray([[0.,0.,1.],[1.,0.,0.],[0.,1.,0.]])
def stabilize_masked_rotation_islands(*args, **kwargs):
    raise ValueError("Display stabilization is excluded from this numerical conversion toolkit")


def _yup_rotation_to_zup(rotation: np.ndarray) -> np.ndarray:
    """Change a Greenwich Y-up orientation into the viewer's Z-up frame."""
    rotation = np.asarray(rotation, np.float64)
    return np.einsum(
        "ij,...jk,lk->...il", YUP_TO_ZUP, rotation, YUP_TO_ZUP)

def _rot6d_to_matrix_numpy(rot6d: np.ndarray) -> np.ndarray:
    """Greenwich's continuous 6-D rotation representation, in NumPy."""
    value = np.asarray(rot6d, np.float64).copy()
    first = value[..., :3]
    first /= np.maximum(np.linalg.norm(first, axis=-1, keepdims=True), 1.0e-12)
    second = value[..., 3:]
    second -= np.sum(first * second, axis=-1, keepdims=True) * first
    second /= np.maximum(np.linalg.norm(second, axis=-1, keepdims=True), 1.0e-12)
    return np.stack((first, second, np.cross(first, second)), axis=-1)

def _mesh_color(model, geom_id: int):
    rgba = np.asarray(model.geom_rgba[geom_id])
    if abs(float(rgba[0]) - 0.5) < 0.01 and abs(float(rgba[1]) - 0.5) < 0.01:
        return (232, 148, 60)
    if float(np.max(rgba[:3])) < 0.18:
        return (173, 181, 191)
    return tuple(int(255 * np.clip(channel, 0.0, 1.0)) for channel in rgba[:3])

def _rigid_fk_from_global_rotations(
    global_rotation: np.ndarray,
    parents: np.ndarray,
    rest_offsets: np.ndarray,
) -> np.ndarray:
    """Reconstruct connected joints without changing Greenwich rotations."""
    global_rotation = np.asarray(global_rotation, np.float64)
    parents = np.asarray(parents, np.int64)
    rest_offsets = np.asarray(rest_offsets, np.float64)
    positions = np.zeros((*global_rotation.shape[:2], 3), np.float64)
    for joint, parent_joint in enumerate(parents):
        if parent_joint >= 0:
            positions[:, joint] = (
                positions[:, parent_joint]
                + np.einsum(
                    "tij,j->ti",
                    global_rotation[:, parent_joint],
                    rest_offsets[joint],
                )
            )
    return positions

def _sample_greenwich_head(
    trace_path: Path,
    xml: str,
    body: str,
    *,
    stabilize: bool = False,
    rotation_key: str = "raw_rot6d",
    world_position_key: str | None = None,
    floor_y_cm: float | None = None,
):
    """Pose vendor meshes from Greenwich rotations through rigid-bone FK.

    This is the sparse-completion output shown before the optional mechanical
    joint-space projection.  The decoder position head is diagnostic-only: it
    predicts joints independently and therefore cannot be used to place rigid
    robot links.  Every visual geom instead follows the robot's fixed skeleton
    and Greenwich's predicted global rotations; no IK is performed here.
    """
    import mujoco as mj
    from scipy.spatial.transform import Rotation
    from alphamotion.embodiment.registry import load, load_from_xml
    from alphamotion.viz.kinematics import visual_mesh_geom_ids

    result = np.load(trace_path, allow_pickle=True)
    raw_rotation = _rot6d_to_matrix_numpy(result[rotation_key])
    visible_joints = set(np.asarray(
        result["constraint_joint_indices"], np.int64).tolist())
    root_t = np.asarray(result["root_t"], np.float64)
    joint_names = [str(value) for value in result["joint_names"]]
    if stabilize:
        stabilization = stabilize_masked_rotation_islands(
            raw_rotation, visible_joints, joint_names)
        raw_rotation = stabilization.rotation
        stabilization_report = stabilization.report
    else:
        stabilization_report = {
            "algorithm": "none",
            "uses_target_error": False,
            "visible_constraints_immutable": True,
            "modified_joint_frame_values": 0,
            "modified_frames": 0,
            "modified_joints": 0,
            "mean_change_deg": 0.0,
            "max_change_deg": 0.0,
            "repairs": [],
        }

    # Use the exact skeleton on which Greenwich inferred this sample.  G1 is a
    # bundled descriptor; H2 was ingested from its vendor MJCF for this proof.
    try:
        embodiment = load(body)
        if list(embodiment.spec.joint_names) != joint_names:
            embodiment = load_from_xml(xml, body)
    except KeyError:
        embodiment = load_from_xml(xml, body)
    spec = embodiment.spec
    if list(spec.joint_names) != joint_names:
        raise ValueError(
            f"Greenwich/result skeleton mismatch for {body}: "
            f"{len(joint_names)} result joints vs {len(spec.joint_names)} FK joints"
        )

    # Greenwich rotations are global. Reconstruct connected joint positions
    # from immutable rest offsets exactly as alphamotion.engine.spatial.fk_pos.
    # Using raw_position_root_cm here is what previously exploded the meshes.
    parents = np.asarray(spec.parents, np.int64)
    offsets = np.asarray(spec.rest_offsets, np.float64)
    rigid_position = (
        np.asarray(result[world_position_key], np.float64)
        - root_t[:, None]
        if world_position_key is not None
        else _rigid_fk_from_global_rotations(
            raw_rotation, parents, offsets)
    )

    model = mj.MjModel.from_xml_path(str(xml))
    data = mj.MjData(model)
    data.qpos[:] = model.qpos0
    mj.mj_forward(model, data)
    result_index = {name: slot for slot, name in enumerate(joint_names)}
    body_to_result = {}
    for body_id in range(1, model.nbody):
        name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, body_id)
        if name in result_index:
            body_to_result[body_id] = result_index[name]
    geom_ids = visual_mesh_geom_ids(model)

    # Cache Y-up -> vendor MuJoCo Z-up.  This is AX.T from mjcf_build.
    world_joint = (
        np.asarray(result[world_position_key], np.float64)
        if world_position_key is not None
        else rigid_position + root_t[:, None]
    )
    origin = root_t[0]
    world_joint_zup = np.einsum(
        "ij,tkj->tki", YUP_TO_ZUP,
        (world_joint - origin[None, None]) / 100.0)
    world_rotation_zup = _yup_rotation_to_zup(raw_rotation)

    positions = np.zeros((len(rigid_position), len(geom_ids), 3), np.float32)
    matrices = np.zeros((len(rigid_position), len(geom_ids), 3, 3), np.float64)
    meshes = []
    parent = np.asarray(model.body_parentid, np.int64)
    for slot, geom_id in enumerate(geom_ids):
        owner = int(model.geom_bodyid[geom_id])
        while owner not in body_to_result and owner > 0:
            owner = int(parent[owner])
        if owner not in body_to_result:
            raise ValueError(f"visual geom {geom_id} has no descriptor owner")
        result_joint = body_to_result[owner]
        frame_id = owner
        frame_rotation = np.asarray(
            data.xmat[frame_id], np.float64).reshape(3, 3)
        frame_position = np.asarray(data.xpos[frame_id], np.float64)
        geom_rotation = np.asarray(
            data.geom_xmat[geom_id], np.float64).reshape(3, 3)
        geom_position = np.asarray(data.geom_xpos[geom_id], np.float64)
        local_position = frame_rotation.T @ (geom_position - frame_position)
        local_rotation = frame_rotation.T @ geom_rotation
        joint_rotation = world_rotation_zup[:, result_joint]
        positions[:, slot] = (
            world_joint_zup[:, result_joint]
            + np.einsum("tij,j->ti", joint_rotation, local_position)
        ).astype(np.float32)
        matrices[:, slot] = np.einsum(
            "tij,jk->tik", joint_rotation, local_rotation)

        mesh_id = int(model.geom_dataid[geom_id])
        va, vn = int(model.mesh_vertadr[mesh_id]), int(model.mesh_vertnum[mesh_id])
        fa, fn = int(model.mesh_faceadr[mesh_id]), int(model.mesh_facenum[mesh_id])
        meshes.append((
            np.asarray(model.mesh_vert[va:va + vn], np.float32),
            np.asarray(model.mesh_face[fa:fa + fn], np.uint32),
            _mesh_color(model, geom_id),
        ))

    # Realized motor output must use the unmerged vendor tree. In particular,
    # TALOS has co-located wrist joints whose intermediate meshes do NOT share
    # the final merged wrist rotation. Do not use this for raw decoder heads.
    if rotation_key == "rot6d" and "q" in result and not stabilize:
        from .native_visual_fk import descriptor_visual_fk
        root = int(np.flatnonzero(parents < 0)[0])
        positions, matrices = descriptor_visual_fk(
            xml, body, joint_names, result["q"], geom_ids,
            world_joint_zup[:, root], world_rotation_zup[:, root])
        positions = positions.astype(np.float32)
        stabilization_report["visual_pose_source"] = "native_unmerged_motor_fk"

    # A single first-frame lift defines the floor; never re-ground each frame.
    lows = []
    for slot, (vertices, _faces, _color) in enumerate(meshes):
        lows.append(float((vertices @ matrices[0, slot].T)[:, 2].min()
                          + positions[0, slot, 2]))
    ground_z = ((float(origin[1])-float(floor_y_cm))/100.0
                if floor_y_cm is not None else -min(lows) if lows else 0.0)
    positions[..., 2] += ground_z
    wxyz = Rotation.from_matrix(matrices.reshape(-1, 3, 3)).as_quat(
        scalar_first=True).reshape(len(rigid_position), len(geom_ids), 4).astype(np.float32)
    if stabilization_report["modified_joints"]:
        print(
            f"DISPLAY STABILIZATION {body}: "
            f"{stabilization_report['modified_joints']} masked joints / "
            f"{stabilization_report['modified_frames']} frames",
            flush=True,
        )
    return (positions, wxyz, meshes, float(ground_z), world_joint,
            world_rotation_zup, stabilization_report)
