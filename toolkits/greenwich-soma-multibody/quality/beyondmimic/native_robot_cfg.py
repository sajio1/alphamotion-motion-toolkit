"""Native BeyondMimic quality-evaluation adapters for A3/G1/Booster motions.

This module adapts robot assets and controller interfaces only.  Reward weights,
motion data, and task success thresholds are shared across robots and motions.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path

import mujoco

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.observation_manager import ObservationGroupCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg


ROOT = Path(__file__).resolve().parent
ASSET_ROOT = Path(os.environ.get("GREENWICH_BM_ASSET_ROOT", ROOT / "assets")).resolve()


@dataclass(frozen=True)
class RobotSpec:
    key: str
    xml: Path
    root_body: str
    anchor_body: str
    body_names: tuple[str, ...]
    ee_bodies: tuple[str, ...]
    feet: tuple[str, str]
    foot_geom_expr: str
    root_height: float
    linear_velocity_sensor: str
    angular_velocity_sensor: str


ROBOTS = {
    "a3": RobotSpec(
        key="a3",
        xml=ASSET_ROOT / "a3" / "a3_t3d0_native.xml",
        root_body="pelvis_link",
        anchor_body="torso_Link",
        body_names=(
            "pelvis_link", "left_hip_roll_Link", "left_knee_Link",
            "left_ankle_roll_Link", "right_hip_roll_Link", "right_knee_Link",
            "right_ankle_roll_Link", "torso_Link", "left_shoulder_roll_Link",
            "left_elbow_Link", "left_wrist_yaw_Link", "right_shoulder_roll_Link",
            "right_elbow_Link", "right_wrist_yaw_Link",
        ),
        ee_bodies=(
            "left_ankle_roll_Link", "right_ankle_roll_Link",
            "left_wrist_yaw_Link", "right_wrist_yaw_Link",
        ),
        feet=("left_ankle_roll_Link", "right_ankle_roll_Link"),
        foot_geom_expr=r"^(left|right)_foot([1-9]|1[0-3])_collision$",
        root_height=1.3,
        linear_velocity_sensor="robot/pelvis-linear-vel",
        angular_velocity_sensor="robot/pelvis-angular-velocity",
    ),
    "booster_t1_29": RobotSpec(
        key="booster_t1_29",
        xml=ASSET_ROOT / "booster" / "booster_t1_27dof_native.xml",
        root_body="Trunk",
        anchor_body="Trunk",
        body_names=(
            "Trunk", "Hip_Roll_Left", "Shank_Left", "left_foot_link",
            "Hip_Roll_Right", "Shank_Right", "right_foot_link", "AL2", "AL4",
            "left_hand_link", "AR2", "AR4", "right_hand_link",
        ),
        ee_bodies=(
            "left_foot_link", "right_foot_link", "left_hand_link", "right_hand_link",
        ),
        feet=("left_foot_link", "right_foot_link"),
        foot_geom_expr=r"^(left|right)_foot_collision$",
        root_height=0.7,
        linear_velocity_sensor="robot/linear-velocity",
        angular_velocity_sensor="robot/angular-velocity",
    ),
}


def _joint_efforts(xml: Path) -> dict[str, float]:
    model = mujoco.MjModel.from_xml_path(str(xml))
    efforts: dict[str, float] = {}
    for actuator_id in range(model.nu):
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        if not name:
            continue
        actuator_range = model.actuator_ctrlrange[actuator_id]
        joint_range = model.jnt_actfrcrange[joint_id]
        value = max(abs(float(actuator_range[0])), abs(float(actuator_range[1])))
        if value <= 1.0 and int(model.jnt_actfrclimited[joint_id]):
            value = max(abs(float(joint_range[0])), abs(float(joint_range[1])))
        efforts[name] = max(value, 1.0)
    return efforts


def _inertia_proxy(name: str) -> float:
    """One role-based rule shared by all robots; never tuned per motion."""
    if any(part in name for part in ("hip", "knee", "ankle")):
        return 0.12
    if "waist" in name:
        return 0.08
    if any(part in name for part in ("shoulder", "elbow", "wrist")):
        return 0.03
    return 0.01


def _robot_cfg(spec: RobotSpec) -> EntityCfg:
    efforts = _joint_efforts(spec.xml)

    def get_spec() -> mujoco.MjSpec:
        model_spec = mujoco.MjSpec.from_file(str(spec.xml))
        for actuator in list(model_spec.actuators):
            model_spec.delete(actuator)
        return model_spec

    actuators = []
    for joint_name, effort in efforts.items():
        # Use the same effort-normalized, critically damped interface rule for both
        # custom robots. Limits come from their vendor MJCFs.
        stiffness = max(8.0, effort / 0.4)
        inertia = _inertia_proxy(joint_name)
        # Match the native G1 actuator construction: include reflected motor
        # inertia and use a safely over-damped (zeta=2) position interface.
        damping = 2.0 * 2.0 * math.sqrt(stiffness * inertia)
        actuators.append(
            BuiltinPositionActuatorCfg(
                target_names_expr=(joint_name,),
                stiffness=stiffness,
                damping=damping,
                effort_limit=effort,
                armature=inertia,
            )
        )

    return EntityCfg(
        init_state=EntityCfg.InitialStateCfg(
            pos=(0.0, 0.0, spec.root_height),
            joint_pos={".*": 0.0},
            joint_vel={".*": 0.0},
        ),
        spec_fn=get_spec,
        articulation=EntityArticulationInfoCfg(
            actuators=tuple(actuators), soft_joint_pos_limit_factor=0.9
        ),
    )


def native_tracking_env_cfg(
    robot: str,
    *,
    has_state_estimation: bool = True,
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    if robot == "g1":
        from mjlab.tasks.tracking.config.g1.env_cfgs import (
            unitree_g1_flat_tracking_env_cfg,
        )

        return unitree_g1_flat_tracking_env_cfg(
            has_state_estimation=has_state_estimation, play=play
        )

    spec = ROBOTS[robot]
    cfg = make_tracking_env_cfg()
    cfg.scene.entities = {"robot": _robot_cfg(spec)}
    cfg.scene.sensors = (
        ContactSensorCfg(
            name="self_collision",
            primary=ContactMatch(
                mode="subtree", pattern=spec.root_body, entity="robot"
            ),
            secondary=ContactMatch(
                mode="subtree", pattern=spec.root_body, entity="robot"
            ),
            fields=("found", "force"),
            reduce="none",
            num_slots=1,
            history_length=4,
        ),
    )

    action = cfg.actions["joint_pos"]
    assert isinstance(action, JointPositionActionCfg)
    # Match BeyondMimic's native G1 interface normalization: a unit policy
    # output requests 25% of the actuator's effort/stiffness displacement.
    # This is derived from each hardware model rather than tuned by robot or
    # motion, and avoids injecting an arbitrary 0.5-rad step into every joint.
    efforts = _joint_efforts(spec.xml)
    action.scale = {
        name: 0.25 * effort / max(8.0, effort / 0.4)
        for name, effort in efforts.items()
    }

    command = cfg.commands["motion"]
    assert isinstance(command, MotionCommandCfg)
    command.anchor_body_name = spec.anchor_body
    command.body_names = spec.body_names

    cfg.events["foot_friction"].params["asset_cfg"].geom_names = spec.foot_geom_expr
    cfg.events["base_com"].params["asset_cfg"].body_names = (spec.anchor_body,)
    cfg.terminations["ee_body_pos"].params["body_names"] = spec.ee_bodies
    cfg.viewer.body_name = spec.anchor_body
    # The vendor assets expose more collision pairs than G1.  Raise fixed solver
    # capacities for both custom robots; this does not alter the motion or reward.
    cfg.sim.nconmax = 128
    cfg.sim.njmax = 512
    for group in ("actor", "critic"):
        cfg.observations[group].terms["base_lin_vel"].params["sensor_name"] = (
            spec.linear_velocity_sensor
        )
        cfg.observations[group].terms["base_ang_vel"].params["sensor_name"] = (
            spec.angular_velocity_sensor
        )

    if not has_state_estimation:
        cfg.observations["actor"] = ObservationGroupCfg(
            terms={
                key: value
                for key, value in cfg.observations["actor"].terms.items()
                if key not in ("motion_anchor_pos_b", "base_lin_vel")
            },
            concatenate_terms=True,
            enable_corruption=True,
        )
    if play:
        cfg.episode_length_s = int(1e9)
        cfg.observations["actor"].enable_corruption = False
        cfg.events.pop("push_robot", None)
        command.pose_range = {}
        command.velocity_range = {}
        command.sampling_mode = "start"
    return cfg


def robot_eval_names(robot: str) -> dict[str, tuple[str, ...]]:
    if robot == "g1":
        return {
            "wrists": ("left_wrist_yaw_link", "right_wrist_yaw_link"),
            "feet": ("left_ankle_roll_link", "right_ankle_roll_link"),
        }
    spec = ROBOTS[robot]
    return {"wrists": spec.ee_bodies[2:], "feet": spec.feet}
