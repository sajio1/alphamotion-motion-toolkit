from __future__ import annotations
from pathlib import Path
import subprocess
import numpy as np
WIDTH, HEIGHT = 1920, 1080
SHOWCASE_FOV_DEGREES = 42.0
SHOWCASE_CAMERA_POSITION_M = np.asarray([2.05,1.55,1.62])
SHOWCASE_CAMERA_TARGET_M = np.asarray([0.,0.,.82])


def _camera_matrix(position: np.ndarray, target: np.ndarray) -> np.ndarray:
    forward = np.asarray(target, np.float64) - np.asarray(position, np.float64)
    forward /= max(float(np.linalg.norm(forward)), 1.0e-9)
    up_hint = np.asarray([0.0, 0.0, 1.0], np.float64)
    right = np.cross(forward, up_hint)
    right /= max(float(np.linalg.norm(right)), 1.0e-9)
    up = np.cross(right, forward)
    up /= max(float(np.linalg.norm(up)), 1.0e-9)
    # MuJoCo fixed cameras look along local -Z and use local +Y as image up.
    return np.column_stack((right, up, -forward))

def _configure_model(model):
    model.vis.global_.offwidth = WIDTH
    model.vis.global_.offheight = HEIGHT
    model.vis.quality.shadowsize = 8192
    model.vis.quality.offsamples = 8
    model.vis.headlight.ambient[:] = [0.48, 0.48, 0.48]
    model.vis.headlight.diffuse[:] = [0.66, 0.66, 0.66]
    model.vis.headlight.specular[:] = [0.12, 0.12, 0.12]

def _set_fixed_camera(model, data, mujoco, name, position, target):
    camera_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_CAMERA, name)
    _update_fixed_camera_pose(data, camera_id, position, target)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
    camera.fixedcamid = camera_id
    return camera

def _update_fixed_camera_pose(data, camera_id, position, target):
    data.cam_xpos[camera_id] = position
    data.cam_xmat[camera_id] = _camera_matrix(position, target).reshape(-1)

def _writer(ffmpeg: Path, output: Path, fps: float):
    output.parent.mkdir(parents=True, exist_ok=True)
    return subprocess.Popen([
        str(ffmpeg), "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{WIDTH}x{HEIGHT}", "-r", f"{fps:.8f}", "-i", "-",
        "-an", "-c:v", "libx264", "-preset", "slow", "-crf", "14",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        "-y", str(output),
    ], stdin=subprocess.PIPE)

def _close_writer(writer):
    if writer.stdin is not None:
        writer.stdin.close()
    code = writer.wait()
    if code:
        raise subprocess.CalledProcessError(code, writer.args)
