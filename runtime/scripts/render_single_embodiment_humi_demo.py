from __future__ import annotations
from pathlib import Path
import subprocess
import numpy as np
WIDTH, HEIGHT = 1920, 1080
SHOWCASE_FOV_DEGREES = 42.0
SHOWCASE_CAMERA_POSITION_M = np.asarray([2.05,1.55,1.62])
SHOWCASE_CAMERA_TARGET_M = np.asarray([0.,0.,.82])


def _fit_action_camera(positions, matrices, meshes, direction=None):
    """Fit actual mesh extents over the full clip, not just link centers."""
    clouds = []
    for slot, (vertices, _faces, _color) in enumerate(meshes):
        low, high = np.min(vertices, axis=0), np.max(vertices, axis=0)
        corners = np.asarray([
            [x, y, z] for x in (low[0], high[0])
            for y in (low[1], high[1]) for z in (low[2], high[2])])
        clouds.append((np.einsum(
            "tij,kj->tki", matrices[:, slot], corners)
            + positions[:, slot, None]).reshape(-1, 3))
    points = np.concatenate(clouds, axis=0)
    target = (points.min(axis=0) + points.max(axis=0)) * 0.5
    if direction is None:
        direction = SHOWCASE_CAMERA_POSITION_M - SHOWCASE_CAMERA_TARGET_M
    direction = direction / np.linalg.norm(direction)
    right = np.cross(-direction, np.asarray([0.0, 0.0, 1.0]))
    right /= np.linalg.norm(right)
    up = np.cross(right, -direction)
    relative = points - target
    tan_y = np.tan(np.deg2rad(SHOWCASE_FOV_DEGREES * 0.5))
    tan_x = tan_y * WIDTH / HEIGHT
    required = relative @ direction + np.maximum(
        np.abs(relative @ right) / tan_x,
        np.abs(relative @ up) / tan_y)
    distance = max(float(required.max()) * 1.16, 1.0)
    return target + direction * distance, target, distance
