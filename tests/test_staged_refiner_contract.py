"""The public full-body SDK exposes only the fixed 100+100 recipe."""

import subprocess
import sys
from pathlib import Path

import pytest

from greenwich_motion_sdk.pipeline import Pipeline, RunRequest


ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "toolkits/greenwich-soma-multibody/scripts/run_soma_locomotion.py"


def test_public_sdk_has_no_iteration_override(tmp_path):
    request = RunRequest((1,), tmp_path / "out", tmp_path / "robots.json")
    command = Pipeline(ROOT / "toolkits/greenwich-soma-multibody").command(request)
    assert "-ContactIterations" not in command
    with pytest.raises(TypeError):
        RunRequest((1,), tmp_path / "out", tmp_path / "robots.json", contact_iterations=120)
    with pytest.raises(TypeError):
        Pipeline(ROOT / "toolkits/greenwich-soma-multibody").convert(
            tmp_path / "sources.json", tmp_path / "out", tmp_path / "robots.json",
            contact_iterations=120)


def test_backend_cli_has_no_contact_only_mode():
    result = subprocess.run([sys.executable, str(BACKEND), "--help"],
                            check=True, capture_output=True, text=True)
    assert "--contact-iters" not in result.stdout
    assert "--coordination-config" not in result.stdout
    source = BACKEND.read_text(encoding="utf-8")
    assert "a.contact_iters=100" in source
    assert "iterations=100,palm_frames=" in source
    assert "optimize_trunk=False" in source


def test_body_coordination_is_packaged():
    from greenwich_motion_sdk.body_coordination import upper_subtree

    assert callable(upper_subtree)
