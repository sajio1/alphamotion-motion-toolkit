from pathlib import Path
import numpy as np


def test_compact_smoke_bundle_is_reconstruction_complete():
    path = Path("outputs/soma-compact-smoke/motions/crawl_ff_start_180_R_003__A232.npz")
    if not path.exists(): return
    with np.load(path, allow_pickle=False) as data:
        assert str(data["schema"]) == "greenwich.soma18.compact.v1"
        for robot in ("a3", "gr3"):
            assert data[robot + "__q"].dtype == np.float32
            assert data[robot + "__root_rot6d"].shape[-1] == 6
            assert data[robot + "__root_t_cm"].shape[-1] == 3
