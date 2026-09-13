# Source adapter contract

MotionClip stores hierarchy, bind offsets, global SO(3) rotations, world joint
positions, timestamps and provenance. Canonical frame: right-handed Y-up,
centimetres, floor Y=0. No resizing for a different robot.

```python
from greenwich_motion_sdk import load_motion
clip = load_motion("motion.bvh", format="soma", bind_path="soma_bind.bvh")
clip.save("canonical.npz")
```

`register_adapter(name, loader)` adds a format returning a validated MotionClip.
SOMA encodes all 77 anatomical joints, removing only the identity scene Root.
Generic BVH requires explicit bind, CoordinateFrame and position-channel semantics.
Native SMPL requires axis-angle poses/trans and caller-supplied shaped skeleton
and pelvis offset J0; inspect `adapters.smpl` for exact keyword arguments.

Validation status: SOMA real data + robot pipeline; generic BVH checked against
SOMA BVHs; native SMPL loader unit tests and a SOMA-derived SMPL-format GPU smoke;
canonical NPZ round trip and manifest tests. No independent native SMPL corpus
accuracy claim. tests/smoke_full_body.py reproduces the manual integration check.
`smpl22-derived` is an ablation derived from SOMA, not native SMPL corpus support.
EGO/UMI sparse SE(3) pipelines are separate and excluded from this bundle.

## Explicit source manifest

```json
{
  "schema": "alphamotion.sources.v1",
  "sources": [{
    "id": "clip_001", "format": "smpl", "path": "clip.npz",
    "options_file": "smpl-profile.json",
    "roles": ["pelvis", "head", "left_wrist", "right_wrist", "left_ankle", "right_ankle"],
    "foot_pairs": [["left_ankle", "left_toe"], ["right_ankle", "right_toe"]],
    "foot_landmark_height_cm": [0, 0]
  }]
}

```

Roles are exact names in the supplied hierarchy; do not assume dataset naming.
SMPL profile contains names, parents, shaped rest_offsets, shaped
pelvis_offset_source_units and frame {basis_to_yup, cm_per_unit, ground_y_cm}.
Offsets are local parent-frame vectors in source units; root trans + J0 defines
world pelvis. Never substitute nominal robot height for a measured source floor.
BVH options contain bind_path, frame and position_channels; canonical needs no
options. SOMA requires bind_path and has default SOMA77 roles. File paths are
relative to the manifest; profile bind_path is relative to the profile.
foot_landmark_height_cm is the declared neutral ankle/toe-to-floor offset per
side, used for airborne clearance, not trajectory translation. Default zero is
an uncalibrated geometric proxy. Foot support phase is independently inferred
from source ankle/toe height/speed and is not measured force/contact truth.
Sources must have uniform timing and an integer FPS stride. No interpolation,
body-size scaling or hidden horizontal recentering is performed by Convert.
Noncompact output includes sampled source.npz plus source role/timeline fields;
Evaluate supports these outputs. Compact keeps reconstruction fields but requires
the separate corpus manifest/source workflow for evaluation.
