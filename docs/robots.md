# Add an embodiment

Provide your robot's vendor model and meshes externally. The current native
builder consumes MJCF. Import/normalize raw URDF with your model tooling first;
raw URDF alone is not a promise of automatic closed-chain or semantic mapping.

1. Preserve actual hinge axes, joint limits, rest transforms and native link tree.
2. Normalize model world coordinates into canonical Y-up centimetres, floor Y=0.
3. Supply a semantic descriptor name and verify wrists, feet and left/right roles.
4. Create a roster entry `{name, label, body, xml}` in a local JSON file.
5. Pass that file through `-Robots`, without changing data loaders or shared costs.
6. Export a native visual descriptor with `export_viewer_robot.py --help` and
   compare saved native FK with vendor FK before claiming a new robot works.

Meshes remain needed locally for exact sole/surface geometry. Do not insert
per-task trajectory shifts, scales, pose seeds or robot-specific refiner tuning.
Interface normalization is distinct from zero-shot task adaptation.
Supported native joints and closed-chain modeling must be checked per descriptor;
arbitrary URDF compatibility is not guaranteed merely by adding a roster row.
