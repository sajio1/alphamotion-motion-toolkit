# AlphaMotion Motion Toolkit

Data adapters -> canonical MotionClip -> AlphaMotion model -> native joint
projection -> shared contact refinement -> saved NPZ -> evaluation and review.

This is a data-free source snapshot of the current pipeline, not model weights.
Components are configurable independently; evaluation reads saved output and
does not rerun inference. Existing model APIs retain their `Greenwich` class name.

## Install

```bash
python -m pip install -e ".[runtime,test]"
```

Install your existing AlphaMotion repository/checkpoint separately and keep its
environment lock/version. FFmpeg and PowerShell 7 are external executables.
The v2 checkpoints bumblebee, megatron and prime are selectable via `-Model`
(default prime); `-RunDir` supplies an external checkpoint override. The Python
convert API exposes `model` and `run_dir`. Model source and weights are external.
CUDA is required by the current generation backend. Copy
`toolkits/greenwich-soma-multibody/config/local-paths.example.json` to
`local-paths.json`, then supply absolute local paths. The `pipeline` location is
this package's `runtime` directory; `sole_backend` is this package's `src`.
No network download is triggered by the generation entrypoint.

## Convert SOMA

Supply a robot roster JSON using `config/robots.example.json` as the template.
Prepare a workspace containing `outputs/seed-100-contact/selection.json` and
`outputs/seed-100-contact/source/{motion.bvh,soma_base_skel_minimal.bvh}`.
`prepare_soma_subset.py --help` documents streaming extraction from an existing
local archive and metadata. Dataset files are never checked into this repo.

```powershell
& toolkits/greenwich-soma-multibody/scripts/invoke.ps1 `
  -Stage Generate -Indices '1,2,3' -Robots /path/to/robots.json `
  -DatasetWorkspace /path/to/workspace -Output /path/to/results `
  -ContactIterations 100 -SkipPreview -Compact
```

Noncompact runs save each robot's `motion.npz`; compact runs save one NPZ per
source with named robot prefixes. Full-SOMA conversion has been exercised across
multiple robots. Native SMPL, generic BVH and canonical files use the same backend:

```powershell
& toolkits/greenwich-soma-multibody/scripts/invoke.ps1 `
  -Stage Convert -SourceManifest /path/to/sources.json `
  -Robots /path/to/robots.json -Output /path/to/results `
  -ContactIterations 100 -SkipPreview
```

See [formats](docs/formats.md) for the explicit manifest and validation scope.

## Evaluate saved outputs

For noncompact Convert output, run:

```powershell
& toolkits/greenwich-soma-multibody/scripts/invoke.ps1 `
  -Stage Evaluate -AuditReport /path/to/results -SampleRobot my_robot `
  -Robots /path/to/robots.json -ViewerDescriptor /path/to/robot.json.gz `
  -Output /path/to/evaluation -SlipCmS 15 -PenetrationCm 0.5 -BadDurationSeconds 0.1
```

Changing thresholds reads saved outputs; it does not rerun the model or refiner.
Use `python -m greenwich_motion_sdk.locomotion_audit --help` for compact traces
with an exact source manifest and native viewer descriptor. It measures slip,
penetration and disclosed source/robot pose differences. Corpus entrypoints
`ScreenSample`, `ScreenStatistical`, `ScreenAll`, `SurfaceAudit` and
`RescreenSurfaces` reuse saved traces and caches. See [evaluation](docs/evaluation.md).

## Review and extend

See [robot integration](docs/robots.md), [Python API](docs/api.md),
[architecture](docs/architecture.md) and [review](docs/review.md).
The package does not include data, pictures, videos, weights or robot meshes.
SOURCE_MANIFEST.json records the reviewed source snapshot and file hashes.
