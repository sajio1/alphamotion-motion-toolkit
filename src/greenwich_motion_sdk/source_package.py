"""Build an allowlisted, data-free source distribution for review before publication.

No GitHub writes, data downloads, model inference or runtime execution occurs.
The bundle records every source hash and explicitly externalizes model/assets.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import zipfile

CORE = ('__init__', 'motion', 'adapters', '_bvh', '_smpl_bind', 'pipeline', 'source_batch', 'evaluation',
        'reports', 'compact', 'contact_labels', 'surface_contacts', 'physics_audit',
        'fixed_support', 'locomotion_audit', 'kimodo_foot_skate',
        'support_geometry', 'native_projection', 'corpus_screen',
        'corpus_support', 'support_sweep', 'sampling_statistics', 'surface_audit',
        'render_review', 'card_gallery', 'review_sampling', 'review_sources',
        'preview_candidates', 'source_package', 'body_coordination')
SCRIPTS = ('run_soma_locomotion.py', 'soma_contact_refine.py',
           'audit_transfer_accuracy.py', 'audit_soma_contact.py',
           'audit_soma_foot_phases.py', 'audit_ground_error.py',
           'summarize_soma_locomotion.py', 'render_soma_robot_comparison.py',
           'render_soma_robot_grid.py', 'export_viewer_robot.py',
           'prepare_soma_subset.py', 'bulk_convert_soma.py',
           'compact_robot_motion_delivery.py', 'package_compact_shards.py',
           'stream_compact50_delivery.py', 'finalize_h2_compact_delivery.py')
QC_FILES = ('README.md', 'collect_delivery_videos.py', 'compose_variant_videos.py',
            'dex_contact_terms.py', 'g1_csv_to_mjlab.py',
            'greenwich_to_mjlab_native.py', 'native_robot_cfg.py',
            'package_sim_delivery.py', 'run_queue_after.py', 'sim9_evaluate.py',
            'sim9_queue.py', 'sim9_worker.py', 'smoke_native.py',
            'summarize_sim_runs.py', 'validate_benchmark_bundle.py',
            'video_contact_sheet.py')
CONFIGS = ('contact.default.json', 'contact.fast-projection.json',
           'physics-screen.json', 'support-screen.conservative.json',
           'support-screen.references.json')
KIT = 'toolkits/greenwich-soma-multibody'


def function_subset(path, names, header):
    """Retain exact shared function bodies, excluding unrelated demo/UI code."""
    text=Path(path).read_text(encoding='utf-8'); tree=ast.parse(text)
    functions={n.name:n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
    selected=set(names); pending=list(names)
    while pending:
        node=functions[pending.pop()]
        for ref in ast.walk(node):
            if isinstance(ref,ast.Name) and isinstance(ref.ctx,ast.Load) and ref.id in functions and ref.id not in selected:
                selected.add(ref.id);pending.append(ref.id)
    lines=text.splitlines(keepends=True)
    return header+'\n\n'+'\n\n'.join(''.join(lines[n.lineno-1:n.end_lineno]).rstrip()
                                    for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in selected)+'\n'


def build(sdk_root, runtime, output):
    sdk_root,runtime,output=map(lambda p:Path(p).resolve(),(sdk_root,runtime,output))
    if output.exists() and any(output.iterdir()):raise ValueError('Use a new, empty staging directory; existing files are never deleted')
    output.mkdir(parents=True,exist_ok=True); root=output/'alphamotion-locomotion-toolkit'; root.mkdir()
    files={}; origins={}
    def add(name,payload,source=None):
        if name in files:raise ValueError('Duplicate package file: '+name)
        files[name]=payload.encode('utf-8') if isinstance(payload,str) else payload
        if source is not None:origins[name]=hashlib.sha256(Path(source).read_bytes()).hexdigest()
    def copy(source,name):add(name,Path(source).read_bytes(),source)

    # Include transitive SDK imports, not the entire workspace or historical runs.
    modules=set(CORE);pending=list(CORE)
    while pending:
        path=sdk_root/'src/greenwich_motion_sdk'/(pending.pop()+'.py')
        for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
            if isinstance(node,ast.ImportFrom) and node.level==1 and node.module:
                module=node.module.split('.')[0]
                if module not in modules:modules.add(module);pending.append(module)
    for module in sorted(modules):copy(sdk_root/'src/greenwich_motion_sdk'/(module+'.py'),'src/greenwich_motion_sdk/'+module+'.py')
    toolkit=sdk_root/KIT
    for name in SCRIPTS:copy(toolkit/'scripts'/name,KIT+'/scripts/'+name)
    for name in QC_FILES:
        copy(toolkit/'quality/beyondmimic'/name,
             KIT+'/quality/beyondmimic/'+name)
    for name in CONFIGS:copy(toolkit/'config'/name,KIT+'/config/'+name)
    for name in ('test_sdk.py','test_source_batch.py','test_physics_audit.py','test_contact_labels.py','test_surface_contacts.py','test_locomotion_source_phase.py','test_compact_soma_bundle.py'):
        copy(sdk_root/'tests'/name,'tests/'+name)
    copy(sdk_root/'tests/smoke_full_body.py','tests/smoke_full_body.py')
    copy(sdk_root/'tests/test_staged_refiner_contract.py','tests/test_staged_refiner_contract.py')

    # Existing generation uses just these native geometry functions. Do not
    # distribute the unrelated sparse-manipulation server/demo package.
    native=runtime/'src/greenwich_umi_proof'
    if not native.is_dir():native=sdk_root/'src/greenwich_umi_proof'
    add('src/greenwich_umi_proof/__init__.py','"""Compatibility namespace for shared native geometry only."""\n')
    for name in ('sole_geometry.py','native_visual_fk.py'):copy(native/name,'src/greenwich_umi_proof/'+name)
    viewer_header='''from __future__ import annotations
from pathlib import Path
import numpy as np
YUP_TO_ZUP = np.asarray([[0.,0.,1.],[1.,0.,0.],[0.,1.,0.]])
def stabilize_masked_rotation_islands(*args, **kwargs):
    raise ValueError("Display stabilization is excluded from this numerical conversion toolkit")
'''
    add('src/greenwich_umi_proof/interactive_viewer.py',function_subset(native/'interactive_viewer.py',('_sample_greenwich_head',),viewer_header),native/'interactive_viewer.py')
    render_header='''from __future__ import annotations
from pathlib import Path
import subprocess
import numpy as np
WIDTH, HEIGHT = 1920, 1080
SHOWCASE_FOV_DEGREES = 42.0
SHOWCASE_CAMERA_POSITION_M = np.asarray([2.05,1.55,1.62])
SHOWCASE_CAMERA_TARGET_M = np.asarray([0.,0.,.82])
'''
    helper=runtime/'scripts/render_hiw500_transfer_showcase.py'
    add('runtime/scripts/render_hiw500_transfer_showcase.py',function_subset(helper,('_configure_model','_set_fixed_camera','_writer','_close_writer'),render_header),helper)
    helper=runtime/'scripts/render_single_embodiment_humi_demo.py'
    add('runtime/scripts/render_single_embodiment_humi_demo.py',function_subset(helper,('_fit_action_camera',),render_header),helper)
    add('runtime/README.md','Shared rendering functions only. Configure local-paths.pipeline to this runtime directory. Native geometry is installed from src/. No model or robot assets are included.\n')

    invoke=(toolkit/'scripts/invoke.ps1').read_text(encoding='utf-8')
    for stage in ('RefreshGallery','PublishReview'):
        invoke=re.sub(r"^if \(\$Stage -eq '"+stage+r"'\) \{.*?^\}\n",'',invoke,flags=re.M|re.S)
    invoke=re.sub(r"^if \(\$Stage -in @\('PublishCard','PublishBundle'\)\) \{.*?^\}\n",'',invoke,flags=re.M|re.S)
    for stage in ('RefreshGallery','PublishReview','PublishCard','PublishBundle'):
        invoke=invoke.replace(", '"+stage+"'",'').replace(",'"+stage+"'",'')
    add(KIT+'/scripts/invoke.ps1',invoke,toolkit/'scripts/invoke.ps1')
    physics=(toolkit/'scripts/invoke_physics_audit.ps1').read_text(encoding='utf-8')
    physics=physics.replace("[string]$Robot = 'h2',","[string]$Robot = 'h2',\n    [string]$Robots,")
    physics=physics.replace("$argsList+=@('--source'","if (-not $Robots) { $Robots=Join-Path $kit 'config/robots.example.json' }\n    $argsList+=@('--source'")
    physics=physics.replace("(Join-Path $kit 'config/robots.a3-h2-g1.json')",'$Robots')
    add(KIT+'/scripts/invoke_physics_audit.ps1',physics,toolkit/'scripts/invoke_physics_audit.ps1')
    paths={'repo':'/absolute/path/to/alphamotion-model-repository','pipeline':'/absolute/path/to/this-package/runtime',
           'dataset_workspace':'/absolute/path/to/prepared-soma-workspace','python':'/absolute/path/to/python',
           'ffmpeg':'/absolute/path/to/ffmpeg','sole_backend':'/absolute/path/to/this-package/src','optional_pythonlibs':''}
    add(KIT+'/config/local-paths.example.json',json.dumps(paths,indent=2)+'\n')
    robot=[{'name':'my_robot','label':'My robot','body':'my_robot_descriptor_name','xml':'/absolute/path/to/robot/model.xml'}]
    add(KIT+'/config/robots.example.json',json.dumps(robot,indent=2)+'\n')
    add('.gitignore','''__pycache__/
*.py[cod]
.venv/
.pytest_cache/
*.egg-info/
.env*
**/local-paths.json
data/
outputs/
cache/
weights/
assets/
*.npz
*.npy
*.bvh
*.parquet
*.h5
*.pt
*.pth
*.safetensors
*.png
*.jpg
*.jpeg
*.gif
*.mp4
*.zip
*.log
MUJOCO_LOG.TXT
''')
    add('.gitattributes','* -text\n')  # Keep source-manifest byte hashes stable across platforms.
    add('pyproject.toml','''[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"
[project]
name = "alphamotion-locomotion-toolkit"
version = "0.2.0"
description = "Modular motion adapters, native robot retargeting, contact refinement and offline evaluation"
requires-python = ">=3.10"
dependencies = ["numpy>=1.24", "scipy>=1.10"]
[project.optional-dependencies]
runtime = ["torch", "mujoco>=3.2", "opencv-python", "imageio-ffmpeg", "pandas", "pyarrow", "matplotlib", "Pillow"]
test = ["pytest"]
[tool.setuptools.packages.find]
where = ["src"]
''')
    for name,text in DOCUMENTS.items():add(name,text)
    add('LICENSE','''MIT License

Copyright (c) 2026 AlphaMotion motion toolkit contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

This license covers the toolkit software only; external model, robot and source
dataset licenses are independent. See NOTICE.md.
''')
    add('NOTICE.md','''# External components

AlphaMotion implementation/checkpoints, robot URDF/MJCF/meshes and source datasets
are external dependencies, not redistributed here. Their licenses continue to
apply independently. Native geometry and rendering compatibility functions are
extracted from the local pipeline; exact source hashes are in SOURCE_MANIFEST.json.
No source-data license is replaced by this software bundle.
''')
    for name,payload in files.items():
        if name.endswith('.py'):compile(payload,name,'exec')
        if name.endswith('.json'):json.loads(payload)
        if re.search(rb'[A-Za-z]:[/\\](?:Users|GreenwichData)|hf_[A-Za-z0-9]{20,}|gh[pousr]_[A-Za-z0-9]{20,}',payload):
            raise ValueError('Machine-specific path or credential-like string in '+name)
        target=root/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(payload)
    manifest={'schema':'alphamotion.source-package.v1','file_count':len(files),
              'source_modules':sorted(modules),'files':[{'path':n,'bytes':len(p),'sha256':hashlib.sha256(p).hexdigest(),
              **({'source_sha256':origins[n]} if n in origins else {})} for n,p in sorted(files.items())],
              'excluded':['datasets','weights','robot meshes','photos','videos','gifs','logs','credentials','machine paths','historical Git metadata','cloud session/account scripts','HF publication scripts'],
              'publication_performed_by_builder':False}
    (root/'SOURCE_MANIFEST.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    tree='\n'.join(sorted([*files,'SOURCE_MANIFEST.json']))+'\n'
    (output/'UPLOAD_PREVIEW.txt').write_text(tree,encoding='utf-8')
    archive=output/'alphamotion-locomotion-toolkit-source.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for path in sorted(root.rglob('*')):
            if path.is_file():z.write(path,path.relative_to(output).as_posix())
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:raise ValueError('Source ZIP failed CRC verification')
    result={'directory':str(root),'zip':str(archive),'files':len(files)+1,'source_bytes':sum(map(len,files.values())),
            'zip_bytes':archive.stat().st_size,'zip_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'published':False}
    (output/'PACKAGE_REPORT.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result),flush=True);return result


DOCUMENTS = {
 'README.md': '''# AlphaMotion Locomotion Toolkit

Data adapters -> canonical MotionClip -> AlphaMotion model -> native joint
projection -> 100 lower/contact/root iterations, then 100 body-relative upper-body
iterations -> saved NPZ -> evaluation and review. This is the only supported
conversion recipe; there is no contact-only or 120-iteration mode.
The upper-body stage requires a valid anatomical arm mapping in the source and
target robot; unsupported mappings fail explicitly rather than falling back to
a lower-body-only output.

Root height follows the active load-bearing endpoint family. Ordinary support
uses feet; inverted support uses semantic hand endpoints. Fixed wrist/TCP
origins are used when a robot has no hand mesh, without per-robot offsets.

This conversion path is the primary product. Optional controller-training adapters are downstream
quality checks and do not replace the saved-motion audits or alter generated trajectories.

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
  -SkipPreview -Compact
```

Noncompact runs save each robot's `motion.npz`; compact runs save one NPZ per
source with named robot prefixes. Full-SOMA conversion has been exercised across
multiple robots. Native SMPL, generic BVH and canonical files use the same backend:

```powershell
& toolkits/greenwich-soma-multibody/scripts/invoke.ps1 `
  -Stage Convert -SourceManifest /path/to/sources.json `
  -Robots /path/to/robots.json -Output /path/to/results `
  -SkipPreview
```

See [formats](docs/formats.md) for the explicit manifest and validation scope.

## Package compact robot deliveries

Reusable delivery tools live beside the SOMA scripts. They produce one compact robot-motion schema,
deterministic `tar.zst` shards, resumable transfer state, checksums, and an explicit failure ledger.
Source SOMA arrays and model intermediates are not duplicated into the robot-motion delivery.

```powershell
python toolkits/greenwich-soma-multibody/scripts/compact_robot_motion_delivery.py `
  --source /path/to/generated --output /path/to/compact --robot h2 --workers 4
python toolkits/greenwich-soma-multibody/scripts/package_compact_shards.py `
  --source /path/to/compact --output /path/to/shards --prefix h2 --name h2-compact50
```

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

## Optional BeyondMimic QC

`toolkits/greenwich-soma-multibody/quality/beyondmimic` contains the parameterized adapter used to
test whether a saved Greenwich trajectory can be learned by a particular MjLab/BeyondMimic robot
and controller configuration. It is intentionally secondary to conversion. Robot assets, MjLab,
policies, motion data and videos remain external. See its README for queue, evaluation, packaging
and interpretation boundaries.

## Review and extend

See [robot integration](docs/robots.md), [Python API](docs/api.md),
[architecture](docs/architecture.md) and [review](docs/review.md).
The package does not include data, pictures, videos, weights or robot meshes.
SOURCE_MANIFEST.json records the fresh reviewed source snapshot and file hashes.
''',
 'docs/architecture.md': '''# Module boundaries

| Layer | Current implementation | Swappable input |
| --- | --- | --- |
| Source reading | adapters.py / MotionClip | SOMA, BVH, native SMPL, canonical NPZ |
| Robot interface | roster + AlphaMotion descriptor builder | normalized vendor MJCF/URDF import |
| Model | externally installed AlphaMotion | checkpoint selected in backend |
| Realization | native joint projection | target axes, rest frames and limits |
| Refinement | soma_contact_refine.py | shared contact/flight weights and iteration budget |
| Evaluation | locomotion_audit, fixed_support, surface_audit, physics_audit | thresholds and saved caches |
| Delivery | compact_robot_motion_delivery, package_compact_shards | robot schema, shard size and destination |
| Optional controller QC | quality/beyondmimic | MjLab robot/controller/training configuration |
| Review | render_review, card_gallery | source-synchronized views and GIF files |

SDK orchestration is Pipeline.run / Pipeline.convert / scripts/invoke.ps1. Native geometry lives in
a small compatibility namespace `greenwich_umi_proof`; it is not the unrelated
UMI/EGO server. runtime/scripts contains only extracted camera/encoder helpers.

All full-body source loaders share MotionClip and the same model, projection and
contact refiner. Indexed SOMA batches use Pipeline.run; explicit file manifests
use Pipeline.convert. evaluation.evaluate_outputs reuses the native sole screen.
Independent native SMPL corpus quality is not established by an adapter smoke test.

BeyondMimic is downstream QC only. Its results must not be folded back into a claim that the
converter passed or failed until reference feasibility and robot/controller effects are separated.
''',
 'docs/formats.md': '''# Source adapter contract

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
''',
 'docs/robots.md': '''# Add an embodiment

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
''',
 'docs/evaluation.md': '''# Numeric evaluation and fast re-thresholding

Saved-motion evaluation is independent of generation. Legacy fixed-support
screening requires source-stationary persistent sole patches; excludes flight,
swing and rolling transitions. Default local tolerances: 15 cm/s drift,
0.5 cm penetration and 0.1 s persistence. These are configurable engineering
thresholds, not a reproduced laboratory/industry standard.

`support-screen.conservative.json` controls masks and support extraction.
`-SlipCmS`, `-PenetrationCm` and `-BadDurationSeconds` control rejection.
Unknown support is accepted by the current review policy but coverage is retained.
Changing rejection thresholds can reuse numeric caches; changing geometry,
motion or contact-mask extraction requires regeneration of the appropriate cache.

SRS statistical screening uses distinct random unique clips, explicit seed and
exact finite-population 95% intervals; qualitative GIF selection is not SRS.
Kimodo toe skating is a separate protocol selected with `-MetricProfile kimodo`;
it is a metric, not an automatic clip pass/fail criterion.

physics_audit extracts native FK/COM/inertia and solves necessary centroidal
contact-wrench feasibility with approximate friction cones. Use
`invoke_physics_audit.ps1 -Source /path/to/motion.npz -Robot my_robot
-Robots /path/to/robots.json -Output /path/to/audit`, then `-Cache` and `-Config`
for cheap threshold changes. This adapter assumes flat-floor foot support;
hands/knees/seats/object support is unmodeled. No torque, controller or hardware
execution certificate is issued by a pass.

Compact NPZ predicted contact fields retain two foot channels. Green all-surface
paint is geometric proximity, not force, pressure, fractional area or a measured
full-body contact annotation. Keep those fields and meanings separate.

## Downstream controller QC

The optional `toolkits/greenwich-soma-multibody/quality/beyondmimic` adapter trains and evaluates a
tracking policy against saved robot trajectories. Report success rate, full-duration coverage,
body/joint error, support slip proxy and actuator-force statistics together. A failure can originate
in the converted reference, robot collision geometry, controller interface, reward/contact inputs,
termination thresholds or training budget. It is not by itself proof that the Greenwich model did
not understand the source motion. A pass is not a dynamics or hardware-safety certificate.
''',
 'docs/api.md': '''# Python integration

```python
from pathlib import Path
from greenwich_motion_sdk import Pipeline, RunRequest, load_motion

pipeline = Pipeline(Path("toolkits/greenwich-soma-multibody"))
request = RunRequest(indices=(1, 2), output=Path("/path/to/results"),
                     robots=Path("/path/to/robots.json"),
                     representation="soma77", stage="Generate")
result = pipeline.run(request)

# Explicit native SMPL / generic BVH / SOMA / canonical full-body sources:
result = pipeline.convert("/path/to/sources.json", "/path/to/results",
                          "/path/to/robots.json")

# In a configured model/geometry environment, rescreen without inference:
from greenwich_motion_sdk.evaluation import evaluate_outputs
report = evaluate_outputs("/path/to/results", robot_dict,
                          "/path/to/robot.json.gz", "/path/to/evaluation",
                          slip_cm_s=15, penetration_cm=0.5, bad_duration_s=0.1)
```

Blocking orchestration can run in a GUI worker/job queue. Stages are Generate,
Audit, Render or All; persisted requests support review and errors propagate.
Evaluation modules accept saved motion arrays/manifests, not model latents.
Use `python -m greenwich_motion_sdk.MODULE --help` for exact CLI signatures.
Inspect actual adapters' signatures before passing format-specific kwargs.
''',
 'docs/batch.md': '''# Batch generation and reuse

`bulk_convert_soma.py --help` documents local source archive, metadata, bind,
roster, output, cache, batch size and free-disk guard options. Generation can
use the same direct backend or invoke.ps1; failures are bisected to isolate a
bad capture. A persisted ledger supports resume, and temporary source batches
are removed after their output has been recorded.

`--shard-count N --shard-index K` partitions work for independent workers.
Use a distinct output/ledger per worker and avoid overlapping source shards.
GPU memory and budget determine safe worker count; do not assume linear speedup.
SOMA floor/crawl-type filtering in this legacy bulk entrypoint is explicit in
GROUND_PATTERN; inspect it before choosing the corpus. It is not format-generic.

Model generation, evaluation and rendering are distinct jobs. Evaluate existing
NPZs without rerunning the model; render only clips you want to inspect. No
account-specific Modal launch/session scripts are redistributed.
''',
 'docs/review.md': '''# Synchronized review exports

`Render -ReviewManifest ... -ViewerDescriptor ... -ViewerXml ...
-ReviewSourceComparison` reads saved compact motion with native meshes and exact
source timing. It does not alter or regenerate the motion.

`Gallery -GalleryClips ... -GalleryHideUI -GalleryContactGlow` exports scene-only
comparison GIFs. Add `-GallerySeparate` for one GIF per action; width and duration
are configurable. Green contact regions and ground grid remain visible.
Do not judge published motion validity from solving success alone. Inspect
source/robot timing, left/right wrist direction, whole-body posture and contacts.
No generated review assets are part of this source repository.
''',
}


DOCUMENTS['SDK.md'] = '''# SDK guide

See [Python API](docs/api.md), [source contracts](docs/formats.md),
[robot descriptors](docs/robots.md), [evaluation](docs/evaluation.md) and
[architecture](docs/architecture.md). `greenwich_motion_sdk.Pipeline` is the
public orchestration entrypoint; `register_adapter` extends source formats.
All full-body inputs reuse the same native projection and contact refiner.

Tests: `python -m pytest tests -q`. Manual GPU integration:
`python tests/smoke_full_body.py --help`. The manual SMPL-format fixture is
derived from SOMA, not an independent native SMPL benchmark.
'''
DOCUMENTS[KIT+'/SKILL.md'] = '''---
name: alphamotion-full-body-pipeline
description: Reusable full-body conversion, contact refinement and saved-output evaluation.
---

Read SDK.md and docs/formats.md before extending adapters or robot mappings.
Use scripts/invoke.ps1 or greenwich_motion_sdk.Pipeline, never duplicate loaders
or solvers per task. Configure config/local-paths.json from its example.
Models, source data and robot assets are external; never commit generated data.
Use Generate for indexed SOMA; Convert for explicit full-body source manifests.
Use Evaluate to rescreen saved noncompact Convert outputs without inference.
Compact corpus traces use locomotion_audit and independent source manifests.
Keep source floor, unit, rotation and role mappings explicit. No trajectory
resizing or hidden floor estimation. Shared contact refinement is in
scripts/soma_contact_refine.py with config/contact.default.json.
The locomotion entrypoint derives foot and hand support from source semantics;
root height follows the active support family. Fixed wrist/TCP endpoints are
used when a robot has no native hand mesh, without per-robot height offsets.
Support/penetration screens are necessary checks, not dynamics certificates.
Use scripts/compact_robot_motion_delivery.py and package_compact_shards.py for robot-motion
delivery. BeyondMimic under quality/beyondmimic is optional downstream QC, never the primary
conversion path or a substitute for reference-feasibility screening.
EGO/UMI sparse pipelines belong to the separate legacy toolkit.
'''


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sdk-root',type=Path,required=True);p.add_argument('--runtime',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    build(a.sdk_root,a.runtime,a.output)


if __name__=='__main__':main()
