# BeyondMimic Quality-Evaluation Adapter

This is a secondary QC adapter for Greenwich-generated motions, not the Greenwich conversion
pipeline. Its results measure trainability under one simulator/controller configuration and must
be interpreted together with reference-trajectory feasibility checks.

This directory intentionally contains no instance IP addresses, dated queue files, fixed local
delivery paths, model weights, or robot assets. Run-specific files belong in an external run
directory.

The adapter depends on an external MjLab checkout and compatible BeyondMimic task definitions.
Set `GREENWICH_BM_ASSET_ROOT` to a directory with the following locally licensed assets:

```text
<asset-root>/a3/a3_t3d0_native.xml
<asset-root>/a3/meshes/...
<asset-root>/booster/booster_t1_27dof_native.xml
<asset-root>/booster/meshes/...
```

G1 uses MjLab's installed native G1 configuration. A3 and Booster use a shared effort-normalized
adapter rule; no motion-specific reward or controller tuning is applied.

## Core flow

1. Convert a Greenwich motion into the native MjLab format with
   `greenwich_to_mjlab_native.py` (or import a G1 CSV with `g1_csv_to_mjlab.py`).
2. Put jobs in a JSON list. Each job needs `clip`, `robot`, and `motion`; `envs` is optional.
3. Start one queue per GPU with `sim9_queue.py`.
4. Inspect progress with `summarize_sim_runs.py`.
5. Package completed results with `package_sim_delivery.py` and extract verified videos with
   `collect_delivery_videos.py`.

Supported robot identifiers are `a3`, `g1`, and `booster_t1_29`.

Example queue file:

```json
[
  {
    "clip": "jump_001",
    "robot": "g1",
    "motion": "/data/motions/jump_001/motion.npz",
    "envs": 2048
  }
]
```

Example launch:

```bash
CUDA_VISIBLE_DEVICES=0 python sim9_queue.py \
  --jobs /data/run/queue-gpu0.json \
  --results-root /data/run/results \
  --sim9-root /home/ubuntu/sim9 \
  --mjlab-root /home/ubuntu/mjlab \
  --envs 2048 --iterations 5000
```

Queues are restart-safe: completed jobs with a verified video are skipped, checkpointed jobs
resume, and an incomplete directory without a checkpoint is not overwritten. `run_queue_after.py`
can chain a second queue after a first queue reaches a terminal state.

## Video and delivery defaults

Formal evaluation video defaults are 1280x720, H.264 CRF 18. They can be overridden with
`--video-width`, `--video-height`, and `--video-crf` at queue, worker, or evaluator level.

```bash
python package_sim_delivery.py --output /data/delivery \
  --group g1 /data/g1-results /data/g1-motions \
  --group cross /data/cross-results /data/cross-motions

python collect_delivery_videos.py --delivery /data/delivery --output /data/videos
```

`package_sim_delivery.py` keeps best/last policies, verified videos, result and evaluation JSON,
TensorBoard events, logs, configs, queue state, and input robot motions. It intentionally omits
intermediate checkpoints.

## Review utilities

- `compose_variant_videos.py`: pair two filename variants and render synchronized side-by-side
  videos. The shorter clip freezes on its last frame so timing remains comparable.
- `video_contact_sheet.py`: sample a video without loading every frame into memory.
- `validate_benchmark_bundle.py`: validate prepared benchmark inputs.
- `smoke_native.py`: small environment smoke test.

Example side-by-side comparison:

```powershell
python compose_variant_videos.py `
  --source D:\videos\G1 `
  --output D:\videos\G1-comparison `
  --prefix benchmark_ `
  --left-suffix __greenwich__g1 `
  --right-suffix __retarget__g1 `
  --left-label "Greenwich Generated" `
  --right-label "Traditional Retargeting"
```

`dex_contact_terms.py` contains the reusable robot-independent reference-contact observation and
reward term. The input motion must already contain `reference_contact` and
`reference_contact_names`; this module does not infer or fabricate contact labels.

## Interpretation boundary

A failed training run is not automatically a failed Greenwich conversion. Separate reference
feasibility (collision penetration, discontinuity, limits and contact sequence) from controller
capability (actuator limits, action scaling, exploration and reward). Likewise, a successful policy
is evidence of trainability in this configuration, not a hardware safety certificate.
