# Module boundaries

| Layer | Current implementation | Swappable input |
| --- | --- | --- |
| Source reading | adapters.py / MotionClip | SOMA, BVH, native SMPL, canonical NPZ |
| Robot interface | roster + AlphaMotion descriptor builder | normalized vendor MJCF/URDF import |
| Model | externally installed AlphaMotion | checkpoint selected in backend |
| Realization | native joint projection | target axes, rest frames and limits |
| Refinement | soma_contact_refine.py | shared foot/hand support constraints, contact/flight weights and iteration budget |
| Evaluation | locomotion_audit, fixed_support, surface_audit, physics_audit | thresholds and saved caches |
| Delivery | compact_robot_motion_delivery, package_compact_shards | robot schema, shard size and destination |
| Optional controller QC | quality/beyondmimic | MjLab robot/controller/training configuration |
| Review | render_review, card_gallery | source-synchronized views and GIF files |

SDK orchestration is Pipeline.run / Pipeline.convert / scripts/invoke.ps1. Native geometry lives in
a small compatibility namespace `greenwich_umi_proof`; it is not the unrelated
UMI/EGO server. runtime/scripts contains only extracted camera/encoder helpers.

All full-body source loaders share MotionClip and the same model, projection and
contact refiner. Root height is projected from the active support family rather
than always from the feet; fixed hand TCPs are valid semantic support endpoints
when a robot has no hand mesh. Indexed SOMA batches use Pipeline.run; explicit file manifests
use Pipeline.convert. evaluation.evaluate_outputs reuses the native sole screen.
Independent native SMPL corpus quality is not established by an adapter smoke test.

BeyondMimic is downstream QC only. Its results must not be folded back into a claim that the
converter passed or failed until reference feasibility and robot/controller effects are separated.
