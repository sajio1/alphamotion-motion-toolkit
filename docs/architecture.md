# Module boundaries

| Layer | Current implementation | Swappable input |
| --- | --- | --- |
| Source reading | adapters.py / MotionClip | SOMA, BVH, native SMPL, canonical NPZ |
| Robot interface | roster + AlphaMotion descriptor builder | normalized vendor MJCF/URDF import |
| Model | externally installed AlphaMotion | checkpoint selected in backend |
| Realization | native joint projection | target axes, rest frames and limits |
| Refinement | soma_contact_refine.py | shared contact/flight weights and iteration budget |
| Evaluation | locomotion_audit, fixed_support, surface_audit, physics_audit | thresholds and saved caches |
| Review | render_review, card_gallery | source-synchronized views and GIF files |

SDK orchestration is Pipeline.run / Pipeline.convert / scripts/invoke.ps1. Native geometry lives in
a small compatibility namespace `greenwich_umi_proof`; it is not the unrelated
UMI/EGO server. runtime/scripts contains only extracted camera/encoder helpers.

All full-body source loaders share MotionClip and the same model, projection and
contact refiner. Indexed SOMA batches use Pipeline.run; explicit file manifests
use Pipeline.convert. evaluation.evaluate_outputs reuses the native sole screen.
Independent native SMPL corpus quality is not established by an adapter smoke test.
