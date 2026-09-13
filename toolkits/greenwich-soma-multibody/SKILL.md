---
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
Support/penetration screens are necessary checks, not dynamics certificates.
EGO/UMI sparse pipelines belong to the separate legacy toolkit.
