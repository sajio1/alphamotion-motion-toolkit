# Numeric evaluation and fast re-thresholding

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
