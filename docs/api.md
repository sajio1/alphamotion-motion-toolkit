# Python integration

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
