# SDK guide

See [Python API](docs/api.md), [source contracts](docs/formats.md),
[robot descriptors](docs/robots.md), [evaluation](docs/evaluation.md) and
[architecture](docs/architecture.md). `greenwich_motion_sdk.Pipeline` is the
public orchestration entrypoint; `register_adapter` extends source formats.
All full-body inputs reuse the same native projection and contact refiner.

Tests: `python -m pytest tests -q`. Manual GPU integration:
`python tests/smoke_full_body.py --help`. The manual SMPL-format fixture is
derived from SOMA, not an independent native SMPL benchmark.
