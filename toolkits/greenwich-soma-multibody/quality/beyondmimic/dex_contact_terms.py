"""Robot-independent reference-contact terms for BeyondMimic tracking."""

from __future__ import annotations

import numpy as np
import torch


class ReferenceContact:
    def __init__(self, cfg, env):
        del cfg
        command = env.command_manager.get_term("motion")
        with np.load(command.cfg.motion_file, allow_pickle=False) as motion:
            labels = np.asarray(motion["reference_contact"], dtype=np.float32)
            names = tuple(str(value) for value in motion["reference_contact_names"])
        if labels.ndim != 2 or labels.shape[0] != command.motion.time_step_total:
            raise ValueError("Reference-contact frames do not match motion frames")
        if not set(np.unique(labels)).issubset({0.0, 1.0}):
            raise ValueError("Reference-contact labels must be binary")
        self.labels = torch.as_tensor(labels, device=env.device)
        self.names = names
        self.command = command

    def current(self) -> torch.Tensor:
        index = self.command.time_steps.clamp(max=self.labels.shape[0] - 1)
        return self.labels[index]


class reference_contact_observation(ReferenceContact):
    def __call__(self, env, future_offsets=(0, 2, 5, 10)) -> torch.Tensor:
        del env
        values = []
        for offset in future_offsets:
            index = (self.command.time_steps + int(offset)).clamp(
                max=self.labels.shape[0] - 1
            )
            values.append(self.labels[index])
        return torch.cat(values, dim=-1)


class reference_contact_match(ReferenceContact):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.sensor = env.scene["reference_ground_contact"]
        if tuple(self.sensor.primary_names) != self.names:
            raise ValueError(
                f"Contact sensor order {self.sensor.primary_names} != labels {self.names}"
            )

    def __call__(self, env) -> torch.Tensor:
        del env
        target = self.current() > 0.5
        found = self.sensor.data.found
        if found is None:
            raise RuntimeError("reference_ground_contact has no found field")
        actual = found > 0
        desired_count = target.sum(dim=1)
        undesired = ~target
        positive = (actual & target).sum(dim=1) / desired_count.clamp(min=1)
        negative = 1.0 - (actual & undesired).sum(dim=1) / undesired.sum(dim=1).clamp(min=1)
        return torch.where(desired_count > 0, 0.7 * positive + 0.3 * negative, negative)
