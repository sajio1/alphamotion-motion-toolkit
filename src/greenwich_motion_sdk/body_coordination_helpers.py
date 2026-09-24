"""Shared, robot-agnostic arm continuity helpers for staged refinement."""

import numpy as np


def arm_branches(parents, wrists):
    chains = []
    for wrist in wrists:
        chain = set()
        joint = int(wrist)
        while joint >= 0:
            chain.add(joint)
            joint = int(parents[joint])
        chains.append(chain)
    shared = chains[0] & chains[1]
    mask = np.zeros(len(parents), bool)
    mask[list((chains[0] | chains[1]) - shared)] = True
    return mask


def repair_short_excursions(q, fps, groups, *, window_s=.6,
                            excursion_deg=55., max_gap_s=.4, margin_frames=1):
    """Bridge brief arm-angle outliers; leave sustained gestures unchanged."""
    from scipy.ndimage import median_filter

    if fps <= 0 or window_s <= 0 or excursion_deg <= 0 or max_gap_s <= 0:
        raise ValueError('Positive timing and thresholds required')
    q = np.asarray(q)
    result = q.copy()
    events = []
    width = max(3, round(window_s * fps) | 1)
    for group in groups:
        group = np.asarray(group, bool)
        median = median_filter(q[:, group], size=(width, 1, 1), mode='nearest')
        deviation = np.max(np.abs(q[:, group] - median), axis=(1, 2))
        flags = deviation > np.deg2rad(excursion_deg)
        transitions = np.diff(np.r_[False, flags, False].astype(int))
        spans = list(zip(np.flatnonzero(transitions == 1),
                         np.flatnonzero(transitions == -1)))
        expanded = []
        for start, end in spans:
            start = max(0, start - margin_frames)
            end = min(len(q), end + margin_frames)
            if expanded and start <= expanded[-1][1]:
                expanded[-1] = (expanded[-1][0], end)
            else:
                expanded.append((start, end))
        for start, end in expanded:
            if start == 0 or end == len(q) or (end - start) / fps > max_gap_s:
                continue
            bridge = np.max(np.abs(q[end, group] - q[start - 1, group]))
            if bridge > np.deg2rad(excursion_deg):
                continue
            alpha = np.arange(1, end - start + 1) / (end - start + 1)
            result[start:end, group] = (
                q[start - 1, group][None] * (1 - alpha[:, None, None])
                + q[end, group][None] * alpha[:, None, None]
            )
            events.append({
                'start_frame': int(start),
                'end_frame_exclusive': int(end),
                'duration_s': (end - start) / fps,
                'max_deviation_deg': float(np.rad2deg(deviation[start:end].max())),
                'arm_joints': np.flatnonzero(group).tolist(),
            })
    return result, events
