"""NumPy adapter for NVIDIA Kimodo's toe-height foot skating metrics.

Reference: nv-tlabs/kimodo, kimodo/metrics/foot_skate.py. Uses Y-up metres,
3D forward-difference speed, strict thresholds and the official epsilon.
Robot toe landmarks must be declared by the embodiment adapter.
"""
import numpy as np

REFERENCE='https://github.com/nv-tlabs/kimodo/blob/main/kimodo/metrics/foot_skate.py'


def toe_metrics(toes_m, fps):
    toes=np.asarray(toes_m,dtype=np.float64)
    if toes.ndim!=3 or toes.shape[1:]!=(2,3) or len(toes)<2 or not np.isfinite(toes).all() or fps<=0:
        raise ValueError('Expected finite [T,2,3] toe positions in Y-up metres and positive fps')
    velocity=np.linalg.norm(np.diff(toes,axis=0),axis=-1)*fps
    low=toes[:,:,1]<.05
    contact=low[:-1]&low[1:]
    skating=contact&(velocity>.2)
    mean_mask=low[:-1]
    metrics={'foot_skate_ratio':float(skating.sum()/(contact.sum()+1e-6)),
        'foot_skate_from_height':float((velocity*mean_mask).sum()/(mean_mask.sum()+1e-6)),
        'contact_toe_frame_pairs':int(contact.sum()),'skating_toe_frame_pairs':int(skating.sum()),
        'height_mean_toe_frame_pairs':int(mean_mask.sum()),
        'height_mean_velocity_sum_m_s':float((velocity*mean_mask).sum()),
        'total_toe_frame_pairs':int(contact.size)}
    return metrics,velocity,contact,skating
