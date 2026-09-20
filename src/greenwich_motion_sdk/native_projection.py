"""Native joint projection with optional dimensionless model-geometry fidelity.

Targets come from decoded robot rotations and its own skeleton, never source
human end-effectors. Zero weight preserves the historical orientation-only fit.
"""
import numpy as np


def geometry_weight(spec, strength=0.):
    if not np.isfinite(strength) or strength < 0:
        raise ValueError('Projection geometry strength must be finite and nonnegative')
    length=np.zeros(len(spec.parents),dtype=float)
    for j,p in enumerate(spec.parents):
        if p>=0:length[j]=length[p]+np.linalg.norm(spec.rest_offsets[j])
    scale=float(length.max())
    if scale<=0:raise ValueError('Nonzero native skeleton length required')
    return float(strength)/scale**2,scale


def project(raw,spec,dof,rest,*,geometry_strength=0.,iterations=12,method='global'):
    from alphamotion.engine import constraints as c
    if method not in ('global','greedy'):raise ValueError('Projection method must be global or greedy')
    if method=='greedy' and geometry_strength:raise ValueError('Greedy native projection cannot use the global geometry term')
    weight,scale=geometry_weight(spec,geometry_strength)
    q,active=c.fit_angles(raw,spec,dof,rest=rest,method=method,lm_iters=iterations,soft_margin=1.,clamp=True,w_pos=weight)
    return q,active,{'geometry_strength':geometry_strength,'geometry_weight_cm_minus2':weight,
        'native_chain_scale_cm':scale,'iterations':iterations if method=='global' else 0,'method':method,
        'target':'decoded robot FK, root-relative; no human position targets'}
