"""Format-independent full-body source manifests for the shared native backend."""
import json
from pathlib import Path
import numpy as np
from .adapters import load_motion
from .motion import CoordinateFrame


def resolve(base, value):
    path=Path(value)
    return path if path.is_absolute() else (base/path).resolve()


def read_sources(manifest):
    path=Path(manifest).resolve(); data=json.loads(path.read_text(encoding='utf-8'))
    if data.get('schema')!='alphamotion.sources.v1' or not data.get('sources'):
        raise ValueError('Expected nonempty alphamotion.sources.v1 manifest')
    rows=[];seen=set()
    for index,source in enumerate(data['sources'],1):
        stem=source['id']
        if not stem or stem in seen or '/' in stem or '\\' in stem or stem in ('.','..'):
            raise ValueError('Source IDs must be unique safe basenames')
        seen.add(stem);fmt=source['format']; options=dict(source.get('options',{}))
        if source.get('options_file'):
            profile=resolve(path.parent,source['options_file'])
            options={**json.loads(profile.read_text(encoding='utf-8')),**options}
            profile_base=profile.parent
        else:profile_base=path.parent
        if 'frame' in options:options['frame']=CoordinateFrame(**options['frame'])
        if 'bind_path' in options:options['bind_path']=resolve(profile_base,options['bind_path'])
        clip=load_motion(resolve(path.parent,source['path']),format=fmt,**options)
        # Explicit semantics are an interface mapping, not a pose/IK adjustment.
        roles=source.get('roles')
        pairs=source.get('foot_pairs')
        if fmt=='soma':
            roles=roles or ['Hips','Head','LeftHand','RightHand','LeftFoot','RightFoot']
            pairs=pairs or [['LeftFoot','LeftToeBase'],['RightFoot','RightToeBase']]
        if not roles or len(roles)!=6 or not pairs or len(pairs)!=2 or any(len(p)!=2 for p in pairs):
            raise ValueError('Provide six roles (root/head/L-hand/R-hand/L-foot/R-foot) and two ankle/toe foot_pairs')
        role_indices=[clip.names.index(n) for n in roles]
        foot_indices=[clip.names.index(n) for pair in pairs for n in pair]
        if role_indices[0]!=0:raise ValueError('Source root role must be the canonical hierarchy root')
        clear=np.asarray(source.get('foot_landmark_height_cm',[0.,0.]),float)
        if clear.shape!=(2,) or not np.isfinite(clear).all() or np.any(clear<0):raise ValueError('Invalid neutral ground landmark offsets')
        rows.append({'index':index,'stem':stem,'semantic_type':source.get('label',stem),
                     'source_format':fmt,'source_path':str(resolve(path.parent,source['path'])),
                     'source_role_indices':role_indices,'foot_indices':foot_indices,
                     'foot_landmark_height_cm':clear.tolist(),'_clip':clip})
    return rows


def sample_clip(clip,fps,max_seconds):
    if len(clip.timestamps_s)<3:raise ValueError('At least three source frames required')
    delta=np.diff(clip.timestamps_s);dt=float(np.median(delta))
    if not np.allclose(delta,dt,atol=1e-6,rtol=1e-4):raise ValueError('Uniform source timing required; resample explicitly before conversion')
    stride=round(1/dt/fps)
    if stride<1 or abs(1/dt/stride-fps)>.02:raise ValueError('Requested FPS requires explicit resampling')
    take=np.flatnonzero(clip.timestamps_s-clip.timestamps_s[0]<max_seconds)[::stride]
    if len(take)<3:raise ValueError('Requested window needs at least three frames')
    return take,dt
