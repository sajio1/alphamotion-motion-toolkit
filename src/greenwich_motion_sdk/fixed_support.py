"""Conservative fixed-support patch drift diagnostics, not a dynamics verdict."""
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import numpy as np
from scipy.ndimage import minimum_filter1d
from .physics_audit import intervals


@dataclass
class SupportConfig:
    source_height_cm: float = 8.
    source_speed_cm_s: float = 2.
    source_position_tolerance_cm: float = .5
    transition_margin_s: float = .1
    minimum_support_s: float = .2
    target_tilt_speed_deg_s: float = 20.
    contact_band_cm: float = 1.
    patch_depth_cm: float = .5
    minimum_patch_vertices: int = 3

    @classmethod
    def load(cls, path):
        config=cls(**json.loads(Path(path).read_text(encoding='utf-8')))
        if any(not np.isfinite(v) or v<=0 for v in asdict(config).values()):
            raise ValueError('Fixed-support settings must be finite and positive')
        return config


def measure_fixed_support(landmark, vertices, rotation, fps, config,
                          slip_cm_s=15., penetration_cm=.5, bad_duration_s=.1):
    """Infer stationary reference support, then track persistent material patches.

Both source ankle/toe landmarks must be nearly stationary in 3D, excluding
deliberate glides and heel/toe roll. Target ground proximity, a stable foot
orientation and a persistent mesh patch provide conservative geometric evidence.
Never infer support from low TARGET translation speed. A partial sole can qualify.
No pressure/load-bearing inference and no all-body success label is returned.
"""
    T=len(landmark);margin=round(config.transition_margin_s*fps)
    source_speed=np.full(T,np.inf)
    source_speed[1:]=np.linalg.norm(np.diff(landmark,axis=0),axis=-1).max(1)*fps
    candidate=(landmark[:,:,1].min(1)<config.source_height_cm)&(source_speed<config.source_speed_cm_s)
    candidate=minimum_filter1d(candidate.astype(np.uint8),size=2*margin+1,mode='constant')>0
    planted=np.zeros(T,bool);source_episodes=[];excluded_source_drift=0
    for event in intervals(candidate,fps,config.minimum_support_s):
        a,b=event['start_frame'],event['end_frame_exclusive']
        drift=float(np.linalg.norm(landmark[a:b]-landmark[a],axis=-1).max())
        if drift>config.source_position_tolerance_cm:
            excluded_source_drift+=b-a;continue
        planted[a:b]=True
        source_episodes.append({**event,'max_source_landmark_displacement_cm':drift})
    # Exclude rolling transitions, retain yaw slip of a planted material patch.
    up=rotation[:,:,1]
    angular_speed=np.full(T,np.inf)
    angular_speed[1:]=np.degrees(np.arccos(np.clip(np.sum(up[1:]*up[:-1],axis=1),-1,1)))*fps
    heights=vertices[:,:,1].min(1)
    near=(vertices[:,:,1]<=config.contact_band_cm)&(vertices[:,:,1]>=-penetration_cm)
    near&=vertices[:,:,1]<=heights[:,None]+config.patch_depth_cm
    stable_geometry=(near.sum(1)>=config.minimum_patch_vertices)&(angular_speed<config.target_tilt_speed_deg_s)
    stable_geometry=minimum_filter1d(stable_geometry.astype(np.uint8),size=2*margin+1,mode='constant')>0
    confirmed=planted&stable_geometry
    speed=np.full(T,np.nan);episodes=[];evaluated=np.zeros(T,bool);no_persistent_patch_frames=0
    for event in intervals(confirmed,fps,config.minimum_support_s):
        a,b=event['start_frame'],event['end_frame_exclusive']
        patch=np.flatnonzero(near[a:b].all(0))
        if len(patch)<config.minimum_patch_vertices:
            no_persistent_patch_frames+=b-a;continue
        xy=vertices[a:b,patch][:,:,[0,2]]
        velocity=np.median(np.linalg.norm(np.diff(xy,axis=0),axis=-1),axis=1)*fps
        speed[a+1:b]=velocity;evaluated[a+1:b]=True
        displacement=np.median(np.linalg.norm(xy-xy[0],axis=-1),axis=1)
        episodes.append({**event,'patch_vertices':len(patch),
            'patch_max_median_displacement_cm':float(displacement.max()),
            'patch_median_net_displacement_cm':float(np.median(np.linalg.norm(xy[-1]-xy[0],axis=-1))),
            'patch_median_path_length_cm':float(np.median(np.linalg.norm(np.diff(xy,axis=0),axis=-1).sum(0))),
            'patch_speed_p95_cm_s':float(np.percentile(velocity,95)),
            'max_source_landmark_displacement_cm':float(np.linalg.norm(landmark[a:b]-landmark[a],axis=-1).max())})
    flags=intervals(np.isfinite(speed)&(np.nan_to_num(speed)>slip_cm_s),fps,bad_duration_s)
    for event in flags:
        a,b=event['start_frame'],event['end_frame_exclusive']
        event['median_patch_speed_cm_s']=float(np.median(speed[a:b]))
    values=speed[np.isfinite(speed)]
    return {'fixed_source_support_frames':int(planted.sum()),
        'fixed_patch_evaluated_frames':int(evaluated.sum()),
        'fixed_source_support_episodes':source_episodes,'fixed_patch_support_episodes':episodes,
        'fixed_patch_slip_events':flags,
        'fixed_patch_speed_p95_cm_s':float(np.percentile(values,95)) if len(values) else None,
        'fixed_patch_speed_max_cm_s':float(values.max()) if len(values) else None,
        'fixed_patch_max_median_displacement_cm':max((e['patch_max_median_displacement_cm'] for e in episodes),default=None),
        'excluded_source_drift_frames':excluded_source_drift,
        'excluded_target_roll_or_off_ground_frames':int((planted&~stable_geometry).sum()),
        'no_persistent_patch_frames':no_persistent_patch_frames},speed,evaluated
