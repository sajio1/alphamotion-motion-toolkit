"""Format adapters. Units, floor and bind are explicit, not inferred per action."""
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from .motion import MotionClip,CoordinateFrame
from ._bvh import load_bvh

_ADAPTERS={}

def register_adapter(name,loader):
    if name in _ADAPTERS:raise ValueError(f'Adapter already registered: {name}')
    _ADAPTERS[name]=loader

def load_motion(path,*,format,**options):
    if format not in _ADAPTERS:raise ValueError(f'Unknown adapter {format}; registered: {list(_ADAPTERS)}')
    return _ADAPTERS[format](Path(path),**options).validate()

def bvh(path,*,bind_path,frame:CoordinateFrame,position_channels='replace',remove_identity_root=False):
    frame.validate()
    if position_channels not in ('replace','add'):raise ValueError('Specify position channel semantics: replace or add')
    motion=load_bvh(path,position_channels=position_channels)
    bind=load_bvh(bind_path,position_channels=position_channels)
    if motion['names']!=bind['names'] or not np.array_equal(motion['parents'],bind['parents']):
        raise ValueError('Bind hierarchy must match motion')
    start=0
    if remove_identity_root:
        if not np.allclose(motion['positions'][:,0],0,atol=1e-7) or not np.allclose(motion['global_rot'][:,0],np.eye(3),atol=1e-7):
            raise ValueError('Cannot remove a moving scene root')
        if np.count_nonzero(motion['parents']==0)!=1:raise ValueError('Scene root must have exactly one child')
        if not np.allclose(bind['positions'][:,0],0) or not np.allclose(bind['global_rot'][:,0],np.eye(3)):
            raise ValueError('Bind scene root is not identity')
        start=1
    names=motion['names'][start:];parents=motion['parents'][start:]-start
    B=np.asarray(frame.basis_to_yup)
    positions=(motion['positions'][:,start:]@B.T)*frame.cm_per_unit
    positions[:,:,1]-=frame.ground_y_cm
    bp=(bind['positions'][0,start:]@B.T)*frame.cm_per_unit;bp[:,1]-=frame.ground_y_cm
    offsets=bp.copy()
    for j,p in enumerate(parents):
        if p>=0:offsets[j]=bp[j]-bp[p]
    delta=motion['global_rot'][:,start:]@bind['global_rot'][0,start:].swapaxes(-1,-2)
    rotation=B@delta@B.T
    return MotionClip(names,parents,offsets,rotation,positions,
        np.arange(len(positions))*motion['dt'],{'source':str(path),'bind':str(bind_path),'format':'bvh',
        'units':'cm','up':'Y','ground_y_cm':0,'position_channels':position_channels,'removed_identity_root':remove_identity_root})

def soma(path,*,bind_path):
    clip=bvh(path,bind_path=bind_path,frame=CoordinateFrame(np.eye(3),1.,0.),remove_identity_root=True)
    if len(clip.names)!=77 or clip.names[0]!='Hips':raise ValueError('Expected SOMA77 skeleton')
    clip.metadata['skeleton']='soma77';return clip

def smpl(path,*,names,parents,rest_offsets,frame:CoordinateFrame,pelvis_offset_source_units):
    """Native SMPL-family axis-angle NPZ adapter.

    Caller provides canonical SMPL rest joint offsets in SOURCE coordinates and
    the shaped pelvis offset J0. `trans` is SMPL model translation, not pelvis.
    Body prefix length is determined by the explicit skeleton. No model fitting.
    This interface is unit tested; native-dataset robot accuracy is not yet measured.
    """
    frame.validate();J=len(names)
    with np.load(path,allow_pickle=False) as f:
        poses=np.asarray(f['poses']);trans=np.asarray(f['trans'])
        fk='mocap_framerate' if 'mocap_framerate' in f else 'mocap_frame_rate'
        fps=float(f[fk])
    if poses.ndim!=2 or poses.shape[1]<J*3 or trans.shape!=(len(poses),3):raise ValueError('Invalid SMPL poses/trans')
    if not np.isfinite(fps) or fps<=0:raise ValueError('Invalid sample rate')
    local=Rotation.from_rotvec(poses[:,:J*3].reshape(-1,3)).as_matrix().reshape(-1,J,3,3)
    global_R=local.copy()
    for j,p in enumerate(parents):
        if p>=0:global_R[:,j]=global_R[:,p]@local[:,j]
    B=np.asarray(frame.basis_to_yup);global_R=B@global_R@B.T
    offsets=np.asarray(rest_offsets)@B.T*frame.cm_per_unit
    positions=np.zeros((len(poses),J,3))
    positions[:,0]=(trans+np.asarray(pelvis_offset_source_units))@B.T*frame.cm_per_unit
    positions[:,0,1]-=frame.ground_y_cm
    for j,p in enumerate(parents):
        if p>=0:positions[:,j]=positions[:,p]+np.einsum('tij,j->ti',global_R[:,p],offsets[j])
    return MotionClip(list(names),np.asarray(parents),offsets,global_R,positions,np.arange(len(poses))/fps,
        {'source':str(path),'format':'smpl-axis-angle','units':'cm','up':'Y','ground_y_cm':0,'pelvis_translation':'trans + supplied shaped J0'})

register_adapter('bvh',bvh)
register_adapter('soma',soma)
register_adapter('smpl',smpl)
register_adapter('canonical',lambda path:MotionClip.load(path))
