"""Lossless compact artifact reader for playback; no fitting or grounding."""
from io import BytesIO
from pathlib import Path
import zipfile
import numpy as np

SCHEMA = 'greenwich.soma18.compact.v1'

def load_compact(path, *, member=None, robot='h2'):
    """Read one NPZ, optionally directly from an export ZIP without extraction."""
    if member is not None:
        with zipfile.ZipFile(path) as archive:
            if not member.endswith('.npz'): raise ValueError('Select an NPZ member')
            stream = BytesIO(archive.read(member))
    else:
        stream = path if hasattr(path,'read') else Path(path)
    with np.load(stream, allow_pickle=False) as data:
        if str(data['schema'].item()) != SCHEMA: raise ValueError('Unsupported compact schema')
        out = {key: data[f'{robot}__{key}'].copy() for key in
               ('q', 'root_rot6d', 'root_t_cm', 'joint_names', 'sole_height_cm', 'model_contact')}
        out['fps'] = float(data['fps'].item())
        out['stem'] = str(data['stem'].item())
    t, j, slots = out['q'].shape
    expected = {'root_rot6d': (t,6), 'root_t_cm': (t,3), 'joint_names': (j,),
                'sole_height_cm': (t,2), 'model_contact': (t,2)}
    if t < 1 or slots != 3 or not np.isfinite(out['fps']) or out['fps'] <= 0:
        raise ValueError('Invalid frame count, angle slots or fps')
    for key, shape in expected.items():
        if out[key].shape != shape: raise ValueError(f'Invalid {key} shape')
    for key in ('q','root_rot6d','root_t_cm','sole_height_cm'):
        if not np.isfinite(out[key]).all(): raise ValueError(f'Nonfinite {key}')
    return out

def rot6d_matrix(values):
    x = np.asarray(values, dtype=float)
    a, b = x[..., :3], x[..., 3:]
    n = np.linalg.norm(a, axis=-1, keepdims=True)
    if np.any(n < 1e-8): raise ValueError('Degenerate root rot6d')
    a = a / n
    b = b - (a*b).sum(-1, keepdims=True)*a
    n = np.linalg.norm(b, axis=-1, keepdims=True)
    if np.any(n < 1e-8): raise ValueError('Degenerate root rot6d')
    b = b/n
    return np.stack((a,b,np.cross(a,b)), axis=-1)

def native_visual_frame(descriptor, q, root_t_cm, root_rot6d):
    """Replay the unmerged MJCF hinge tree exported by the viewer toolkit.

    Returns visual geom positions (meters, Y up) and rotation matrices. Native
    intermediate motor links are preserved rather than driven by merged joints.
    """
    from scipy.spatial.transform import Rotation
    positions, rotations = [], []
    for body in descriptor['bodies']:
        p = np.asarray(body['position'], float).copy()
        r = np.asarray(body['rotation'], float).reshape(3,3).copy()
        for joint in body['joints']:
            slot = joint['slot']
            a = np.asarray(joint['position'])
            if 'fixed_rotation' in joint:
                jr = np.asarray(joint['fixed_rotation']).reshape(3,3)
            else:
                angle = float(q[slot[0],slot[1]]) if slot is not None else joint['angle']
                jr = Rotation.from_rotvec(np.asarray(joint['axis'])*angle).as_matrix()
            p += r @ (a-jr@a)
            r = r @ jr
        parent = body['parent']
        if parent >= 0:
            p = positions[parent] + rotations[parent]@p
            r = rotations[parent]@r
        positions.append(p); rotations.append(r)
    basis = np.asarray(descriptor['zup_to_yup']).reshape(3,3)
    delta = rot6d_matrix(root_rot6d) @ basis @ rotations[descriptor['root_frame']].T
    root = np.asarray(root_t_cm)/100
    origin = positions[descriptor['root_body']]
    visual_p, visual_r = [], []
    for geom in descriptor['geoms']:
        b = geom['body']
        gp = positions[b] + rotations[b]@np.asarray(geom['position'])
        gr = rotations[b]@np.asarray(geom['rotation']).reshape(3,3)
        visual_p.append(root + delta@(gp-origin)); visual_r.append(delta@gr)
    return np.asarray(visual_p), np.asarray(visual_r)


def native_visual_motion(descriptor, q, root_t_cm, root_rot6d):
    """Exact native visual FK, vectorized across timestamps; metres, Y up.

    Identical tree, hinge pivots and root-frame correction to native_visual_frame.
    No grounding, scaling or inference is applied.
    """
    from scipy.spatial.transform import Rotation
    q=np.asarray(q);T=len(q);positions=[];rotations=[]
    for body in descriptor['bodies']:
        p=np.broadcast_to(np.asarray(body['position'],float),(T,3)).copy()
        r=np.broadcast_to(np.asarray(body['rotation'],float).reshape(3,3),(T,3,3)).copy()
        for joint in body['joints']:
            a=np.asarray(joint['position']);slot=joint['slot']
            if 'fixed_rotation' in joint:
                jr=np.broadcast_to(np.asarray(joint['fixed_rotation']).reshape(3,3),(T,3,3))
            else:
                angle=q[:,slot[0],slot[1]] if slot is not None else np.full(T,joint['angle'])
                jr=Rotation.from_rotvec(angle[:,None]*np.asarray(joint['axis'])).as_matrix()
            p+=np.einsum('tij,tj->ti',r,a-np.einsum('tij,j->ti',jr,a))
            r=r@jr
        parent=body['parent']
        if parent>=0:
            p=positions[parent]+np.einsum('tij,tj->ti',rotations[parent],p)
            r=rotations[parent]@r
        positions.append(p);rotations.append(r)
    basis=np.asarray(descriptor['zup_to_yup']).reshape(3,3)
    delta=rot6d_matrix(root_rot6d)@basis@rotations[descriptor['root_frame']].transpose(0,2,1)
    root=np.asarray(root_t_cm)/100;origin=positions[descriptor['root_body']]
    visual_p=[];visual_r=[]
    for geom in descriptor['geoms']:
        b=geom['body'];gp=positions[b]+np.einsum('tij,j->ti',rotations[b],geom['position'])
        gr=rotations[b]@np.asarray(geom['rotation']).reshape(3,3)
        visual_p.append(root+np.einsum('tij,tj->ti',delta,gp-origin));visual_r.append(delta@gr)
    return np.stack(visual_p,axis=1),np.stack(visual_r,axis=1)
