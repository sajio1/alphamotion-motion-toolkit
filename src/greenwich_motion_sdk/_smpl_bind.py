"""Deterministic SOMA BVH -> SMPL22 pose adapter, with bind-frame alignment.

Consumes the entire SOMA hierarchy to compute world transforms, then exports
SMPL22 body channels. No training, target robot, temporal processing or IK.
The static calibration is derived from the separately supplied bind BVH only.
"""
import numpy as np
from scipy.spatial.transform import Rotation

SMPL_TO_SOMA = ['Hips','LeftLeg','RightLeg','Spine1','LeftShin','RightShin',
 'Spine2','LeftFoot','RightFoot','Chest','LeftToeBase','RightToeBase','Neck1',
 'LeftShoulder','RightShoulder','Head','LeftArm','RightArm','LeftForeArm',
 'RightForeArm','LeftHand','RightHand']

class SomaSmplRestAdapter:
 def __init__(self,bind,smpl_spec):
  self.spec=smpl_spec;self.source_names=list(bind['names'])
  self.ids=np.array([self.source_names.index(n) for n in SMPL_TO_SOMA])
  self.bind_R=bind['global_rot'][0,self.ids].copy()
  self.bind_P=bind['positions'][0,self.ids].copy()
  self.alignment=np.tile(np.eye(3),(smpl_spec.J,1,1))
  for j in range(smpl_spec.J):
   children=np.where(smpl_spec.parents==j)[0]
   if not len(children):continue
   target=np.asarray(smpl_spec.rest_offsets[children],float)
   source=self.bind_P[children]-self.bind_P[j]
   target/=np.linalg.norm(target,axis=-1,keepdims=True)
   source/=np.linalg.norm(source,axis=-1,keepdims=True)
   if len(children)==1:
    quat=np.r_[np.cross(target[0],source[0]),1+np.dot(target[0],source[0])]
    if np.linalg.norm(quat)<1e-8:
     # Deterministic perpendicular axis for antiparallel reference bones.
     basis=np.eye(3)[np.argmin(np.abs(target[0]))];axis=np.cross(target[0],basis);axis/=np.linalg.norm(axis)
     quat=np.r_[axis,0.]
    self.alignment[j]=Rotation.from_quat(quat).as_matrix()
   else:
    U,_,Vh=np.linalg.svd(source.T@target)
    self.alignment[j]=U@np.diag([1,1,np.linalg.det(U@Vh)])@Vh

 def convert(self,motion,mode='local_basis'):
  if list(motion['names'])!=self.source_names:raise ValueError('SOMA hierarchy does not match supplied bind skeleton')
  delta=motion['global_rot'][:,self.ids]@self.bind_R.transpose(0,2,1)
  world=delta@self.alignment if mode=='pose_bias' else delta
  local=world.copy()
  for j,p in enumerate(self.spec.parents):
   if p>=0:local[:,j]=world[:,p].transpose(0,2,1)@world[:,j]
  if mode=='local_basis':
   # A change of joint coordinate basis is conjugation, not an added pose.
   # In particular, bind maps to identity SMPL local rotations.
   local=self.alignment.transpose(0,2,1)@local@self.alignment
   # Preserve measured world heading at the root.
   local[:,0]=delta[:,0]
   world=local.copy()
   for j,p in enumerate(self.spec.parents):
    if p>=0:world[:,j]=world[:,p]@local[:,j]
  axis_angle=Rotation.from_matrix(local.reshape(-1,3,3)).as_rotvec().reshape(len(world),self.spec.J,3)
  return dict(local_rotation=local,global_rotation=world,global_orient=axis_angle[:,0],body_pose=axis_angle[:,1:],root_translation_cm=motion['positions'][:,self.ids[0]],dt=motion['dt'])
