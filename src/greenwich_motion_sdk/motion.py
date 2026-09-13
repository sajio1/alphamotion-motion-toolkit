from dataclasses import dataclass, field
from pathlib import Path
import json
import numpy as np

@dataclass(frozen=True)
class CoordinateFrame:
    """Source-to-canonical rotation, source units per cm, and source floor height.

    Canonical coordinates are right-handed Y-up, centimetres. No body scaling.
    ground_y_cm is measured after basis/unit conversion, never inferred from a pose.
    """
    basis_to_yup: np.ndarray
    cm_per_unit: float
    ground_y_cm: float

    def validate(self):
        b=np.asarray(self.basis_to_yup)
        if b.shape!=(3,3) or not np.allclose(b.T@b,np.eye(3),atol=1e-6) or not np.isclose(np.linalg.det(b),1):
            raise ValueError('Coordinate basis must be a proper rotation; declare handedness conversion in the adapter')
        if not np.isfinite(self.cm_per_unit) or self.cm_per_unit<=0 or not np.isfinite(self.ground_y_cm):
            raise ValueError('Explicit finite units and ground required')

@dataclass
class MotionClip:
    """Full-skeleton clip in one canonical space; global rotations match rest offsets.

    For sparse EE inputs, use a dedicated future adapter with an observation mask;
    missing joints must not be silently filled with identity in this contract.
    """
    names: list[str]
    parents: np.ndarray
    rest_offsets_cm: np.ndarray
    global_rotation: np.ndarray
    world_position_cm: np.ndarray
    timestamps_s: np.ndarray
    metadata: dict = field(default_factory=dict)

    def validate(self, tolerance_cm=1e-3):
        T=len(self.timestamps_s);J=len(self.names)
        if T<1 or len(set(self.names))!=J: raise ValueError('Empty clip or duplicate joint names')
        for x,shape in [(self.parents,(J,)),(self.rest_offsets_cm,(J,3)),(self.global_rotation,(T,J,3,3)),(self.world_position_cm,(T,J,3)),(self.timestamps_s,(T,))]:
            if np.asarray(x).shape!=shape or not np.isfinite(x).all():raise ValueError(f'Invalid shape or values, expected {shape}')
        if self.parents[0]!=-1 or any(p<0 or p>=j for j,p in enumerate(self.parents[1:],1)):
            raise ValueError('Require one topologically ordered root at index zero')
        if np.any(np.diff(self.timestamps_s)<=0):raise ValueError('Timestamps must increase strictly')
        r=self.global_rotation
        if not np.allclose(r.swapaxes(-1,-2)@r,np.eye(3),atol=2e-5) or not np.allclose(np.linalg.det(r),1,atol=2e-5):
            raise ValueError('Invalid SO(3) rotations')
        for j,p in enumerate(self.parents):
            if p<0:continue
            reconstructed=self.world_position_cm[:,p]+np.einsum('tij,j->ti',r[:,p],self.rest_offsets_cm[j])
            if np.max(np.linalg.norm(reconstructed-self.world_position_cm[:,j],axis=-1))>tolerance_cm:
                raise ValueError(f'FK mismatch at {self.names[j]}: incompatible bind or animated non-root translations')
        return self

    def save(self,path):
        self.validate();path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(path,names=np.asarray(self.names,dtype=str),parents=self.parents,
            rest_offsets_cm=self.rest_offsets_cm,global_rotation=self.global_rotation,
            world_position_cm=self.world_position_cm,timestamps_s=self.timestamps_s,
            metadata=np.array(json.dumps(self.metadata)),schema=np.array('greenwich.motion.v1'))

    @classmethod
    def load(cls,path):
        with np.load(path,allow_pickle=False) as f:
            if str(f['schema'])!='greenwich.motion.v1':raise ValueError('Unsupported motion schema')
            return cls(f['names'].tolist(),f['parents'],f['rest_offsets_cm'],f['global_rotation'],
                       f['world_position_cm'],f['timestamps_s'],json.loads(str(f['metadata']))).validate()
