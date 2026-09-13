from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

def load_bvh(path,position_channels='replace'):
    return load_bvh_text(Path(path).read_text(),position_channels)


def load_bvh_text(text,position_channels='replace',target_fps=None):
    """Parse an in-memory capture with the identical file adapter semantics."""
    lines=text.splitlines()
    names=[]; parents=[]; offsets=[]; channels=[]; stack=[]; current=None; end=False
    for k,line in enumerate(lines):
        s=line.strip().split()
        if not s: continue
        if s[0]=='MOTION': break
        if s[0] in ('ROOT','JOINT'):
            current=len(names); names.append(s[1]); parents.append(stack[-1] if stack else -1)
            offsets.append([0,0,0]); channels.append([]); end=False
        elif s[0]=='End': current=None; end=True
        elif s[0]=='{': stack.append(current)
        elif s[0]=='}': stack.pop(); current=stack[-1] if stack else None; end=current is None
        elif s[0]=='OFFSET' and current is not None: offsets[current]=list(map(float,s[1:]))
        elif s[0]=='CHANNELS' and current is not None: channels[current]=s[2:]
    n=int(lines[k+1].split(':')[1]); dt=float(lines[k+2].split(':')[1])
    values=np.loadtxt(lines[k+3:]).reshape(n,-1)
    if target_fps is not None:
        stride=round(1/dt/target_fps)
        if stride<1 or abs(1/dt/stride-target_fps)>.02:raise ValueError('Nonintegral source/target FPS ratio')
        values=values[::stride];n=len(values);dt*=stride
    J=len(names); local=np.broadcast_to(np.eye(3),(n,J,3,3)).copy()
    trans=np.broadcast_to(np.asarray(offsets),(n,J,3)).copy(); cursor=0
    for j,ch in enumerate(channels):
        rot=[(i,c[0]) for i,c in enumerate(ch) if c.endswith('rotation')]
        if rot: local[:,j]=Rotation.from_euler(''.join(c for i,c in rot),values[:,[cursor+i for i,c in rot]],degrees=True).as_matrix()
        for i,c in enumerate(ch):
            # SOMA's official reader replaces OFFSET with position channels.
            if c.endswith('position'):
                axis='XYZ'.index(c[0])
                trans[:,j,axis]=values[:,cursor+i]+(offsets[j][axis] if position_channels=='add' else 0)
        cursor+=len(ch)
    assert cursor==values.shape[1]
    gp=np.zeros((n,J,3)); gr=local.copy()
    for j,p in enumerate(parents):
        if p<0: gp[:,j]=trans[:,j]
        else:
            gr[:,j]=gr[:,p]@local[:,j]
            gp[:,j]=gp[:,p]+np.einsum('tij,tj->ti',gr[:,p],trans[:,j])
    return dict(names=names,parents=np.array(parents),offsets=np.array(offsets),local=local,global_rot=gr,positions=gp,dt=dt,channels=channels,values=values)

if __name__=='__main__':
    import sys
    b=load_bvh(sys.argv[1]); print('joints',len(b['names']),'frames',len(b['values']),'fps',1/b['dt'])
    for j,n in enumerate(b['names']):
        if any(x in n for x in ['Hips','Head','Shoulder','Arm','Hand','Foot','Toe']): print(n,np.round(b['positions'][0,j],2))
