"""Export separate source, model and final-geometry foot contact labels."""
import json
from pathlib import Path
import numpy as np

def export_contacts(motion, output, contact_band_cm=1., penetration_cm=.5):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    height=np.asarray(motion['sole_surface_height_cm'])
    source=np.asarray(motion['source_contact_proxy'],dtype=bool)
    predicted=np.asarray(motion['model_contact'],dtype=bool)
    if height.ndim!=2 or height.shape[1]!=2 or source.shape!=height.shape or predicted.shape!=height.shape:
        raise ValueError('Expected synchronized [T,2] left/right foot arrays')
    fps=float(motion['fps'])
    if fps<=0 or not np.isfinite(height).all():raise ValueError('Invalid FPS or sole heights')
    penetration=height < -penetration_cm
    geometric=(height<=contact_band_cm)&~penetration
    time=np.arange(len(height))/fps
    np.savez_compressed(output/'contact_labels.npz',time_s=time,source_proxy=source,
        model_prediction=predicted,geometry_near_floor=geometric,penetration=penetration,sole_height_cm=height)
    columns=np.column_stack([np.arange(len(height)),time,source,predicted,geometric,penetration,height])
    np.savetxt(output/'contact_labels.csv',columns,delimiter=',',comments='',
        header='frame,time_s,source_L,source_R,model_L,model_R,geometry_L,geometry_R,penetration_L,penetration_R,sole_L_cm,sole_R_cm')
    (output/'contact_labels.json').write_text(json.dumps(dict(schema='foot-contact-labels-v1',order=['left','right'],
        frames=len(height),fps=fps,contact_band_cm=contact_band_cm,penetration_cm=penetration_cm,
        source='SOMA ankle/toe height and velocity proxy; not force ground truth',
        model='Existing decoder contact classification saved by generation',
        geometry='Final lowest sole vertex within ground band, excluding excessive penetration; not force contact or no-slip certification'),indent=2))
    return dict(source=source,model=predicted,geometry=geometric,penetration=penetration)
