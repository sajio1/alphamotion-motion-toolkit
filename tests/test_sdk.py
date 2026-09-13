import tempfile,unittest
from pathlib import Path
import numpy as np
from greenwich_motion_sdk import CoordinateFrame,MotionClip,load_motion,Pipeline,RunRequest

class SDKTests(unittest.TestCase):
    def test_smpl_units_pelvis_origin_and_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'smpl.npz'
            np.savez(path,poses=np.zeros((2,6)),trans=np.array([[1.,2.,3.],[2.,2.,3.]]),mocap_framerate=30.)
            clip=load_motion(path,format='smpl',names=['Pelvis','Head'],parents=[-1,0],
                rest_offsets=[[0,0,0],[0,0,0.5]],pelvis_offset_source_units=[0,0,0.1],
                frame=CoordinateFrame(np.array([[1,0,0],[0,0,1],[0,-1,0]]),100.,10.))
            np.testing.assert_allclose(clip.world_position_cm[0,0],[100,300,-200])
            np.testing.assert_allclose(clip.world_position_cm[0,1],[100,350,-200])
            clip.save(Path(tmp)/'canonical.npz')
            restored=MotionClip.load(Path(tmp)/'canonical.npz')
            np.testing.assert_allclose(restored.world_position_cm,clip.world_position_cm)
            np.testing.assert_allclose(restored.timestamps_s,[0,1/30])

    def test_fk_and_reflection_rejected(self):
        with self.assertRaises(ValueError):CoordinateFrame(np.diag([-1,1,1]),1.,0.).validate()
        c=MotionClip(['root','child'],np.array([-1,0]),np.array([[0,0,0],[0,1,0]]),
            np.tile(np.eye(3),(2,2,1,1)),np.zeros((2,2,3)),np.array([0.,1.]))
        with self.assertRaisesRegex(ValueError,'FK mismatch'):c.validate()

    def test_request_preserves_paths_and_rejects_invalid_modes(self):
        p=Pipeline(Path('toolkit with spaces'))
        req=RunRequest((3,),Path('output with spaces'),Path('robot.json'))
        cmd=p.command(req)
        self.assertEqual(cmd[cmd.index('-Indices')+1],'3')
        self.assertIn('output with spaces',cmd[cmd.index('-Output')+1])
        with self.assertRaises(ValueError):p.command(RunRequest((3,),Path('o'),Path('r'),representation='guess'))

if __name__=='__main__':unittest.main()
