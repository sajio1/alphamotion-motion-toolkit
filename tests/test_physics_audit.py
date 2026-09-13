import unittest
import tempfile
from pathlib import Path
import numpy as np
from greenwich_motion_sdk.physics_audit import wrench_fit, intervals, evaluate, Thresholds


class PhysicsTests(unittest.TestCase):
    def setUp(self):
        self.points=np.array([[-.1,0,-.1],[-.1,0,.1],[.1,0,-.1],[.1,0,.1]])
    def test_static_supported(self):
        f,m,_=wrench_fit(self.points,[0,1,0],[0,981,0],[0,0,0],100,.7)
        self.assertLess(f,1e-8);self.assertLess(m,1e-8)
    def test_unsupported_static_rejected(self):
        f,_,_=wrench_fit([],np.array([0,1,0]),[0,981,0],[0,0,0],100,.7)
        self.assertAlmostEqual(f,1.)
    def test_ballistic_flight_allowed(self):
        f,m,_=wrench_fit([],np.array([0,1,0]),[0,0,0],[0,0,0],100,.7)
        self.assertEqual(f+m,0)
    def test_static_com_outside_support_rejected(self):
        f,m,_=wrench_fit(self.points,np.array([.7,1,0]),[0,981,0],[0,0,0],100,.7)
        self.assertGreater(f+m,.1)
    def test_friction_limit(self):
        f,m,_=wrench_fit(self.points,np.array([0,1,0]),[1962,981,0],[0,0,-1962],100,.3)
        self.assertGreater(f,.1)
    def test_event_timing(self):
        events=intervals([0,1,1,0,1],10,.15)
        self.assertEqual(len(events),1);self.assertAlmostEqual(events[0]['start_s'],.1);self.assertAlmostEqual(events[0]['end_s'],.3)
    def test_cache_threshold_re_evaluation(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);n=31;feet=np.broadcast_to(self.points,(n,4,3)).copy();feet[:,:,1]=-.006
            cache=dict(q=np.zeros((n,1,3)),fps=30.,link_mass_kg=np.array([100.]),link_inertia_kg_m2=np.ones((1,3)),
                link_com_m=np.broadcast_to([0,1,0],(n,1,3)),link_inertial_rotation=np.broadcast_to(np.eye(3),(n,1,3,3)),
                foot_0_vertices_m=feet,foot_1_vertices_m=feet,native_fk_error_m=np.zeros(n),serialized_fk_error_m=np.zeros(n),
                native_equality_error=np.zeros(n),unmapped_native_joints=np.array([],dtype=str),joint_limit_excess_deg=np.zeros(n))
            np.savez(p/'cache.npz',**cache)
            a=evaluate(p/'cache.npz',p/'strict');b=evaluate(p/'cache.npz',p/'loose',Thresholds(penetration_m=.01))
            self.assertEqual(a['status'],'rejected');self.assertEqual(b['status'],'passes_necessary_screen')
            self.assertFalse(b['dynamically_certified'])


if __name__=='__main__':unittest.main()
