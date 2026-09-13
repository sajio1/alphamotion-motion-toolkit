import unittest
import numpy as np
from greenwich_motion_sdk.locomotion_audit import source_support_phase


class SourcePhaseRegression(unittest.TestCase):
    def test_low_fast_airborne_foot_is_not_support(self):
        points=np.zeros((10,2,3));points[:,:,1]=4
        points[:,:,0]=np.arange(10)[:,None]
        phase,low,speed=source_support_phase(points,30)
        self.assertTrue(low.all())
        self.assertTrue((speed[1:]==30).all())
        self.assertFalse(phase[1:].any())

    def test_stationary_low_foot_remains_support(self):
        points=np.zeros((10,2,3));points[:,:,1]=4
        phase,_,_=source_support_phase(points,30)
        self.assertTrue(phase.all())

    def test_quiet_raised_foot_is_not_support(self):
        points=np.zeros((10,2,3));points[:,:,1]=12
        phase,_,_=source_support_phase(points,30)
        self.assertFalse(phase.any())


if __name__=='__main__':unittest.main()
