import tempfile
import unittest
import numpy as np
from greenwich_motion_sdk.contact_labels import export_contacts

class ContactLabels(unittest.TestCase):
    def test_geometric_labels_do_not_copy_source_or_accept_penetration(self):
        motion=dict(sole_surface_height_cm=np.array([[0.,4.],[-2.,.2]]),
            source_contact_proxy=np.ones((2,2),bool),model_contact=np.zeros((2,2),bool),fps=30.)
        with tempfile.TemporaryDirectory() as out:
            labels=export_contacts(motion,out)
        np.testing.assert_array_equal(labels['geometry'],[[True,False],[False,True]])
        np.testing.assert_array_equal(labels['penetration'],[[False,False],[True,False]])
        self.assertTrue(labels['source'].all())
        self.assertFalse(labels['model'].any())

if __name__=='__main__':unittest.main()
