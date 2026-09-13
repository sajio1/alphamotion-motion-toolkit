import numpy as np

from greenwich_motion_sdk.surface_contacts import classify_surface_contacts


def test_classifies_any_surface_from_height():
    contact, penetration = classify_surface_contacts([[0.2, 4.0, -0.7]], 1.0, 0.5)
    np.testing.assert_array_equal(contact, [[True, False, False]])
    np.testing.assert_array_equal(penetration, [[False, False, True]])

