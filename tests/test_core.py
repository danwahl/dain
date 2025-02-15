import numpy as np
from dain.core import add_one

def test_add_one():
    arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    result = add_one(arr)
    expected = np.array([2.0, 3.0, 4.0], dtype=np.float32)
    np.testing.assert_array_almost_equal(result, expected)
