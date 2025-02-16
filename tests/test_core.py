import numpy as np
import pytest

from dain import matmul


def test_matmul_basic():
    """Test basic 2x2 matrix multiplication."""
    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    b = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
    result = matmul(a, b)
    expected = np.array([[19.0, 22.0], [43.0, 50.0]], dtype=np.float32)
    np.testing.assert_array_almost_equal(result, expected)


def test_matmul_non_square():
    """Test multiplication with non-square matrices."""
    a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)  # 2x3
    b = np.array([[7.0], [8.0], [9.0]], dtype=np.float32)  # 3x1
    result = matmul(a, b)
    expected = np.array([[50.0], [122.0]], dtype=np.float32)  # 2x1
    np.testing.assert_array_almost_equal(result, expected)


def test_matmul_non_array():
    """Test error when inputs are not numpy arrays."""
    a = [[1.0, 2.0], [3.0, 4.0]]
    b = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
    with pytest.raises(TypeError, match="Inputs must be numpy arrays"):
        matmul(a, b)


def test_matmul_non_2d():
    """Test error when inputs are not 2D arrays."""
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    b = np.array([[4.0], [5.0], [6.0]], dtype=np.float32)
    with pytest.raises(ValueError, match="Inputs must be 2D arrays"):
        matmul(a, b)


def test_matmul_incompatible_shapes():
    """Test error when matrix shapes are incompatible."""
    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    b = np.array(
        [[5.0, 6.0, 7.0], [8.0, 9.0, 10.0], [11.0, 12.0, 13.0]], dtype=np.float32
    )
    with pytest.raises(ValueError, match="Inner dimensions must match"):
        matmul(a, b)
