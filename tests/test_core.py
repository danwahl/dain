import numpy as np
import pytest

from dain import add, matmul, relu, relu_grad


def test_add_basic():
    """Test basic element-wise addition."""
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    b = np.array([4.0, 5.0, 6.0], dtype=np.float32)
    result = add(a, b)
    expected = np.array([5.0, 7.0, 9.0], dtype=np.float32)
    np.testing.assert_array_almost_equal(result, expected)


def test_add_2d():
    """Test 2D array addition."""
    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    b = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
    result = add(a, b)
    expected = np.array([[6.0, 8.0], [10.0, 12.0]], dtype=np.float32)
    np.testing.assert_array_almost_equal(result, expected)


def test_add_shape_mismatch():
    """Test error when array shapes don't match."""
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    b = np.array([4.0, 5.0], dtype=np.float32)
    with pytest.raises(ValueError, match="Arrays must have the same shape"):
        add(a, b)


def test_add_non_array():
    """Test error when inputs are not numpy arrays."""
    a = [1.0, 2.0, 3.0]
    b = np.array([4.0, 5.0, 6.0], dtype=np.float32)
    with pytest.raises(TypeError, match="Inputs must be numpy arrays"):
        add(a, b)


def test_add_non_contiguous():
    """Test with non-contiguous arrays."""
    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    b = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
    # Create non-contiguous array by transposing
    a_non_contig = a.T.copy().T
    assert not a_non_contig.flags["C_CONTIGUOUS"]
    result = add(a_non_contig, b)
    expected = np.array([[6.0, 8.0], [10.0, 12.0]], dtype=np.float32)
    np.testing.assert_array_almost_equal(result, expected)


def test_matmul_basic():
    """Test basic 2x2 matrix multiplication."""
    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    b = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
    result = matmul(a, b)
    expected = np.array([[19.0, 22.0], [43.0, 50.0]], dtype=np.float32)
    np.testing.assert_array_almost_equal(result, expected)


def test_matmul_non_square():
    """Test multiplication with non-square matrices."""
    a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    b = np.array([[7.0], [8.0], [9.0]], dtype=np.float32)
    result = matmul(a, b)
    expected = np.array([[50.0], [122.0]], dtype=np.float32)
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


def test_matmul_non_contiguous():
    """Test with non-contiguous arrays (should handle internally)."""
    a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    b = np.array([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]], dtype=np.float32)
    # Create non-contiguous array by transposing
    a_non_contig = a.T.copy().T
    assert not a_non_contig.flags["C_CONTIGUOUS"]
    result = matmul(a_non_contig, b)
    expected = np.array([[58.0, 64.0], [139.0, 154.0]], dtype=np.float32)
    np.testing.assert_array_almost_equal(result, expected)


def test_relu_basic():
    """Test basic ReLU functionality."""
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    result = relu(x)
    expected = np.array([0.0, 0.0, 0.0, 1.0, 2.0], dtype=np.float32)
    np.testing.assert_array_almost_equal(result, expected)


def test_relu_2d():
    """Test 2D array ReLU."""
    x = np.array([[-1.0, 2.0], [-3.0, 4.0]], dtype=np.float32)
    result = relu(x)
    expected = np.array([[0.0, 2.0], [0.0, 4.0]], dtype=np.float32)
    np.testing.assert_array_almost_equal(result, expected)


def test_relu_zeros():
    """Test ReLU with array of zeros."""
    x = np.zeros((3, 3), dtype=np.float32)
    result = relu(x)
    np.testing.assert_array_almost_equal(result, x)


def test_relu_non_array():
    """Test error when input is not a numpy array."""
    x = [-1.0, 0.0, 1.0]
    with pytest.raises(TypeError, match="Input must be a numpy array"):
        relu(x)


def test_relu_non_contiguous():
    """Test with non-contiguous array."""
    x = np.array([[-1.0, 2.0], [-3.0, 4.0]], dtype=np.float32)
    x_non_contig = x.T.copy().T
    assert not x_non_contig.flags["C_CONTIGUOUS"]
    result = relu(x_non_contig)
    expected = np.array([[0.0, 2.0], [0.0, 4.0]], dtype=np.float32)
    np.testing.assert_array_almost_equal(result, expected)


def test_relu_grad_basic():
    """Test basic ReLU gradient functionality."""
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    grad_in = np.ones_like(x)
    result = relu_grad(x, grad_in)
    expected = np.array([0.0, 0.0, 0.0, 1.0, 1.0], dtype=np.float32)
    np.testing.assert_array_almost_equal(result, expected)


def test_relu_grad_2d():
    """Test 2D array ReLU gradient."""
    x = np.array([[-1.0, 2.0], [-3.0, 4.0]], dtype=np.float32)
    grad_in = np.ones_like(x)
    result = relu_grad(x, grad_in)
    expected = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=np.float32)
    np.testing.assert_array_almost_equal(result, expected)


def test_relu_grad_with_upstream():
    """Test ReLU gradient with non-unit upstream gradient."""
    x = np.array([-1.0, 1.0, 2.0], dtype=np.float32)
    grad_in = np.array([0.5, 1.0, 2.0], dtype=np.float32)
    result = relu_grad(x, grad_in)
    expected = np.array([0.0, 1.0, 2.0], dtype=np.float32)
    np.testing.assert_array_almost_equal(result, expected)


def test_relu_grad_shape_mismatch():
    """Test error when array shapes don't match."""
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    grad_in = np.array([1.0, 2.0], dtype=np.float32)
    with pytest.raises(ValueError, match="Arrays must have the same shape"):
        relu_grad(x, grad_in)


def test_relu_grad_non_array():
    """Test error when inputs are not numpy arrays."""
    x = [1.0, 2.0, 3.0]
    grad_in = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    with pytest.raises(TypeError, match="Inputs must be numpy arrays"):
        relu_grad(x, grad_in)


def test_relu_grad_non_contiguous():
    """Test with non-contiguous arrays."""
    x = np.array([[-1.0, 2.0], [-3.0, 4.0]], dtype=np.float32)
    grad_in = np.ones_like(x)
    # Create non-contiguous array by transposing
    x_non_contig = x.T.copy().T
    grad_in_non_contig = grad_in.T.copy().T
    assert not x_non_contig.flags["C_CONTIGUOUS"]
    assert not grad_in_non_contig.flags["C_CONTIGUOUS"]
    result = relu_grad(x_non_contig, grad_in_non_contig)
    expected = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=np.float32)
    np.testing.assert_array_almost_equal(result, expected)
