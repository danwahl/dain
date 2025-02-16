import ctypes
import pathlib

import numpy as np

lib_path = pathlib.Path(__file__).parent / "kernels.so"
_lib = ctypes.CDLL(str(lib_path))

_lib.dain_add.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float32),
    np.ctypeslib.ndpointer(dtype=np.float32),
    np.ctypeslib.ndpointer(dtype=np.float32),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
]
_lib.dain_add.restype = ctypes.c_int

_lib.dain_matmul.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float32),
    np.ctypeslib.ndpointer(dtype=np.float32),
    np.ctypeslib.ndpointer(dtype=np.float32),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
]
_lib.dain_matmul.restype = ctypes.c_int

_lib.dain_mse.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float32),
    np.ctypeslib.ndpointer(dtype=np.float32),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int,
]
_lib.dain_mse.restype = ctypes.c_int

_lib.dain_mse_grad.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float32),
    np.ctypeslib.ndpointer(dtype=np.float32),
    np.ctypeslib.ndpointer(dtype=np.float32),
    ctypes.c_int,
]
_lib.dain_mse_grad.restype = ctypes.c_int

_lib.dain_relu.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float32),
    np.ctypeslib.ndpointer(dtype=np.float32),
    ctypes.c_int,
]
_lib.dain_relu.restype = ctypes.c_int

_lib.dain_relu_grad.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float32),
    np.ctypeslib.ndpointer(dtype=np.float32),
    np.ctypeslib.ndpointer(dtype=np.float32),
    ctypes.c_int,
]
_lib.dain_relu_grad.restype = ctypes.c_int


def add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Element-wise addition with broadcasting support.

    Args:
        a: First input array
        b: Second input array

    Returns:
        Array containing element-wise sum with broadcasting
    """
    if not (isinstance(a, np.ndarray) and isinstance(b, np.ndarray)):
        raise TypeError("Inputs must be numpy arrays")

    if not (a.ndim == 2 and b.ndim == 2):
        raise ValueError("Inputs must be 2D arrays")

    a_rows, a_cols = a.shape
    b_rows, b_cols = b.shape

    if not ((b_rows == 1 or b_rows == a_rows) and (b_cols == 1 or b_cols == a_cols)):
        raise ValueError(f"Cannot broadcast shapes {a.shape} and {b.shape}")

    if not a.flags["C_CONTIGUOUS"]:
        a = np.ascontiguousarray(a)
    if not b.flags["C_CONTIGUOUS"]:
        b = np.ascontiguousarray(b)

    a = a.astype(np.float32)
    b = b.astype(np.float32)
    c = np.empty_like(a)

    if _lib.dain_add(a, b, c, a_rows, a_cols, b_rows, b_cols) != 0:
        raise RuntimeError("Operation failed")

    return c


def matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Matrix multiplication of two 2D arrays.

    Args:
        a: First input array of shape (M, K)
        b: Second input array of shape (K, N)

    Returns:
        Result array of shape (M, N)
    """
    if not (isinstance(a, np.ndarray) and isinstance(b, np.ndarray)):
        raise TypeError("Inputs must be numpy arrays")

    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("Inputs must be 2D arrays")

    m, k = a.shape
    k2, n = b.shape

    if k != k2:
        raise ValueError(f"Inner dimensions must match: {k} != {k2}")

    if not a.flags["C_CONTIGUOUS"]:
        a = np.ascontiguousarray(a)
    if not b.flags["C_CONTIGUOUS"]:
        b = np.ascontiguousarray(b)

    a = a.astype(np.float32)
    b = b.astype(np.float32)

    c = np.empty((m, n), dtype=np.float32)

    if _lib.dain_matmul(a, b, c, m, n, k) != 0:
        raise RuntimeError("Operation failed")

    return c


def mse(pred: np.ndarray, target: np.ndarray) -> float:
    """Compute Mean Squared Error loss.

    Args:
        pred: Predicted values
        target: Target values

    Returns:
        MSE loss value
    """
    if not (isinstance(pred, np.ndarray) and isinstance(target, np.ndarray)):
        raise TypeError("Inputs must be numpy arrays")

    if pred.shape != target.shape:
        raise ValueError("Arrays must have the same shape")

    if not pred.flags["C_CONTIGUOUS"]:
        pred = np.ascontiguousarray(pred)
    if not target.flags["C_CONTIGUOUS"]:
        target = np.ascontiguousarray(target)

    pred = pred.astype(np.float32)
    target = target.astype(np.float32)

    loss = ctypes.c_float()

    if _lib.dain_mse(pred, target, ctypes.byref(loss), pred.size) != 0:
        raise RuntimeError("Operation failed")

    return loss.value


def mse_grad(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Compute gradient of MSE loss with respect to predictions.

    Args:
        pred: Predicted values
        target: Target values

    Returns:
        Gradient of MSE with respect to predictions
    """
    if not (isinstance(pred, np.ndarray) and isinstance(target, np.ndarray)):
        raise TypeError("Inputs must be numpy arrays")

    if pred.shape != target.shape:
        raise ValueError("Arrays must have the same shape")

    if not pred.flags["C_CONTIGUOUS"]:
        pred = np.ascontiguousarray(pred)
    if not target.flags["C_CONTIGUOUS"]:
        target = np.ascontiguousarray(target)

    pred = pred.astype(np.float32)
    target = target.astype(np.float32)
    grad = np.empty_like(pred)

    if _lib.dain_mse_grad(pred, target, grad, pred.size) != 0:
        raise RuntimeError("Operation failed")

    return grad


def relu(x: np.ndarray) -> np.ndarray:
    """Apply ReLU activation function element-wise.

    Args:
        x: Input array

    Returns:
        Array with ReLU activation applied (max(0, x))
    """
    if not isinstance(x, np.ndarray):
        raise TypeError("Input must be a numpy array")

    if not x.flags["C_CONTIGUOUS"]:
        x = np.ascontiguousarray(x)

    x = x.astype(np.float32)
    y = np.empty_like(x)

    if _lib.dain_relu(x, y, x.size) != 0:
        raise RuntimeError("Operation failed")

    return y


def relu_grad(x: np.ndarray, grad_in: np.ndarray) -> np.ndarray:
    """Compute ReLU gradient.

    Args:
        x: Input array that was passed to ReLU
        grad_in: Incoming gradient from upstream

    Returns:
        Gradient with respect to x
    """
    if not (isinstance(x, np.ndarray) and isinstance(grad_in, np.ndarray)):
        raise TypeError("Inputs must be numpy arrays")

    if x.shape != grad_in.shape:
        raise ValueError("Arrays must have the same shape")

    if not x.flags["C_CONTIGUOUS"]:
        x = np.ascontiguousarray(x)
    if not grad_in.flags["C_CONTIGUOUS"]:
        grad_in = np.ascontiguousarray(grad_in)

    x = x.astype(np.float32)
    grad_in = grad_in.astype(np.float32)
    grad_out = np.empty_like(x)

    if _lib.dain_relu_grad(x, grad_in, grad_out, x.size) != 0:
        raise RuntimeError("Operation failed")

    return grad_out
