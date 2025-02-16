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


def add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Element-wise addition of two arrays.

    Args:
        a: First input array
        b: Second input array

    Returns:
        Array containing element-wise sum of inputs
    """
    if not (isinstance(a, np.ndarray) and isinstance(b, np.ndarray)):
        raise TypeError("Inputs must be numpy arrays")

    if a.shape != b.shape:
        raise ValueError("Arrays must have the same shape")

    # Ensure contiguous arrays
    if not a.flags["C_CONTIGUOUS"]:
        a = np.ascontiguousarray(a)
    if not b.flags["C_CONTIGUOUS"]:
        b = np.ascontiguousarray(b)

    a = a.astype(np.float32)
    b = b.astype(np.float32)

    c = np.empty_like(a)

    if _lib.dain_add(a, b, c, a.size) != 0:
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
