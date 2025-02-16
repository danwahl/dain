import ctypes
import pathlib

import numpy as np

lib_path = pathlib.Path(__file__).parent / "kernels.so"
_lib = ctypes.CDLL(str(lib_path))

_lib.add_one_kernel_wrapper.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float32),
    ctypes.c_int,
]
_lib.add_one_kernel_wrapper.restype = ctypes.c_int


def add_one(arr):
    if not isinstance(arr, np.ndarray):
        raise TypeError("Input must be a numpy array")

    if not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr)
    arr = arr.astype(np.float32)

    if _lib.add_one_kernel_wrapper(arr, arr.size) != 0:
        raise RuntimeError("CUDA error")
    return arr
