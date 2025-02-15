import ctypes
import numpy as np
import pathlib

# Load the CUDA library
lib_path = pathlib.Path(__file__).parent / "kernels.so"
_lib = ctypes.CDLL(str(lib_path))

# Set up the function signature
_lib.add_one_kernel_wrapper.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float32),
    ctypes.c_int
]
_lib.add_one_kernel_wrapper.restype = None

def add_one(arr):
    """Add 1 to each element of the array using CUDA."""
    if not arr.flags['C_CONTIGUOUS']:
        arr = np.ascontiguousarray(arr)
    arr = arr.astype(np.float32)
    _lib.add_one_kernel_wrapper(arr, arr.size)
    return arr
