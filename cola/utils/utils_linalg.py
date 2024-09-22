import numpy as np


def get_precision(xnp, dtype):
    if dtype == xnp.float32:
        return 1e-6
    elif dtype == xnp.float64:
        return 1e-15
    else:
        raise TypeError(f"Incorrect dtype {dtype}")


def get_numpy_dtype(dtype):
    if dtype in [np.float32, np.float64, np.complex64, np.complex128, np.int32, np.int64]:
        return dtype
    try:
        import torch
        match dtype:
            case torch.float32:
                return np.float32
            case torch.float64:
                return np.float64
            case torch.complex64:
                return np.complex64
            case torch.complex128:
                return np.complex128
    except ImportError:
        pass
    raise ImportError(f"Dtype: {dtype} is not valid")
