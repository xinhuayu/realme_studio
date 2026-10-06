import numpy as np


def pad_center(data, size, axis=-1, **kwargs):
    """Centre `data` in a buffer of length `size`. Small enough to be exact."""
    n = data.shape[axis]
    if size < n:
        raise ValueError(f"Target size ({size}) must be at least {n}")
    lpad = (size - n) // 2
    lengths = [(0, 0)] * data.ndim
    lengths[axis] = (lpad, size - n - lpad)
    return np.pad(np.asarray(data), lengths, mode=kwargs.get("mode", "constant"))
