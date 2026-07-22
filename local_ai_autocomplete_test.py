import numpy as np


def normalize_skeleton(tensor: np.ndarray) -> np.ndarray:
    """
    Normalize a skeleton tensor by centering it around its origin and scaling it to have a maximum absolute value of 1.

    Args:
        tensor (np.ndarray): A 3D numpy array representing the skeleton, with shape (num_joints, num_frames, num_dims).

    Returns:
        np.ndarray: The normalized skeleton tensor.
    """
    if not isinstance(tensor, np.ndarray):
        raise ValueError("Input must be a numpy array")

    center = tensor[:, 0:1, :]
    tensor = tensor - center

    scale = np.max(np.abs(tensor))
    if scale == 0:
        raise ValueError("Cannot normalize skeleton with zero scale")

    return tensor / scale