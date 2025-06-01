import numpy as np

from ._core import audio


def low_frame_rate(frames: np.ndarray, m: int, n: int) -> np.ndarray:
    """apply low frame rate
    Args:
        frames: input frames. shape: (batch_size, n_frames, n_features)
        m: number of frames to combine
        n: number of frames to skip
    """

    return audio.low_frame_rate(frames, m, n)


def compute_decibel(frames: np.ndarray) -> np.ndarray:
    """compute decibel
    Args:
        frames: input frames. shape: (n_frames, n_features)
    """

    return audio.compute_decibel(frames)
