# from dataclasses import dataclass
# from typing import Optional
# import numpy as np

# @dataclass
# class Audio:
#     path: str
#     id: Optional[str]
#     duration: Optional[float] = None
#     mono: Optional[bool] = None
#     sample_rate: Optional[int] = None
#     waveform: Optional[bytes] = None

#     def load_url(self) -> None: ...
#     def load_local(self) -> None: ...

# def low_frame_rate(frames: np.ndarray, m: int, n: int) -> np.ndarray:
#     """apply low frame rate
#     Args:
#         frames: input frames. shape: (batch_size, n_frames, n_features)
#         m: number of frames to combine
#         n: number of frames to skip
#     """
#     pass

# def compute_decibel(frames: np.ndarray) -> np.ndarray:
#     """compute decibel
#     Args:
#         frames: input frames. shape: (n_frames, n_features)
#     """
#     pass
