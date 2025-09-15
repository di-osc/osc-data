from __future__ import annotations

from typing import Union
from pathlib import Path

import numpy as np
from docarray import BaseDoc
from docarray.typing import NdArray
from pydantic import Field, ConfigDict

from . import _core


class Video(BaseDoc):
    """Video object that represents a video file.

    Args:
        uri (Union[str, Path]): URL of the video file or local path.
        data (NdArray): Frames data (N, H, W, 3) uint8 RGB.
        fps (float): Frames per second.
        duration (float): Duration in seconds.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True, validate_assignment=True, extra="ignore"
    )

    uri: Union[str, Path] | None = Field(
        None, description="URL of the video file or local path."
    )
    data: NdArray = Field(None, description="Frames data (N, H, W, 3) uint8 RGB.")
    fps: float | None = Field(None, description="Frames per second")
    duration: float | None = Field(None, description="Duration in seconds")
    width: int | None = Field(None, description="Frame width")
    height: int | None = Field(None, description="Frame height")
    num_frames: int | None = Field(None, description="Number of frames")

    def load(self) -> "Video":
        """Load the video from local path or URL via Rust core."""
        try:
            if self.uri is None:
                raise ValueError("uri is not set")
            u = str(self.uri)
            if Path(u).exists():
                frames, fps, duration, w, h, n = _core.video.load_from_path(u)
            else:
                # ffmpeg can read URLs directly; keep consistent API
                frames, fps, duration, w, h, n = _core.video.load_from_url(u)
            self.data = np.asarray(frames, dtype=np.uint8)
            self.fps = float(fps)
            self.duration = float(duration)
            self.width = int(w)
            self.height = int(h)
            self.num_frames = int(n)
        except Exception as e:
            raise ValueError(f"Failed to load video from {self.uri}. {e}")
        return self

    @property
    def shape(self):
        return None if self.data is None else self.data.shape

    def keyframes(self):
        """Return keyframe attributes as a list of tuples.

        Each tuple is: (approx_index, time_sec, pict_type, pkt_size)
        """
        if self.uri is None:
            raise ValueError("uri is not set")
        u = str(self.uri)
        if Path(u).exists():
            items = _core.video.keyframes_from_path(u)
        else:
            items = _core.video.keyframes_from_url(u)
        return list(items)
