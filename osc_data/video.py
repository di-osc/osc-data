from __future__ import annotations

from typing import Union
from pathlib import Path

import numpy as np
from docarray import BaseDoc
from docarray.typing import VideoNdArray, VideoUrl, NdArray
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

    uri: Union[str, Path, VideoUrl] | None = Field(
        None, description="URL of the video file or local path."
    )
    data: Union[NdArray, VideoNdArray] = Field(None, description="Frames data (N, H, W, 3) uint8 RGB.")
    fps: float | None = Field(None, description="Frames per second")
    duration: float | None = Field(None, description="Duration in seconds")
    width: int | None = Field(None, description="Frame width")
    height: int | None = Field(None, description="Frame height")
    num_frames: int | None = Field(None, description="Number of frames")
    key_frames: list[int] | None = Field(None, description="Key frames")

    def load(self) -> "Video":
        """Load the video from local path or URL via Rust core."""
        try:
            if self.uri is None:
                raise ValueError("uri is not set")
            
            # Handle VideoUrl type
            if hasattr(self.uri, 'load'):
                result = self.uri.load()
                self.data = result.video
                self.key_frames = result.key_frame_indices
                # Extract other properties if available
                if hasattr(result, 'fps'):
                    self.fps = float(result.fps)
                if hasattr(result, 'duration'):
                    self.duration = float(result.duration)
                if hasattr(result, 'width'):
                    self.width = int(result.width)
                if hasattr(result, 'height'):
                    self.height = int(result.height)
                if hasattr(result, 'num_frames'):
                    self.num_frames = int(result.num_frames)
            else:
                # Handle string/Path type
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

    def split_by_key_frames(self) -> list[VideoNdArray]:
        """Split the video by key frames."""
        if self.key_frames is None:
            raise ValueError("Key frames are not set")
        if len(self.key_frames) == 0 or len(self.key_frames) == 1:
            return [self.data]
        videos = []
        start = 0
        for i in self.key_frames[1:]:
            videos.append(self.data[start:i])
            start = i
        videos.append(self.data[start:])
        return videos

    def display(self):
        """Display the video data."""
        if hasattr(self.data, 'display'):
            self.data.display()
        else:
            print(f"Video data shape: {self.data.shape if self.data is not None else 'None'}")

    @property
    def shape(self):
        """Get the shape of the video data."""
        return None if self.data is None else self.data.shape