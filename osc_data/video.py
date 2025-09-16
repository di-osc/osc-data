from __future__ import annotations


from docarray import BaseDoc
from docarray.typing import VideoNdArray, VideoUrl
from pydantic import Field, ConfigDict



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

    uri: VideoUrl = Field(None, description="URL of the video file.")
    data: VideoNdArray = Field(None, description="Frames data (N, H, W, 3) uint8 RGB.")
    fps: float | None = Field(None, description="Frames per second")
    duration: float | None = Field(None, description="Duration in seconds")
    width: int | None = Field(None, description="Frame width")
    height: int | None = Field(None, description="Frame height")
    key_frames: list[int] | None = Field(None, description="Key frames")

    def load(self) -> "Video":
        """Load the video from local path or URL via Rust core."""
        result = self.uri.load()
        self.data = result.video
        self.key_frames = result.key_frame_indices
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
        self.data.display()
