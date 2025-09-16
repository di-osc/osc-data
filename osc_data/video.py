from __future__ import annotations

import av
import numpy as np
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

    uri: VideoUrl | None = Field(
        None, description="URL of the video file or local path."
    )
    data: VideoNdArray = Field(None, description="Frames data (N, H, W, 3) uint8 RGB.")
    key_frames: list[int] | None = Field(None, description="Key frames")
    fps: int | None = Field(None, description="Frames per second")
    duration: float | None = Field(None, description="Duration in seconds")

    def load(self) -> "Video":
        """Load the video from local path or URL using av library."""
        if self.uri is None:
            raise ValueError("Video URI is not set")

        try:
            with av.open(str(self.uri)) as container:
                video_stream = container.streams.video[0]

                # 获取视频基本信息
                self.fps = round(video_stream.average_rate)
                self.duration = round(float(video_stream.duration * video_stream.time_base), 2)

                # 提取所有帧和关键帧
                frames = []
                key_frame_indices = []

                for i, frame in enumerate(container.decode(video=0)):
                    # 转换为RGB格式的numpy数组
                    frame_rgb = frame.to_ndarray(format="rgb24")
                    frames.append(frame_rgb)

                    # 检查是否为关键帧
                    if frame.key_frame:
                        key_frame_indices.append(i)

                # 将帧数据组合成VideoNdArray格式 (N, H, W, 3)
                if frames:
                    self.data = np.array(frames, dtype=np.uint8)
                    self.key_frames = key_frame_indices
                else:
                    self.data = None
                    self.key_frames = []

        except Exception as e:
            # 如果加载失败，重置所有属性
            self.data = None
            self.key_frames = []
            self.fps = None
            self.duration = None
            raise RuntimeError(f"Failed to load video: {e}")

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
        if hasattr(self.data, "display"):
            self.data.display()
        else:
            print(
                f"Video data shape: {self.data.shape if self.data is not None else 'None'}"
            )
