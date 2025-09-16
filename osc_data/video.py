from __future__ import annotations
from pathlib import Path
import requests
from io import BytesIO

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

        # load local video
        if Path(self.uri).exists():
            try:
                with av.open(str(self.uri)) as container:
                    video_stream = container.streams.video[0]

                    # 获取视频基本信息
                    self.fps = round(video_stream.average_rate)
                    self.duration = round(
                        float(video_stream.duration * video_stream.time_base), 2
                    )

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
                raise RuntimeError(f"Failed to load video: {e}") from e
            return self
        # load remote video
        else:
            try:
                response = requests.get(self.uri)
                with av.open(BytesIO(response.content)) as container:
                    video_stream = container.streams.video[0]
                    self.fps = round(video_stream.average_rate)
                    self.duration = round(
                        float(video_stream.duration * video_stream.time_base), 2
                    )
                    frames = []
                    key_frame_indices = []
                    for i, frame in enumerate(container.decode(video=0)):
                        frame_rgb = frame.to_ndarray(format="rgb24")
                        frames.append(frame_rgb)
                        if frame.key_frame:
                            key_frame_indices.append(i)
                    if frames:
                        self.data = np.array(frames, dtype=np.uint8)
                        self.key_frames = key_frame_indices
                    return self
            except Exception as e:
                raise RuntimeError(f"Failed to load video: {e}") from e

    def split_by_key_frames(self, min_split_duration_s: int = 5) -> list[VideoNdArray]:
        """Split the video by key frames."""
        if self.key_frames is None:
            raise ValueError("Key frames are not set")
        if len(self.key_frames) == 0 or len(self.key_frames) == 1:
            return [self.data]
        videos = []
        start = 0
        min_split_frames = min_split_duration_s * self.fps
        for key_frame_idx in self.key_frames[1:]:
            if key_frame_idx - start < min_split_frames:
                continue
            videos.append(self.data[start:key_frame_idx])
            start = key_frame_idx
        videos.append(self.data[start:])
        return videos

    def save(self, path: str, format: str = "mp4", codec: str = "h264"):
        """Save the video to local path.
        format: https://docs.opencv.org/4.10.0/dd/d9e/group__videoio__flags__base.html#gaeb8dd9c89c10a5c63c139bf7c4f5704d
        Args:
            path (str): Local path to save the video.
            format (str): Format to save the video. https://docs.opencv.org/4.10.0/dd/d9e/group__videoio__flags__base.html#gaeb8dd9c89c10a5c63c139bf7c4f5704d
            codec (str): Codec to save the video.
        """
        if Path(path).exists():
            raise FileExistsError(f"File {path} already exists")
        if self.data is None:
            raise ValueError("Video data is not set")
        if self.fps is None:
            raise ValueError("Video fps is not set")

        try:
            with av.open(path, mode="w", format=format) as container:
                # 添加视频流，使用分数类型的fps
                from fractions import Fraction

                fps_fraction = Fraction(self.fps, 1)
                stream = container.add_stream(codec, rate=fps_fraction)

                # 设置视频流的其他参数
                stream.width = self.data.shape[2]  # 宽度
                stream.height = self.data.shape[1]  # 高度
                stream.pix_fmt = "yuv420p"  # 像素格式

                for frame_data in self.data:
                    # 创建VideoFrame对象
                    frame = av.VideoFrame.from_ndarray(frame_data, format="rgb24")
                    frame = frame.reformat(format="yuv420p")

                    # 编码帧
                    for packet in stream.encode(frame):
                        container.mux(packet)

                # 完成编码
                for packet in stream.encode():
                    container.mux(packet)

        except Exception as e:
            raise RuntimeError(f"Failed to save video: {e}") from e

        return self

    def display(self):
        """Display the video data."""
        if hasattr(self.data, "display"):
            self.data.display()
        else:
            print(
                f"Video data shape: {self.data.shape if self.data is not None else 'None'}"
            )
