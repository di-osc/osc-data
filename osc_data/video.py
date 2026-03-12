from __future__ import annotations
from pathlib import Path
from io import BytesIO
from fractions import Fraction
import tempfile

import av
import numpy as np
from docarray import BaseDoc
from docarray.typing import VideoNdArray, VideoUrl
from pydantic import Field, ConfigDict
import requests
from wasabi import Printer

from osc_data.audio import Audio

# Initialize wasabi printer
msg = Printer()


class Video(BaseDoc):
    """Video object that represents a video file.

    Args:
        uri (Union[str, Path]): URL of the video file or local path.
        data (NdArray): Frames data (N, H, W, 3) uint8 RGB.
        fps (float): Frames per second.
        duration (float): Duration in seconds.
        has_audio (bool): Whether the video has audio track.
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
    has_audio: bool = Field(False, description="Whether the video has audio track.")

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
                msg.fail(f"Failed to load video: {e}")
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

            except Exception as e:
                msg.fail(f"Failed to load video from URL: {e}")
                raise RuntimeError(f"Failed to load video: {e}") from e

    def load_example(self) -> "Video":
        """Load the example video file.

        Returns:
            Video: The loaded example video.
        """
        example_path = Path(__file__).parent / "assets" / "video" / "example.mp4"
        self.uri = str(example_path)
        return self.load()

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
            msg.fail(f"Failed to save video: {e}")
            raise RuntimeError(f"Failed to save video: {e}") from e

        return self

    def crop(self, x: int, y: int, width: int, height: int) -> "Video":
        """Crop video to specified region.

        Args:
            x (int): X coordinate of top-left corner.
            y (int): Y coordinate of top-left corner.
            width (int): Crop width.
            height (int): Crop height.

        Returns:
            Video: New cropped Video object.

        Raises:
            ValueError: If video data is not set or crop region is invalid.
        """
        if self.data is None:
            raise ValueError("Video data is not set")

        video_height, video_width = self.data.shape[1:3]

        if x < 0 or y < 0 or width <= 0 or height <= 0:
            raise ValueError("Crop dimensions must be positive")
        if x + width > video_width or y + height > video_height:
            raise ValueError(f"Crop region exceeds video bounds ({video_width}x{video_height})")

        # Crop all frames using numpy slicing (N, H, W, C)
        cropped_data = self.data[:, y:y+height, x:x+width, :]

        return Video(
            uri=self.uri,
            data=cropped_data,
            fps=self.fps,
            duration=self.duration,
            has_audio=self.has_audio,
        )

    def resize(self, width: int, height: int) -> "Video":
        """Resize video to target dimensions.

        Uses bilinear interpolation for each frame.

        Args:
            width (int): Target width in pixels.
            height (int): Target height in pixels.

        Returns:
            Video: New resized Video object.

        Raises:
            ValueError: If video data is not set.
        """
        if self.data is None:
            raise ValueError("Video data is not set")

        from PIL import Image as PILImage

        resized_frames = []
        for frame in self.data:
            # Convert numpy array to PIL Image
            pil_img = PILImage.fromarray(frame)
            # Resize using bilinear interpolation
            resized = pil_img.resize((width, height), PILImage.Resampling.BILINEAR)
            # Convert back to numpy array
            resized_frames.append(np.array(resized))

        return Video(
            uri=self.uri,
            data=np.array(resized_frames, dtype=np.uint8),
            fps=self.fps,
            duration=self.duration,
            has_audio=self.has_audio,
        )

    @property
    def width(self) -> int | None:
        """Get video width in pixels."""
        if self.data is not None:
            return self.data.shape[2]
        return None

    @property
    def height(self) -> int | None:
        """Get video height in pixels."""
        if self.data is not None:
            return self.data.shape[1]
        return None

    @staticmethod
    def _adjust_audio_duration(
        audio_data: np.ndarray,
        target_duration_s: float,
        sample_rate: int,
        mode: str = "loop",
    ) -> np.ndarray:
        """Adjust audio duration to match target duration.

        Args:
            audio_data: Audio data array (1D or 2D).
            target_duration_s: Target duration in seconds.
            sample_rate: Audio sample rate.
            mode: Duration adjustment mode - "loop" (default) or "silence".
                - "loop": Loop audio to fill duration
                - "silence": Pad with silence

        Returns:
            np.ndarray: Adjusted audio data.
        """
        target_samples = int(target_duration_s * sample_rate)

        # Flatten to 1D for processing
        original_shape = audio_data.shape
        is_2d = len(original_shape) > 1 and original_shape[0] <= 2

        if is_2d:
            # Handle 2D audio (channels, samples)
            channels = original_shape[0]
            flat_data = audio_data.reshape(channels, -1)
            current_samples = flat_data.shape[1]
        else:
            # Handle 1D audio
            flat_data = audio_data.flatten()
            current_samples = len(flat_data)
            channels = 1

        if current_samples >= target_samples:
            # Truncate if audio is longer
            audio_duration = current_samples / sample_rate
            msg.warn(
                f"Audio is longer than video ({audio_duration:.2f}s > {target_duration_s:.2f}s). "
                f"Audio will be truncated to match video duration."
            )
            if is_2d:
                return flat_data[:, :target_samples].reshape(original_shape)
            return flat_data[:target_samples]

        # Need to extend audio
        if mode == "loop":
            # Loop audio to fill duration
            repeats = (target_samples + current_samples - 1) // current_samples
            if is_2d:
                extended = np.tile(flat_data, (1, repeats))[:, :target_samples]
            else:
                extended = np.tile(flat_data, repeats)[:target_samples]
        elif mode == "silence":
            # Pad with silence
            padding_samples = target_samples - current_samples
            if is_2d:
                padding = np.zeros((channels, padding_samples), dtype=flat_data.dtype)
                extended = np.concatenate([flat_data, padding], axis=1)
            else:
                padding = np.zeros(padding_samples, dtype=flat_data.dtype)
                extended = np.concatenate([flat_data, padding])
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'loop' or 'silence'.")

        return extended

    def display(self):
        """Display the video in Jupyter Notebook.

        If video has audio, it will be embedded and playable.
        Requires the video to be saved to a file first or uses the uri.
        """
        try:
            from IPython.display import Video as IPVideo, display as IPDisplay

            # If video has URI and file exists, display it directly
            if self.uri and Path(self.uri).exists():
                IPDisplay(IPVideo(str(self.uri), embed=True))
            else:
                print(
                    f"Video: {self.width}x{self.height}, {self.fps}fps, {self.duration}s"
                )
                print(f"Tip: Save video to file to display with embedded player")
        except ImportError:
            print(f"Video: {self.width}x{self.height}, {self.fps}fps, {self.duration}s")

    def extract_audio(self) -> Audio:
        """Extract audio from the video.

        Returns:
            Audio: Audio object containing the extracted audio.

        Raises:
            ValueError: If video has no audio track or URI is not set.
            RuntimeError: If audio extraction fails.
        """
        if self.uri is None:
            raise ValueError("Video URI is not set")

        path = Path(self.uri)

        try:
            if path.exists():
                container = av.open(str(path))
            else:
                response = requests.get(str(self.uri), timeout=30)
                response.raise_for_status()
                container = av.open(BytesIO(response.content))

            # Check if video has audio stream
            audio_streams = [s for s in container.streams if s.type == "audio"]
            if not audio_streams:
                container.close()
                raise ValueError("Video has no audio track")

            audio_stream = audio_streams[0]

            # Decode audio frames
            audio_frames = []
            sample_rate = audio_stream.sample_rate

            for frame in container.decode(audio=0):
                # Convert to numpy array
                audio_frames.append(frame.to_ndarray())

            container.close()

            if not audio_frames:
                raise ValueError("No audio data found in video")

            # Concatenate audio frames
            audio_data = np.concatenate(audio_frames, axis=0)

            # Create Audio object
            audio = Audio()
            audio.data = audio_data
            audio.sample_rate = sample_rate

            return audio

        except ValueError:
            raise  # Re-raise ValueError as is
        except Exception as e:
            raise RuntimeError(f"Failed to extract audio: {e}") from e

    def merge_audio(
        self, audio: Audio, output_path: str | None = None, audio_mode: str = "loop"
    ) -> "Video":
        """Merge audio with video, replacing existing audio if present.

        Args:
            audio (Audio): Audio object to merge with the video.
            output_path (str, optional): Output path for the merged video.
                If None, returns a new Video object without saving to file.
            audio_mode (str): How to handle audio duration mismatch.
                - "loop": Loop audio to match video duration (default).
                - "silence": Pad with silence if audio is shorter.

        Returns:
            Video: New Video object with merged audio.

        Raises:
            ValueError: If video data or audio data is not set.
            RuntimeError: If merging fails.
        """
        if self.data is None:
            raise ValueError("Video data is not set")
        if audio.data is None:
            raise ValueError("Audio data is not set")
        if self.fps is None:
            raise ValueError("Video fps is not set")

        try:
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                tmp_path = tmp.name

            # Create output container
            container = av.open(tmp_path, mode="w", format="mp4")

            # Add video stream
            fps_fraction = Fraction(self.fps, 1)
            video_stream = container.add_stream("h264", rate=fps_fraction)
            video_stream.width = self.data.shape[2]
            video_stream.height = self.data.shape[1]
            video_stream.pix_fmt = "yuv420p"

            # Add audio stream
            audio_stream = container.add_stream("aac")
            audio_stream.sample_rate = audio.sample_rate
            audio_stream.layout = "stereo"

            # Encode video frames
            for frame_data in self.data:
                frame = av.VideoFrame.from_ndarray(frame_data, format="rgb24")
                frame = frame.reformat(format="yuv420p")
                for packet in video_stream.encode(frame):
                    container.mux(packet)

            # Flush video stream
            for packet in video_stream.encode():
                container.mux(packet)

            # Adjust audio duration to match video
            video_duration = (
                self.duration if self.duration else len(self.data) / self.fps
            )
            adjusted_audio = self._adjust_audio_duration(
                audio.data, video_duration, audio.sample_rate, mode=audio_mode
            )

            # Encode audio frames - simplified mono AAC encoding
            audio_chunk = adjusted_audio.flatten()

            # Normalize and convert to int16
            max_val = np.max(np.abs(audio_chunk))
            if max_val > 0:
                audio_chunk = audio_chunk / max_val * 0.9

            audio_int16 = (audio_chunk * 32767).astype(np.int16)

            # Create mono audio frame
            audio_frame = av.AudioFrame.from_ndarray(
                audio_int16.reshape(1, -1), format="s16", layout="mono"
            )
            audio_frame.sample_rate = audio.sample_rate

            for packet in audio_stream.encode(audio_frame):
                container.mux(packet)

            # Flush audio stream
            for packet in audio_stream.encode():
                container.mux(packet)

            container.close()

            # Load the merged video
            merged_video = Video(uri=tmp_path).load()
            merged_video.has_audio = True

            # If output_path specified, save to that location
            if output_path:
                merged_video.save(output_path)

            return merged_video

        except Exception as e:
            msg.fail(f"Failed to merge audio: {e}")
            raise RuntimeError(f"Failed to merge audio: {e}") from e

    def remove_audio(self, output_path: str | None = None) -> "Video":
        """Remove audio from the video.

        Args:
            output_path (str, optional): Output path for the video without audio.
                If None, returns a new Video object without saving to file.

        Returns:
            Video: New Video object without audio.

        Raises:
            ValueError: If video data is not set.
            RuntimeError: If audio removal fails.
        """
        if self.data is None:
            raise ValueError("Video data is not set")
        if self.fps is None:
            raise ValueError("Video fps is not set")

        try:
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                tmp_path = tmp.name

            # Create output container
            container = av.open(tmp_path, mode="w", format="mp4")

            # Add video stream only
            fps_fraction = Fraction(self.fps, 1)
            video_stream = container.add_stream("h264", rate=fps_fraction)
            video_stream.width = self.data.shape[2]
            video_stream.height = self.data.shape[1]
            video_stream.pix_fmt = "yuv420p"

            # Encode video frames
            for frame_data in self.data:
                frame = av.VideoFrame.from_ndarray(frame_data, format="rgb24")
                frame = frame.reformat(format="yuv420p")
                for packet in video_stream.encode(frame):
                    container.mux(packet)

            # Flush video stream
            for packet in video_stream.encode():
                container.mux(packet)

            container.close()

            # Load the video without audio
            no_audio_video = Video(uri=tmp_path).load()
            no_audio_video.has_audio = False

            # If output_path specified, save to that location
            if output_path:
                no_audio_video.save(output_path)

            return no_audio_video

        except Exception as e:
            msg.fail(f"Failed to remove audio: {e}")
            raise RuntimeError(f"Failed to remove audio: {e}") from e

    @staticmethod
    def combine_video_audio(
        video: "Video",
        audio: Audio,
        output_path: str,
        video_codec: str = "h264",
        audio_codec: str = "aac",
        audio_mode: str = "loop",
    ) -> "Video":
        """Static method to combine video and audio into a new file.

        Args:
            video (Video): Video object with video data.
            audio (Audio): Audio object to merge.
            output_path (str): Output path for the combined video.
            video_codec (str): Video codec to use. Default: "h264".
            audio_codec (str): Audio codec to use. Default: "aac".
            audio_mode (str): How to handle audio duration mismatch.
                - "loop": Loop audio to match video duration (default).
                - "silence": Pad with silence if audio is shorter.

        Returns:
            Video: New Video object with combined video and audio.

        Raises:
            ValueError: If video or audio data is not set.
            RuntimeError: If combination fails.
        """
        if video.data is None:
            raise ValueError("Video data is not set")
        if audio.data is None:
            raise ValueError("Audio data is not set")
        if video.fps is None:
            raise ValueError("Video fps is not set")

        if Path(output_path).exists():
            raise FileExistsError(f"File {output_path} already exists")

        try:
            # Create output container
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            container = av.open(output_path, mode="w", format="mp4")

            # Add video stream
            fps_fraction = Fraction(video.fps, 1)
            video_stream = container.add_stream(video_codec, rate=fps_fraction)
            video_stream.width = video.data.shape[2]
            video_stream.height = video.data.shape[1]
            video_stream.pix_fmt = "yuv420p"

            # Add audio stream
            audio_stream = container.add_stream(audio_codec)
            audio_stream.sample_rate = audio.sample_rate
            audio_stream.layout = "stereo"

            # Encode video frames
            for frame_data in video.data:
                frame = av.VideoFrame.from_ndarray(frame_data, format="rgb24")
                frame = frame.reformat(format="yuv420p")
                for packet in video_stream.encode(frame):
                    container.mux(packet)

            # Flush video stream
            for packet in video_stream.encode():
                container.mux(packet)

            # Adjust audio duration to match video
            video_duration = (
                video.duration if video.duration else len(video.data) / video.fps
            )
            adjusted_audio = Video._adjust_audio_duration(
                audio.data, video_duration, audio.sample_rate, mode=audio_mode
            )

            # Encode audio frames - simplified mono AAC encoding
            audio_chunk = adjusted_audio.flatten()

            # Normalize and convert to int16
            max_val = np.max(np.abs(audio_chunk))
            if max_val > 0:
                audio_chunk = audio_chunk / max_val * 0.9

            audio_int16 = (audio_chunk * 32767).astype(np.int16)

            # Create mono audio frame - use s16p (planar) for AAC compatibility
            audio_frame = av.AudioFrame.from_ndarray(
                audio_int16.reshape(1, -1), format="s16", layout="mono"
            )
            audio_frame.sample_rate = audio.sample_rate

            for packet in audio_stream.encode(audio_frame):
                container.mux(packet)

            # Flush audio stream
            for packet in audio_stream.encode():
                container.mux(packet)

            container.close()

            # Load and return the combined video
            combined_video = Video(uri=output_path).load()
            combined_video.has_audio = True
            return combined_video

        except Exception as e:
            msg.fail(f"Failed to combine video and audio: {e}")
            raise RuntimeError(f"Failed to combine video and audio: {e}") from e
