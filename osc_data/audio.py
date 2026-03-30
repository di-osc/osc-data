from typing import Union
from pathlib import Path
from io import BytesIO

import av
import librosa
import numpy as np
import requests
from docarray import BaseDoc
from docarray.typing import NdArray
from pydantic import Field, ConfigDict


class Audio(BaseDoc):
    """Audio object that represents an audio file.

    Args:
        uri (Union[str, Path]): URL of the audio file or local path.
        sample_rate (Optional[int], optional): Sample rate of the audio file. Defaults to None.
        data (Optional[NdArray], optional): Data of the audio file. Defaults to None.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True, validate_assignment=True, extra="ignore"
    )
    uri: Union[str, Path] | None = Field(
        None, description="URL of the audio file or local path."
    )
    sample_rate: int | None = Field(None, description="Sample rate of the audio file.")
    data: NdArray = Field(None, description="Data of the audio file.")

    def load(self, sample_rate: int | None = None, mono: bool | None = None) -> "Audio":
        """Load the audio file from local path or URL.

        Args:
            sample_rate (int | None, optional): Target sample rate for resampling.
                If None, uses the native sample rate. Defaults to None.
            mono (bool | None, optional): If True, convert to mono channel.
                If None, keep original channels. Defaults to None.

        Returns:
            Audio: The loaded audio object with data populated.

        Raises:
            ValueError: If loading fails.
        """
        try:
            if Path(self.uri).exists():
                data, sample_rate = librosa.load(self.uri, sr=sample_rate, mono=mono)
            else:
                bytes_ = requests.get(self.uri).content
                data, sample_rate = librosa.load(
                    BytesIO(bytes_), sr=sample_rate, mono=mono
                )
            self.data = data
            self.sample_rate = sample_rate
        except Exception as e:
            raise ValueError(f"Failed to load audio from {self.uri}. {e}")
        return self

    def load_example(self) -> "Audio":
        """Load the built-in example audio file.

        Returns:
            Audio: The loaded example audio.
        """
        example_path = Path(__file__).parent / "assets" / "audio" / "example.wav"
        self.uri = str(example_path)
        return self.load()

    @staticmethod
    def _float_audio_to_planar(raw: np.ndarray) -> tuple[np.ndarray, str, str]:
        """将 float 音频转为 PyAV 所需的平面 int16 与 frame 格式。

        Returns:
            tuple: (planar_int16, av_sample_format, layout)
        """
        data = np.asarray(raw, dtype=np.float32)
        if data.ndim == 1:
            clipped = np.clip(data, -1.0, 1.0)
            planar = (clipped * 32767.0).astype(np.int16).reshape(1, -1)
            return planar, "s16", "mono"
        if data.ndim != 2:
            raise ValueError(
                f"Audio data must be 1D or 2D (channels, samples), got shape {data.shape}"
            )
        if data.shape[0] in (1, 2) and data.shape[0] < data.shape[1]:
            ch, _n = data.shape[0], data.shape[1]
            clipped = np.clip(data, -1.0, 1.0)
            planar = (clipped * 32767.0).astype(np.int16)
            if ch == 1:
                return planar, "s16", "mono"
            return planar, "s16p", "stereo"
        _n, ch = data.shape[0], data.shape[1]
        if ch not in (1, 2):
            raise ValueError(f"Unsupported channel count: {ch}")
        clipped = np.clip(data, -1.0, 1.0)
        planar = (clipped.T * 32767.0).astype(np.int16)
        if ch == 1:
            return planar, "s16", "mono"
        return planar, "s16p", "stereo"

    def _save_av(
        self,
        path: Path,
        planar: np.ndarray,
        av_sample_format: str,
        layout: str,
        fmt_upper: str,
    ) -> None:
        """使用 PyAV（FFmpeg）写入音频文件。"""
        fmt_upper = fmt_upper.upper()
        if fmt_upper == "MP3":
            open_kw: dict = {}
            codec = "libmp3lame"
        elif fmt_upper == "WAV":
            open_kw = {"format": "wav"}
            codec = "pcm_s16le"
        elif fmt_upper == "FLAC":
            open_kw = {"format": "flac"}
            codec = "flac"
        elif fmt_upper == "OGG":
            open_kw = {"format": "ogg"}
            codec = "libvorbis"
        else:
            raise ValueError(
                f"Unsupported format: {fmt_upper}. Use MP3, WAV, FLAC, or OGG."
            )

        container = av.open(str(path), mode="w", **open_kw)
        try:
            stream = container.add_stream(codec, rate=self.sample_rate)
            frame = av.AudioFrame.from_ndarray(
                planar, format=av_sample_format, layout=layout
            )
            frame.sample_rate = self.sample_rate
            for packet in stream.encode(frame):
                container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        finally:
            container.close()

    def save(self, path: str | Path, format: str | None = None) -> "Audio":
        """将音频数据写入本地文件（全部经 PyAV / FFmpeg 编码）。

        默认格式为 **MP3**（``libmp3lame``）。单声道为 ``(n_samples,)``；立体声 librosa 风格为
        ``(2, n_samples)``。

        Args:
            path (str | Path): 保存路径；文件已存在时抛出 FileExistsError。
                无后缀时自动追加 ``.mp3``。
            format (str | None): 容器格式，如 ``"MP3"``、``"WAV"``、``"FLAC"``、``"OGG"``。
                为 None 时根据路径后缀推断；无匹配后缀时默认为 MP3。

        Returns:
            Audio: 返回自身，便于链式调用。

        Raises:
            FileExistsError: 目标路径已存在文件。
            ValueError: 未设置 data 或 sample_rate，或不支持的格式。
            RuntimeError: 写入失败。

        Examples:
            >>> from osc_data.audio import Audio
            >>> audio = Audio(uri="input.wav").load()
            >>> audio.save("output.mp3")
            >>> audio.save("out.wav", format="WAV")
        """
        if self.data is None:
            raise ValueError("Audio data is not set")
        if self.sample_rate is None:
            raise ValueError("Sample rate is not set")

        path_obj = Path(path)
        if path_obj.suffix == "":
            path_obj = path_obj.with_suffix(".mp3")

        if path_obj.exists():
            raise FileExistsError(f"File {path_obj} already exists")
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        planar, av_sample_format, layout = self._float_audio_to_planar(self.data)

        if format is None:
            format_map = {
                ".mp3": "MP3",
                ".wav": "WAV",
                ".flac": "FLAC",
                ".ogg": "OGG",
            }
            format = format_map.get(path_obj.suffix.lower(), "MP3")

        try:
            self._save_av(
                path_obj, planar, av_sample_format, layout, format.upper()
            )
        except ValueError:
            raise
        except Exception as e:
            raise RuntimeError(f"Failed to save audio: {e}") from e

        return self

    def display(self):
        """Display the audio file in Jupyter Notebook.

        Returns:
            IPython.display.Audio: Audio player widget.

        Note:
            Requires IPython. In non-Jupyter environments, this will fail.
        """
        from IPython.display import Audio as IPAudio

        return IPAudio(
            data=self.data,
            rate=self.sample_rate,
        )

    @property
    def duration_s(self):
        """Get the duration of the audio file in seconds."""
        return librosa.get_duration(y=self.data, sr=self.sample_rate)

    @property
    def duration_ms(self):
        """Get the duration of the audio file in milliseconds."""
        return self.duration_s * 1000
