from typing import Union
from pathlib import Path
from io import BytesIO
import requests

from docarray import BaseDoc
from docarray.typing import NdArray
from pydantic import Field, ConfigDict
import librosa


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
        """Load the audio file from the URL."""
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
        example_path = Path(__file__).parent / "assets" / "audio" / "example.wav"
        self.uri = example_path
        return self.load()

    def display(self):
        """Display the audio file."""
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
