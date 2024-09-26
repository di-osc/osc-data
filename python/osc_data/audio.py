from osc_data._lowlevel import audio
import requests
import librosa
from io import BytesIO
from pathlib import Path


class Audio(audio.Audio):
    def load_url(self) -> None:
        bytes = BytesIO(requests.get(self.path).content)
        waveform, self.sample_rate = librosa.load(bytes, sr=None, mono=False)
        if waveform.ndim == 1:
            waveform = waveform.reshape(1, -1)
        self.waveform = waveform
        duration = librosa.get_duration(y=self.waveform, sr=self.sample_rate)
        self.duration = duration

    def load_local(self) -> None:
        if not Path(self.path).exists():
            raise ValueError(f"File not found: {self.path}")
        waveform, self.sample_rate = librosa.load(self.path, sr=None, mono=False)
        if waveform.ndim == 1:
            waveform = waveform.reshape(1, -1)
        self.waveform = waveform
        duration = librosa.get_duration(y=self.waveform, sr=self.sample_rate)
        self.duration = duration

    def __repr__(self):
        return f'<Audio path="{self.path}" desc="{self.desc}" sample_rate="{self.sample_rate}">'
