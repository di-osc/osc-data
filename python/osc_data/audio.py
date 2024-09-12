from osc_data._lowlevel import AudioDoc as _AudioDoc
import requests
import librosa
from io import BytesIO


class AudioDoc(_AudioDoc):

    def load(self) -> None:
        bytes = requests.get(self.url).content
        self._audio, self.sr = librosa.load(BytesIO(bytes), sr=None, mono=False)
        duration = librosa.get_duration(y=self._audio, sr=self.sr)
        self.duration = duration

    def __repr__(self):
        return f'<AudioDoc url="{self.url}" id="{self.id}" duration={self.duration} mono={self.mono}>'


__all__ = ["AudioDoc"]
