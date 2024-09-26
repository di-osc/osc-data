from dataclasses import dataclass
from typing import Optional

@dataclass
class Audio:
    path: str
    id: Optional[str]
    duration: Optional[float] = None
    mono: Optional[bool] = None
    sample_rate: Optional[int] = None
    waveform: Optional[bytes] = None

    def load_url(self) -> None:
        pass

    def load_local(self) -> None:
        pass
