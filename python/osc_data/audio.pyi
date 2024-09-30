from dataclasses import dataclass
from typing import Optional
import numpy as np

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
    
def low_frame_rate(frames: np.ndarray, m: int, n: int) -> np.ndarray:
    pass
