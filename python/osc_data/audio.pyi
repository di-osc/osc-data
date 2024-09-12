from dataclasses import dataclass
from typing import Optional

@dataclass
class AudioDoc:
    url: str
    id: Optional[str]
    duration: Optional[float] = None
    mono: Optional[bool] = None
