from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

try:
	from video_core import load_from_path as _load_from_path  # type: ignore
	from video_core import load_from_url as _load_from_url  # type: ignore
except Exception as exc:  # pragma: no cover
	raise RuntimeError(f"Failed to import Rust core: {exc}")


@dataclass(slots=True)
class Video:
	data: np.ndarray
	fps: float
	duration: float
	width: int
	height: int
	num_frames: int
	source: str

	@classmethod
	def from_path(cls, path: str) -> "Video":
		arr, fps, dur, w, h, n = _load_from_path(path)
		data = np.asarray(arr, dtype=np.uint8)
		return cls(data=data, fps=float(fps), duration=float(dur), width=int(w), height=int(h), num_frames=int(n), source=path)

	@classmethod
	def from_url(cls, url: str) -> "Video":
		arr, fps, dur, w, h, n = _load_from_url(url)
		data = np.asarray(arr, dtype=np.uint8)
		return cls(data=data, fps=float(fps), duration=float(dur), width=int(w), height=int(h), num_frames=int(n), source=url)

	@property
	def shape(self) -> Tuple[int, int, int, int]:
		return tuple(self.data.shape)  # type: ignore[return-value]

	def __repr__(self) -> str:  # pragma: no cover
		return (
			f"Video(source={self.source!r}, shape={self.data.shape}, fps={self.fps:.3f}, "
			f"duration={self.duration:.3f}s)"
		)