"""Tests for osc_data.audio.Audio."""

from pathlib import Path
import tempfile

import librosa
import numpy as np
import pytest

from osc_data.audio import Audio


class TestAudioSave:
    """Tests for Audio.save."""

    def test_save_wav_mono(self):
        audio = Audio()
        audio.sample_rate = 22050
        audio.data = np.random.randn(5000).astype(np.float32) * 0.1
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "out.wav"
            audio.save(str(path))
            assert path.exists()
            y, sr = librosa.load(str(path), sr=None, mono=True)
            assert sr == 22050
            assert y.shape == (5000,)

    def test_save_stereo_librosa_layout(self):
        audio = Audio()
        audio.sample_rate = 44100
        audio.data = np.random.randn(2, 3000).astype(np.float32) * 0.05
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "st.wav"
            audio.save(str(path))
            y, sr = librosa.load(str(path), sr=None, mono=False)
            assert sr == 44100
            assert y.shape == (2, 3000)

    def test_save_mp3_default(self):
        audio = Audio()
        audio.sample_rate = 22050
        audio.data = np.random.randn(4000).astype(np.float32) * 0.1
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "out.mp3"
            audio.save(str(path))
            assert path.exists()
            assert path.stat().st_size > 0

    def test_save_no_suffix_appends_mp3(self):
        audio = Audio()
        audio.sample_rate = 16000
        audio.data = np.zeros(2000, dtype=np.float32)
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir) / "clip"
            audio.save(str(base))
            mp3_path = base.with_suffix(".mp3")
            assert mp3_path.exists()

    def test_save_existing_path_raises(self):
        audio = Audio()
        audio.sample_rate = 8000
        audio.data = np.zeros(100, dtype=np.float32)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "exists.wav"
            path.write_bytes(b"")
            with pytest.raises(FileExistsError):
                audio.save(str(path))

    def test_save_without_data_raises(self):
        audio = Audio()
        audio.sample_rate = 8000
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="Audio data is not set"):
                audio.save(str(Path(tmpdir) / "x.wav"))

    def test_save_without_sample_rate_raises(self):
        audio = Audio()
        audio.data = np.zeros(100, dtype=np.float32)
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="Sample rate is not set"):
                audio.save(str(Path(tmpdir) / "x.wav"))
