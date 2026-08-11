"""Tests for osc_data.audio.Audio."""

import struct
import tempfile
import wave

from pathlib import Path

import numpy as np
import pytest

from osc_data.audio import Audio


class TestAudioLoad:
    """Tests for Rust-backed Audio.load."""

    def test_load_local_audio(self):
        path = (
            Path(__file__).parent.parent
            / "osc_data"
            / "assets"
            / "audio"
            / "example.wav"
        )
        audio = Audio(uri=path).load()

        assert audio.sample_rate == 16000
        assert audio.data.dtype == np.float32
        assert audio.data.shape == (67263,)
        assert np.isfinite(audio.data).all()
        assert np.max(np.abs(audio.data)) <= 1.0

    def test_load_supports_resampling_and_mono(self):
        path = (
            Path(__file__).parent.parent
            / "osc_data"
            / "assets"
            / "audio"
            / "example.wav"
        )
        audio = Audio(uri=path).load(sample_rate=8000, mono=True)

        assert audio.sample_rate == 8000
        assert audio.data.ndim == 1
        assert audio.data.shape[0] == 33632

    def test_load_preserves_stereo_layout_when_mono_is_not_requested(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "stereo.wav"
            samples = [0, 1000, -1000, 2000, 2000, -2000]
            with wave.open(str(path), "wb") as writer:
                writer.setnchannels(2)
                writer.setsampwidth(2)
                writer.setframerate(8000)
                writer.writeframes(
                    b"".join(struct.pack("<h", value) for value in samples)
                )

            audio = Audio(uri=path).load(mono=None)

            assert audio.sample_rate == 8000
            assert audio.data.shape == (2, 3)

    def test_load_rejects_zero_target_sample_rate(self):
        path = (
            Path(__file__).parent.parent
            / "osc_data"
            / "assets"
            / "audio"
            / "example.wav"
        )
        with pytest.raises(ValueError, match="sample_rate must be greater than zero"):
            Audio(uri=path).load(sample_rate=0)


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
            with wave.open(str(path), "rb") as reader:
                assert reader.getframerate() == 22050
                assert reader.getnchannels() == 1
                assert reader.getnframes() == 5000

    def test_save_stereo_layout(self):
        audio = Audio()
        audio.sample_rate = 44100
        audio.data = np.random.randn(2, 3000).astype(np.float32) * 0.05
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "st.wav"
            audio.save(str(path))
            with wave.open(str(path), "rb") as reader:
                assert reader.getframerate() == 44100
                assert reader.getnchannels() == 2
                assert reader.getnframes() == 3000

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
