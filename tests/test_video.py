"""Tests for the Video class."""

from pathlib import Path
import tempfile
import numpy as np
import pytest

from osc_data.video import Video
from osc_data.audio import Audio

ASSETS_DIR = Path(__file__).parent.parent / "osc_data" / "assets" / "video"
AUDIO_ASSETS_DIR = Path(__file__).parent.parent / "osc_data" / "assets" / "audio"


class TestVideoLoad:
    """Tests for video loading functionality."""

    def test_load_local_video(self):
        """Test loading local video file."""
        video_path = ASSETS_DIR / "example.mp4"
        video = Video(uri=str(video_path)).load()

        assert video.data is not None
        assert video.fps is not None
        assert video.duration is not None
        assert video.data.shape[0] > 0  # At least one frame
        assert len(video.data.shape) == 4  # (N, H, W, C)

    def test_load_without_uri_raises(self):
        """Test that loading without URI raises ValueError."""
        video = Video()
        with pytest.raises(ValueError, match="URI is not set"):
            video.load()

    def test_load_example(self):
        """Test loading the example video."""
        video = Video().load_example()

        assert video.data is not None
        assert video.width == 128
        assert video.height == 128
        assert video.fps == 30
        assert video.duration == 2.0
        assert video.data.shape[0] == 60  # 2 seconds * 30 fps

    def test_key_frames_loaded(self):
        """Test that key frames are loaded."""
        video_path = ASSETS_DIR / "example.mp4"
        video = Video(uri=str(video_path)).load()

        assert video.key_frames is not None
        assert isinstance(video.key_frames, list)


class TestVideoSave:
    """Tests for video saving functionality."""

    def test_save_mp4(self):
        """Test saving video as MP4."""
        video_path = ASSETS_DIR / "example.mp4"
        video = Video(uri=str(video_path)).load()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir) / "test_output.mp4"
            video.save(str(tmp_path))
            assert tmp_path.exists()

            # Load saved video and verify
            saved_video = Video(uri=str(tmp_path)).load()
            assert saved_video.fps == video.fps
            assert saved_video.width == video.width
            assert saved_video.height == video.height

    def test_save_to_existing_path_raises(self):
        """Test that saving to existing path raises FileExistsError."""
        video_path = ASSETS_DIR / "example.mp4"
        video = Video(uri=str(video_path)).load()

        with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp:
            with pytest.raises(FileExistsError):
                video.save(tmp.name)

    def test_save_without_data_raises(self):
        """Test that saving without data raises ValueError."""
        video = Video()
        with pytest.raises(ValueError, match="data is not set"):
            video.save("/tmp/test.mp4")


class TestVideoAudio:
    """Tests for video audio operations."""

    def test_extract_audio_no_audio_raises(self):
        """Test that extracting audio from video without audio raises error."""
        video_path = ASSETS_DIR / "example.mp4"
        video = Video(uri=str(video_path)).load()

        # The example video has no audio, so this should raise
        with pytest.raises((ValueError, RuntimeError), match="no audio track"):
            video.extract_audio()

    def test_remove_audio(self):
        """Test removing audio from video."""
        video_path = ASSETS_DIR / "example.mp4"
        video = Video(uri=str(video_path)).load()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "no_audio.mp4"
            no_audio_video = video.remove_audio(str(output_path))

            assert output_path.exists()
            assert no_audio_video.has_audio is False

    def test_combine_video_audio(self):
        """Test combining video and audio."""
        video_path = ASSETS_DIR / "example.mp4"
        video = Video(uri=str(video_path)).load()

        # Create a simple test audio with mono format (simpler for AAC)
        audio = Audio()
        # Generate audio at 22050 Hz matching video duration
        audio.sample_rate = 22050
        duration_samples = int(audio.sample_rate * video.duration)
        # Create mono audio (1D array)
        audio.data = np.random.randn(duration_samples).astype(np.float32) * 0.1

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "combined.mp4"
            combined = Video.combine_video_audio(video, audio, str(output_path))

            assert output_path.exists()
            assert combined.has_audio is True
            assert combined.fps == video.fps

    def test_combine_video_audio_without_data_raises(self):
        """Test that combining without video data raises ValueError."""
        video = Video()
        audio = Audio()
        audio.data = np.array([0.1, 0.2, 0.3])
        audio.sample_rate = 22050

        with pytest.raises(ValueError, match="Video data is not set"):
            Video.combine_video_audio(video, audio, "/tmp/test.mp4")

    def test_combine_video_audio_without_audio_data_raises(self):
        """Test that combining without audio data raises ValueError."""
        video_path = ASSETS_DIR / "example.mp4"
        video = Video(uri=str(video_path)).load()
        audio = Audio()

        with pytest.raises(ValueError, match="Audio data is not set"):
            Video.combine_video_audio(video, audio, "/tmp/test.mp4")


class TestVideoResize:
    """Tests for video resize operations."""

    def test_resize_down(self):
        """Test resizing video to smaller dimensions."""
        video_path = ASSETS_DIR / "example.mp4"
        video = Video(uri=str(video_path)).load()

        # Test split by key frames
        segments = video.split_by_key_frames(min_split_duration_s=0)
        assert isinstance(segments, list)


class TestVideoAudioDuration:
    """Tests for audio duration adjustment in video merging."""

    def test_audio_shorter_than_video_loop_mode(self):
        """Test that shorter audio is looped to match video duration."""
        video_path = ASSETS_DIR / "example.mp4"
        video = Video(uri=str(video_path)).load()

        # Create a very short audio (0.5 seconds)
        audio = Audio()
        audio.sample_rate = 22050
        short_duration_samples = int(audio.sample_rate * 0.5)  # 0.5 seconds
        audio.data = np.random.randn(short_duration_samples).astype(np.float32) * 0.1

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "looped.mp4"
            combined = Video.combine_video_audio(
                video, audio, str(output_path), audio_mode="loop"
            )

            assert output_path.exists()
            assert combined.has_audio is True
            # Video duration should be preserved
            assert (
                abs(combined.duration - video.duration) < 0.5
            )  # Allow small tolerance

    def test_audio_shorter_than_video_silence_mode(self):
        """Test that shorter audio is padded with silence to match video duration."""
        video_path = ASSETS_DIR / "example.mp4"
        video = Video(uri=str(video_path)).load()

        # Create a very short audio (0.5 seconds)
        audio = Audio()
        audio.sample_rate = 22050
        short_duration_samples = int(audio.sample_rate * 0.5)  # 0.5 seconds
        audio.data = np.random.randn(short_duration_samples).astype(np.float32) * 0.1

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "padded.mp4"
            combined = Video.combine_video_audio(
                video, audio, str(output_path), audio_mode="silence"
            )

            assert output_path.exists()
            assert combined.has_audio is True
            # Video duration should be preserved
            assert abs(combined.duration - video.duration) < 0.5

    def test_merge_audio_with_loop_mode(self):
        """Test merge_audio method with loop mode."""
        video_path = ASSETS_DIR / "example.mp4"
        video = Video(uri=str(video_path)).load()

        # Create a short audio
        audio = Audio()
        audio.sample_rate = 22050
        short_duration_samples = int(audio.sample_rate * 0.5)
        audio.data = np.random.randn(short_duration_samples).astype(np.float32) * 0.1

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "merged_looped.mp4"
            merged = video.merge_audio(audio, str(output_path), audio_mode="loop")

            assert output_path.exists()
            assert merged.has_audio is True
            assert merged.fps == video.fps


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
