"""Tests for the Image class."""

from pathlib import Path
import tempfile
import numpy as np
import pytest

from osc_data.image import Image

ASSETS_DIR = Path(__file__).parent.parent / "osc_data" / "assets" / "image"


class TestImageLoad:
    """Tests for image loading functionality."""

    def test_load_rgb_png(self):
        """Test loading RGB PNG image from local path."""
        img_path = ASSETS_DIR / "example_rgb.png"
        img = Image(uri=str(img_path)).load()

        assert img.data is not None
        assert img.width == 256
        assert img.height == 256
        assert img.color_mode == "RGB"
        assert img.data.shape == (256, 256, 3)
        assert img.data.dtype == np.uint8

    def test_load_rgba_png(self):
        """Test loading RGBA PNG image."""
        img_path = ASSETS_DIR / "example_rgba.png"
        img = Image(uri=str(img_path)).load()

        assert img.data is not None
        assert img.width == 256
        assert img.height == 256
        assert img.color_mode == "RGBA"
        assert img.data.shape == (256, 256, 4)

    def test_load_grayscale(self):
        """Test loading grayscale image."""
        img_path = ASSETS_DIR / "example_gray.png"
        img = Image(uri=str(img_path)).load()

        assert img.data is not None
        assert img.width == 128
        assert img.height == 128
        assert img.color_mode in ("L", "L8")
        assert img.data.shape == (128, 128, 1)

    def test_load_jpeg(self):
        """Test loading JPEG image."""
        img_path = ASSETS_DIR / "example.jpg"
        img = Image(uri=str(img_path)).load()

        assert img.data is not None
        assert img.width == 256
        assert img.height == 256
        assert img.color_mode == "RGB"

    def test_load_without_uri_raises(self):
        """Test that loading without URI raises ValueError."""
        img = Image()
        with pytest.raises(ValueError, match="URI is not set"):
            img.load()


class TestImageSave:
    """Tests for image saving functionality."""

    def test_save_png(self):
        """Test saving image as PNG."""
        img_path = ASSETS_DIR / "example_rgb.png"
        img = Image(uri=str(img_path)).load()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir) / "test_output.png"
            img.save(str(tmp_path))
            assert tmp_path.exists()

            # Load saved image and verify
            saved_img = Image(uri=str(tmp_path)).load()
            assert saved_img.width == img.width
            assert saved_img.height == img.height
            assert saved_img.color_mode == "RGB"

    def test_save_jpeg(self):
        """Test saving image as JPEG."""
        img_path = ASSETS_DIR / "example_rgb.png"
        img = Image(uri=str(img_path)).load()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir) / "test_output.jpg"
            img.save(str(tmp_path), file_format="jpeg", quality=90)
            assert tmp_path.exists()

            saved_img = Image(uri=str(tmp_path)).load()
            assert saved_img.color_mode == "RGB"

    def test_save_webp(self):
        """Test saving image as WebP."""
        img_path = ASSETS_DIR / "example_rgb.png"
        img = Image(uri=str(img_path)).load()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir) / "test_output.webp"
            img.save(str(tmp_path), file_format="webp")
            assert tmp_path.exists()

            saved_img = Image(uri=str(tmp_path)).load()
            assert saved_img.width == img.width
            assert saved_img.height == img.height

    def test_save_to_existing_path_raises(self):
        """Test that saving to existing path raises FileExistsError."""
        img_path = ASSETS_DIR / "example_rgb.png"
        img = Image(uri=str(img_path)).load()

        with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
            with pytest.raises(FileExistsError):
                img.save(tmp.name)

    def test_save_without_data_raises(self):
        """Test that saving without data raises ValueError."""
        img = Image()
        with pytest.raises(ValueError, match="data is not set"):
            img.save("/tmp/test.png")


class TestImageConvert:
    """Tests for image conversion operations."""

    def test_to_rgb_from_rgba(self):
        """Test converting RGBA to RGB."""
        img_path = ASSETS_DIR / "example_rgba.png"
        img = Image(uri=str(img_path)).load()
        assert img.color_mode == "RGBA"

        rgb_img = img.to_rgb()
        assert rgb_img.color_mode == "RGB"
        assert rgb_img.data.shape == (256, 256, 3)
        assert rgb_img.width == img.width
        assert rgb_img.height == img.height

    def test_to_rgb_from_grayscale(self):
        """Test converting grayscale to RGB."""
        img_path = ASSETS_DIR / "example_gray.png"
        img = Image(uri=str(img_path)).load()
        assert img.color_mode in ("L", "L8")

        rgb_img = img.to_rgb()
        assert rgb_img.color_mode == "RGB"
        assert rgb_img.data.shape == (128, 128, 3)

    def test_to_rgb_already_rgb(self):
        """Test that to_rgb returns same object if already RGB."""
        img_path = ASSETS_DIR / "example_rgb.png"
        img = Image(uri=str(img_path)).load()
        assert img.color_mode == "RGB"

        rgb_img = img.to_rgb()
        assert rgb_img is img  # Should return same object


class TestImageResize:
    """Tests for image resize operations."""

    def test_resize_down(self):
        """Test resizing image to smaller dimensions."""
        img_path = ASSETS_DIR / "example_rgb.png"
        img = Image(uri=str(img_path)).load()

        resized = img.resize(128, 128)
        assert resized.width == 128
        assert resized.height == 128
        assert resized.data.shape == (128, 128, 3)
        assert resized.color_mode == img.color_mode

    def test_resize_up(self):
        """Test resizing image to larger dimensions."""
        img_path = ASSETS_DIR / "example_rgb.png"
        img = Image(uri=str(img_path)).load()

        resized = img.resize(512, 512)
        assert resized.width == 512
        assert resized.height == 512
        assert resized.data.shape == (512, 512, 3)

    def test_resize_without_data_raises(self):
        """Test that resizing without data raises ValueError."""
        img = Image()
        with pytest.raises(ValueError, match="data is not set"):
            img.resize(100, 100)


class TestImageCrop:
    """Tests for image crop operations."""

    def test_crop_center(self):
        """Test cropping center region of image."""
        img_path = ASSETS_DIR / "example_rgb.png"
        img = Image(uri=str(img_path)).load()

        # Crop 100x100 from center
        x = (img.width - 100) // 2
        y = (img.height - 100) // 2
        cropped = img.crop(x, y, 100, 100)

        assert cropped.width == 100
        assert cropped.height == 100
        assert cropped.data.shape == (100, 100, 3)

    def test_crop_top_left(self):
        """Test cropping top-left corner."""
        img_path = ASSETS_DIR / "example_rgb.png"
        img = Image(uri=str(img_path)).load()

        cropped = img.crop(0, 0, 50, 50)
        assert cropped.width == 50
        assert cropped.height == 50

    def test_crop_out_of_bounds_raises(self):
        """Test that cropping out of bounds raises ValueError."""
        img_path = ASSETS_DIR / "example_rgb.png"
        img = Image(uri=str(img_path)).load()

        with pytest.raises(ValueError, match="exceeds image bounds"):
            img.crop(200, 200, 100, 100)  # Would exceed 256x256 bounds


class TestImageBytes:
    """Tests for image bytes conversion."""

    def test_to_bytes_png(self):
        """Test converting image to PNG bytes."""
        img_path = ASSETS_DIR / "example_rgb.png"
        img = Image(uri=str(img_path)).load()

        img_bytes = img.to_bytes(file_format="png")
        assert isinstance(img_bytes, bytes)
        assert len(img_bytes) > 0
        # PNG magic bytes
        assert img_bytes[:8] == b'\x89PNG\r\n\x1a\n'

    def test_to_bytes_jpeg(self):
        """Test converting image to JPEG bytes."""
        img_path = ASSETS_DIR / "example_rgb.png"
        img = Image(uri=str(img_path)).load()

        img_bytes = img.to_bytes(file_format="jpeg", quality=90)
        assert isinstance(img_bytes, bytes)
        assert len(img_bytes) > 0
        # JPEG magic bytes
        assert img_bytes[:3] == b'\xff\xd8\xff'

    def test_from_bytes(self):
        """Test creating image from bytes."""
        img_path = ASSETS_DIR / "example_rgb.png"
        with open(img_path, "rb") as f:
            img_bytes = f.read()

        img = Image.from_bytes(img_bytes)
        assert img.data is not None
        assert img.width == 256
        assert img.height == 256
        assert img.color_mode == "RGB"


class TestImageInfo:
    """Tests for image metadata retrieval."""

    def test_get_image_info(self):
        """Test getting image info without full decoding."""
        img_path = ASSETS_DIR / "example_rgb.png"
        with open(img_path, "rb") as f:
            img_bytes = f.read()

        width, height, fmt = Image.get_info_from_bytes(img_bytes)
        assert width == 256
        assert height == 256
        assert "Png" in fmt or "PNG" in fmt


class TestBatchOperations:
    """Tests for batch image operations."""

    def test_batch_resize(self):
        """Test batch resizing multiple images."""
        img_path = ASSETS_DIR / "example_rgb.png"
        img1 = Image(uri=str(img_path)).load()
        img2 = Image(uri=str(img_path)).load()

        images = [img1, img2]
        resized = Image.batch_resize_images(images, 128, 128)

        assert len(resized) == 2
        for rimg in resized:
            assert rimg.width == 128
            assert rimg.height == 128
            assert rimg.data.shape == (128, 128, 3)

    def test_batch_resize_empty_list(self):
        """Test batch resize with empty list."""
        result = Image.batch_resize_images([], 128, 128)
        assert result == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
