from __future__ import annotations
from pathlib import Path
import requests

import numpy as np
from docarray import BaseDoc
from docarray.typing import ImageNdArray, ImageUrl
from pydantic import Field, ConfigDict

from osc_data._core import (
    convert_to_rgb,
    resize_image,
    crop_image,
    encode_image,
    decode_image,
    get_image_info,
    batch_resize,
)


class Image(BaseDoc):
    """Image object that represents an image file.

    Args:
        uri (Union[str, Path]): URL of the image file or local path.
        data (NdArray): Image data (H, W, C) uint8 RGB/RGBA.
        width (int): Image width in pixels.
        height (int): Image height in pixels.
        color_mode (str): Color mode of the image (e.g., "RGB", "RGBA", "L").
        source_format (str): Original file format (e.g., "png", "jpeg", "webp").
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True, validate_assignment=True, extra="ignore"
    )

    uri: ImageUrl | None = Field(
        None, description="URL of the image file or local path."
    )
    data: ImageNdArray = Field(None, description="Image data (H, W, C) uint8 RGB/RGBA.")
    width: int | None = Field(None, description="Image width in pixels.")
    height: int | None = Field(None, description="Image height in pixels.")
    color_mode: str | None = Field(None, description="Color mode (RGB, RGBA, L).")
    source_format: str | None = Field(None, description="Source file format (png, jpeg, webp).")

    def load(self) -> "Image":
        """Load the image from local path or URL.

        Returns:
            Image: The loaded image object with data populated.
        """
        if self.uri is None:
            raise ValueError("Image URI is not set")

        path = Path(self.uri)

        # Load from local file
        if path.exists():
            try:
                with open(path, "rb") as f:
                    img_bytes = f.read()
            except Exception as e:
                raise RuntimeError(f"Failed to read local image: {e}") from e
        else:
            # Load from URL
            try:
                response = requests.get(str(self.uri), timeout=30)
                response.raise_for_status()
                img_bytes = response.content
            except Exception as e:
                raise RuntimeError(f"Failed to download image from URL: {e}") from e

        # Decode using Rust core
        try:
            data, color_mode_str = decode_image(img_bytes)
            self.data = data
            self.height, self.width = data.shape[:2]
            self.color_mode = self._normalize_color_mode(color_mode_str)
            # Infer source format from URI extension
            self.source_format = path.suffix.lower().lstrip(".")
            if not self.source_format and "." in str(self.uri):
                # Handle URLs without standard extension (e.g., "/image.png?token=xxx")
                self.source_format = str(self.uri).split(".")[-1].split("?")[0].lower()
        except Exception as e:
            raise RuntimeError(f"Failed to decode image: {e}") from e

        return self

    def save(self, path: str, file_format: str | None = None, quality: int = 95) -> "Image":
        """Save the image to local path.

        Args:
            path (str): Local path to save the image.
            file_format (str, optional): Output file format (png, jpeg, webp, bmp, gif, tiff).
                If None, inferred from path extension.
            quality (int): Quality for lossy formats (0-100), default 95.

        Returns:
            Image: The image object itself for chaining.
        """
        if self.data is None:
            raise ValueError("Image data is not set")

        path_obj = Path(path)
        if path_obj.exists():
            raise FileExistsError(f"File {path} already exists")

        # Ensure parent directory exists
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        # Determine file format from extension if not specified
        if file_format is None:
            ext = path_obj.suffix.lower()
            format_map = {
                ".png": "png",
                ".jpg": "jpeg",
                ".jpeg": "jpeg",
                ".webp": "webp",
                ".bmp": "bmp",
                ".gif": "gif",
                ".tiff": "tiff",
                ".tif": "tiff",
            }
            file_format = format_map.get(ext)
            if file_format is None:
                raise ValueError(f"Cannot determine file format from extension: {ext}")

        # Ensure RGB for lossy formats
        if file_format in ("jpeg", "jpg") and self.color_mode != "RGB":
            data_to_save = self.to_rgb().data
        else:
            data_to_save = self.data

        # Encode using Rust core
        try:
            encoded_bytes = encode_image(data_to_save, file_format, quality)
            with open(path, "wb") as f:
                f.write(encoded_bytes)
        except Exception as e:
            raise RuntimeError(f"Failed to save image: {e}") from e

        return self

    def to_rgb(self) -> "Image":
        """Convert image to RGB format.

        Returns:
            Image: New Image object in RGB format.
        """
        if self.data is None:
            raise ValueError("Image data is not set")

        if self.color_mode == "RGB":
            return self

        rgb_data = convert_to_rgb(self.data, self.color_mode or "RGBA")
        return Image(
            uri=self.uri,
            data=rgb_data,
            width=self.width,
            height=self.height,
            color_mode="RGB",
        )

    def resize(self, width: int, height: int) -> "Image":
        """Resize image to target dimensions using high-quality Lanczos3 filter.

        Args:
            width (int): Target width in pixels.
            height (int): Target height in pixels.

        Returns:
            Image: New resized Image object.
        """
        if self.data is None:
            raise ValueError("Image data is not set")

        resized_data = resize_image(self.data, height, width)
        return Image(
            uri=self.uri,
            data=resized_data,
            width=width,
            height=height,
            color_mode=self.color_mode,
        )

    def crop(self, x: int, y: int, width: int, height: int) -> "Image":
        """Crop image to specified region.

        Args:
            x (int): X coordinate of top-left corner.
            y (int): Y coordinate of top-left corner.
            width (int): Crop width.
            height (int): Crop height.

        Returns:
            Image: New cropped Image object.
        """
        if self.data is None:
            raise ValueError("Image data is not set")

        cropped_data = crop_image(self.data, x, y, width, height)
        return Image(
            uri=self.uri,
            data=cropped_data,
            width=width,
            height=height,
            color_mode=self.color_mode,
        )

    @staticmethod
    def from_bytes(data: bytes) -> "Image":
        """Create Image from raw bytes.

        Args:
            data (bytes): Raw image bytes.

        Returns:
            Image: New Image object.
        """
        img_data, color_mode_str = decode_image(data)
        height, width = img_data.shape[:2]
        return Image(
            data=img_data,
            width=width,
            height=height,
            color_mode=Image._normalize_color_mode(color_mode_str),
        )

    def to_bytes(self, file_format: str = "png", quality: int = 95) -> bytes:
        """Convert image to bytes.

        Args:
            file_format (str): Output file format (png, jpeg, webp, etc.).
            quality (int): Quality for lossy formats (0-100).

        Returns:
            bytes: Encoded image bytes.
        """
        if self.data is None:
            raise ValueError("Image data is not set")

        return encode_image(self.data, file_format, quality)

    @staticmethod
    def get_info_from_bytes(data: bytes) -> tuple[int, int, str]:
        """Get image dimensions and format without full decoding.

        Args:
            data (bytes): Raw image bytes.

        Returns:
            tuple: (width, height, format_string)
        """
        return get_image_info(data)

    @staticmethod
    def batch_resize_images(
        images: list["Image"], width: int, height: int
    ) -> list["Image"]:
        """Batch resize multiple images.

        Args:
            images (list[Image]): List of Image objects.
            width (int): Target width.
            height (int): Target height.

        Returns:
            list[Image]: List of resized Image objects.
        """
        if not images:
            return []

        # Stack images into batch
        batch_data = np.stack([img.data for img in images if img.data is not None])
        resized_batch = batch_resize(batch_data, height, width)

        result = []
        for i, img in enumerate(images):
            if img.data is not None:
                result.append(
                    Image(
                        uri=img.uri,
                        data=resized_batch[i],
                        width=width,
                        height=height,
                        color_mode=img.color_mode,
                    )
                )
        return result

    def display(self):
        """Display the image using matplotlib if available."""
        if self.data is None:
            print("Image data is not set")
            return

        try:
            import matplotlib.pyplot as plt

            if self.color_mode == "L" or self.color_mode == "GRAYSCALE":
                plt.imshow(self.data.squeeze(), cmap="gray")
            else:
                plt.imshow(self.data)
            plt.axis("off")
            plt.show()
        except ImportError:
            print(f"Image shape: {self.data.shape}, color_mode: {self.color_mode}")

    @staticmethod
    def _normalize_color_mode(color_mode_str: str) -> str:
        """Normalize color mode string to standard format."""
        color_mode_lower = color_mode_str.lower()
        if "rgb8" in color_mode_lower:
            return "RGB"
        elif "rgba8" in color_mode_lower:
            return "RGBA"
        elif "luma8" in color_mode_lower or "gray" in color_mode_lower or "l8" in color_mode_lower:
            return "L"
        return color_mode_str.upper()
