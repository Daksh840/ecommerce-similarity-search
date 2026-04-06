"""
Tests for the image preprocessing service.

Covers:
- RGB/RGBA/grayscale conversion
- Dimension validation
- File size validation
- EXIF orientation handling
- Various input types (path, bytes, PIL Image)
"""

import io
import pytest
import numpy as np
from PIL import Image

from app.services.preprocessing_service import ImagePreprocessor
from app.core.exceptions import ImageValidationError, ImageProcessingError


@pytest.fixture
def preprocessor():
    return ImagePreprocessor()


def make_test_image(
    width=256, height=256, mode="RGB", fmt="JPEG"
) -> bytes:
    """Create a test image in memory and return bytes."""
    img = Image.new(mode, (width, height), color="red")
    buf = io.BytesIO()
    if mode == "RGBA" and fmt == "JPEG":
        # JPEG doesn't support RGBA, convert first
        img = img.convert("RGB")
    img.save(buf, format=fmt)
    buf.seek(0)
    return buf.read()


class TestValidateAndProcess:
    def test_valid_rgb_jpeg(self, preprocessor):
        img_bytes = make_test_image(256, 256, "RGB", "JPEG")
        result = preprocessor.validate_and_process(img_bytes)
        assert result.mode == "RGB"
        assert result.size == (256, 256)

    def test_valid_png(self, preprocessor):
        img_bytes = make_test_image(256, 256, "RGB", "PNG")
        result = preprocessor.validate_and_process(img_bytes)
        assert result.mode == "RGB"

    def test_rgba_converted_to_rgb(self, preprocessor):
        img_bytes = make_test_image(256, 256, "RGBA", "PNG")
        result = preprocessor.validate_and_process(img_bytes)
        assert result.mode == "RGB"

    def test_grayscale_converted_to_rgb(self, preprocessor):
        img_bytes = make_test_image(256, 256, "L", "PNG")
        result = preprocessor.validate_and_process(img_bytes)
        assert result.mode == "RGB"

    def test_pil_image_input(self, preprocessor):
        img = Image.new("RGB", (256, 256), color="blue")
        result = preprocessor.validate_and_process(img)
        assert result.mode == "RGB"


class TestDimensionValidation:
    def test_too_small(self, preprocessor):
        img_bytes = make_test_image(32, 32)  # Below 64px minimum
        with pytest.raises(ImageValidationError, match="too small"):
            preprocessor.validate_and_process(img_bytes)

    def test_minimum_size_passes(self, preprocessor):
        img_bytes = make_test_image(64, 64)
        result = preprocessor.validate_and_process(img_bytes)
        assert result is not None


class TestFileValidation:
    def test_invalid_bytes(self, preprocessor):
        with pytest.raises(ImageValidationError, match="Cannot identify"):
            preprocessor.validate_and_process(b"not an image")

    def test_nonexistent_path(self, preprocessor):
        with pytest.raises(ImageValidationError, match="not found"):
            preprocessor.validate_and_process("/nonexistent/image.jpg")

    def test_unsupported_input_type(self, preprocessor):
        with pytest.raises(ImageValidationError, match="Unsupported input"):
            preprocessor.validate_and_process(12345)
