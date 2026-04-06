"""
Image preprocessing and validation service.

Why a dedicated preprocessing layer?
- Validates uploads BEFORE expensive GPU inference
- Normalizes inconsistent formats (RGBA→RGB, EXIF rotation, etc.)
- Production images are messy: truncated files, wrong extensions, CMYK color spaces
- Fail fast with clear error messages

Interview Points:
- Always validate image *content*, not just file extension (security)
- EXIF orientation: phone photos may appear rotated without correction
- RGBA→RGB conversion: ResNet expects 3-channel input, not 4
- Memory: use img.verify() for validation, then re-open for processing
"""

from PIL import Image, ImageOps, UnidentifiedImageError
from pathlib import Path
from typing import Union, Tuple
import io
import logging

from app.config import get_settings
from app.core.exceptions import ImageValidationError, ImageProcessingError

logger = logging.getLogger(__name__)
settings = get_settings()


class ImagePreprocessor:
    """
    Validates and normalizes images before embedding extraction.
    
    Pipeline:
    1. Validate format and integrity
    2. Fix EXIF orientation (phone photos)
    3. Convert color mode (RGBA/CMYK → RGB)
    4. Validate dimensions
    5. Return clean PIL.Image ready for embedding
    """

    SUPPORTED_FORMATS = set(settings.supported_formats)
    MIN_SIZE = settings.min_image_size
    MAX_SIZE = settings.max_image_size
    MAX_BYTES = settings.max_upload_size_mb * 1024 * 1024

    def validate_and_process(
        self, image_input: Union[str, bytes, Image.Image]
    ) -> Image.Image:
        """
        Full preprocessing pipeline.
        
        Args:
            image_input: File path (str), raw bytes, or PIL Image
        
        Returns:
            Cleaned RGB PIL.Image ready for embedding extraction
        
        Raises:
            ImageValidationError: If image fails validation
            ImageProcessingError: If preprocessing fails
        """
        try:
            image = self._load_image(image_input)
            self._validate_format(image)
            self._validate_dimensions(image)
            image = self._fix_orientation(image)
            image = self._convert_to_rgb(image)
            return image

        except ImageValidationError:
            raise  # Re-raise validation errors as-is
        except Exception as e:
            logger.error("Preprocessing failed: %s", e)
            raise ImageProcessingError(
                message="Image preprocessing failed",
                detail=str(e)
            )

    def _load_image(self, image_input: Union[str, bytes, Image.Image]) -> Image.Image:
        """Load image from various input types."""
        if isinstance(image_input, Image.Image):
            return image_input

        if isinstance(image_input, bytes):
            if len(image_input) > self.MAX_BYTES:
                raise ImageValidationError(
                    message=f"File too large: {len(image_input) / 1024 / 1024:.1f}MB",
                    detail=f"Maximum allowed: {settings.max_upload_size_mb}MB"
                )
            try:
                return Image.open(io.BytesIO(image_input))
            except UnidentifiedImageError:
                raise ImageValidationError(
                    message="Cannot identify image format",
                    detail="The uploaded file is not a valid image"
                )

        if isinstance(image_input, str):
            path = Path(image_input)
            if not path.exists():
                raise ImageValidationError(
                    message=f"Image file not found: {path.name}",
                    detail=str(path)
                )
            if path.stat().st_size > self.MAX_BYTES:
                raise ImageValidationError(
                    message=f"File too large: {path.stat().st_size / 1024 / 1024:.1f}MB",
                    detail=f"Maximum allowed: {settings.max_upload_size_mb}MB"
                )
            try:
                return Image.open(path)
            except UnidentifiedImageError:
                raise ImageValidationError(
                    message="Cannot identify image format",
                    detail=f"File: {path.name}"
                )

        raise ImageValidationError(
            message="Unsupported input type",
            detail=f"Expected str, bytes, or PIL.Image, got {type(image_input).__name__}"
        )

    def _validate_format(self, image: Image.Image):
        """Check image format is supported."""
        fmt = image.format
        if fmt and fmt.upper() not in self.SUPPORTED_FORMATS:
            raise ImageValidationError(
                message=f"Unsupported image format: {fmt}",
                detail=f"Supported formats: {', '.join(self.SUPPORTED_FORMATS)}"
            )

    def _validate_dimensions(self, image: Image.Image):
        """Check image dimensions are within bounds."""
        w, h = image.size

        if w < self.MIN_SIZE or h < self.MIN_SIZE:
            raise ImageValidationError(
                message=f"Image too small: {w}x{h}",
                detail=f"Minimum dimension: {self.MIN_SIZE}px"
            )

        if w > self.MAX_SIZE or h > self.MAX_SIZE:
            raise ImageValidationError(
                message=f"Image too large: {w}x{h}",
                detail=f"Maximum dimension: {self.MAX_SIZE}px"
            )

    def _fix_orientation(self, image: Image.Image) -> Image.Image:
        """
        Apply EXIF orientation tag.
        
        Phone cameras store rotation as EXIF metadata instead of
        actually rotating the pixels. Without this fix, portrait photos
        appear sideways.
        """
        try:
            return ImageOps.exif_transpose(image)
        except Exception:
            # Some images have malformed EXIF — safe to ignore
            return image

    def _convert_to_rgb(self, image: Image.Image) -> Image.Image:
        """
        Convert to RGB color mode.
        
        - RGBA (PNGs with transparency): composite on white background
        - L (grayscale): convert to 3-channel
        - CMYK (print images): convert to RGB
        - P (palette): convert to RGB
        """
        if image.mode == 'RGB':
            return image

        if image.mode == 'RGBA':
            # Composite on white background (alpha → solid)
            background = Image.new('RGB', image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[3])
            return background

        # Handles L, CMYK, P, etc.
        return image.convert('RGB')
