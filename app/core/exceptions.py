"""
Custom exception hierarchy for the application.

Why custom exceptions?
- Decouples HTTP layer from service layer
- Services raise domain exceptions, FastAPI handlers map to HTTP status codes
- Enables consistent error response formats
- Makes error handling testable and explicit

Interview Point: Never raise HTTPException inside service classes.
Services should be framework-agnostic. Let the API layer translate
domain exceptions into HTTP responses.
"""


class AppException(Exception):
    """Base exception for all application errors."""

    def __init__(self, message: str, detail: str = None):
        self.message = message
        self.detail = detail
        super().__init__(self.message)


class ImageValidationError(AppException):
    """Raised when an uploaded image fails validation checks."""
    pass


class ImageProcessingError(AppException):
    """Raised when image preprocessing or transformation fails."""
    pass


class EmbeddingExtractionError(AppException):
    """Raised when feature extraction from an image fails."""
    pass


class IndexNotFoundError(AppException):
    """Raised when the FAISS index file doesn't exist or can't be loaded."""
    pass


class IndexEmptyError(AppException):
    """Raised when searching an empty FAISS index."""
    pass


class InvalidQueryError(AppException):
    """Raised when a search query is malformed (wrong dimensions, etc.)."""
    pass
