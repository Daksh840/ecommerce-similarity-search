"""
Structured logging setup.

Why structured logging?
- Consistent format across all modules
- Easy to parse in production (ELK, CloudWatch, etc.)
- Configurable log level via environment variable
- Single setup point — all modules use `logging.getLogger(__name__)`

Interview Point: Always use module-level loggers (getLogger(__name__)),
never print(). Loggers are hierarchical — setting root logger level
propagates to all child loggers.
"""

import logging
import sys
from app.config import get_settings


def setup_logging():
    """
    Configure root logger with consistent formatting.
    
    Call once at app startup (in main.py).
    All modules using logging.getLogger(__name__) will inherit this config.
    """
    settings = get_settings()

    # Root logger configuration
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))

    # Avoid duplicate handlers on repeated calls
    if root_logger.handlers:
        return

    # Console handler with formatted output
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))

    formatter = logging.Formatter(
        fmt=settings.log_format,
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # Suppress noisy third-party loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    root_logger.info("Logging initialized at %s level", settings.log_level)
