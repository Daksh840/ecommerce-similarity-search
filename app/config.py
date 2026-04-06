"""
Centralized configuration using Pydantic Settings.

Why Pydantic Settings?
- Type validation at startup (fail fast, not at runtime)
- Automatic .env file loading
- Environment variable override (12-factor app compliance)
- Nested configuration with clear defaults

Interview Point: Config should be immutable after startup.
Use @lru_cache to create a singleton — avoids re-parsing env vars on every request.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings


# Project root directory
BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """
    Application settings with environment variable support.

    Usage:
        settings = get_settings()
        print(settings.model_name)

    Override via environment:
        export EMBEDDING_DIMENSION=512
        export INDEX_PATH=/custom/path
    """

    # --- App ---
    app_name: str = "E-Commerce Similarity Search API"
    app_version: str = "1.0.0"
    debug: bool = False

    # --- Model ---
    model_name: str = "resnet50"
    embedding_dimension: int = 2048
    batch_size: int = 32

    # --- FAISS Index ---
    index_path: str = str(BASE_DIR / "data" / "faiss_index")
    index_type: str = "IndexFlatIP"  # Options: IndexFlatIP, IndexIVFFlat, IndexIVFPQ

    # --- Data ---
    data_dir: str = str(BASE_DIR / "data")
    image_dir: str = str(BASE_DIR / "data" / "images")
    data_path: str = str(BASE_DIR / "data" / "products.json")

    # --- API ---
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    redis_url: str = "redis://localhost:6379"
    max_upload_size_mb: int = 10
    default_top_k: int = 5
    max_top_k: int = 20

    # --- Preprocessing ---
    min_image_size: int = 64       # Minimum dimension in pixels
    max_image_size: int = 4096     # Maximum dimension in pixels
    supported_formats: list = ["JPEG", "PNG", "WEBP", "BMP"]

    # --- Logging ---
    log_level: str = "INFO"
    log_format: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

    model_config = {
        "env_file": str(BASE_DIR / ".env"),
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
    }


@lru_cache()
def get_settings() -> Settings:
    """
    Singleton settings instance.

    @lru_cache ensures this is only created once.
    For testing, call get_settings.cache_clear() to reset.
    """
    return Settings()
