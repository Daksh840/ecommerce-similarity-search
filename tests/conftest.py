"""
Shared test fixtures for the test suite.

Provides:
- Mock embedding extractor (fast, no GPU needed)
- Test image generators
- Temporary directory management
"""

import io
import tempfile
import shutil

import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def tmp_dir():
    """Temporary directory, cleaned up after test."""
    path = tempfile.mkdtemp()
    yield path
    shutil.rmtree(path, ignore_errors=True)


@pytest.fixture
def sample_image_bytes():
    """Generate a valid JPEG image as bytes."""
    img = Image.new("RGB", (256, 256), color="red")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf.read()


@pytest.fixture
def sample_image_pil():
    """Generate a valid PIL RGB image."""
    return Image.new("RGB", (256, 256), color="blue")


def make_random_vectors(n: int, dim: int, normalize: bool = True) -> np.ndarray:
    """Generate random L2-normalized vectors (simulates real embeddings)."""
    rng = np.random.RandomState(42)  # Reproducible
    vectors = rng.randn(n, dim).astype("float32")
    if normalize:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / norms
    return vectors


def make_metadata(n: int):
    """Generate dummy metadata for n products."""
    return [
        {"product_id": f"prod_{i}", "name": f"Product {i}", "category": "test"}
        for i in range(n)
    ]
