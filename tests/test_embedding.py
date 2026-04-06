"""
Tests for the embedding extraction service.

Covers:
- Output dimensions (2048-d)
- L2 normalization (unit vectors)
- Determinism (same input → same output)
- Discriminability (different inputs → different embeddings)
- Batch extraction (valid indices tracking)
"""

import numpy as np
import pytest
from PIL import Image

from app.services.embedding_service import EmbeddingExtractor


@pytest.fixture(scope="module")
def extractor():
    """
    Shared extractor instance for all tests in this module.
    scope='module' avoids loading ResNet50 multiple times.
    """
    return EmbeddingExtractor()


class TestSingleExtraction:
    def test_output_dimensions(self, extractor):
        """Embedding should be 2048-dimensional."""
        img = Image.new("RGB", (224, 224), color="red")
        emb = extractor.extract_single(img)
        assert emb.shape == (2048,)

    def test_l2_normalized(self, extractor):
        """Embedding should be unit-length (L2 norm ≈ 1.0)."""
        img = Image.new("RGB", (224, 224), color="red")
        emb = extractor.extract_single(img)
        assert np.isclose(np.linalg.norm(emb), 1.0, atol=1e-5)

    def test_float32_output(self, extractor):
        """FAISS requires float32."""
        img = Image.new("RGB", (224, 224), color="red")
        emb = extractor.extract_single(img)
        assert emb.dtype == np.float32

    def test_deterministic(self, extractor):
        """Same image → same embedding (no randomness in inference)."""
        img = Image.new("RGB", (224, 224), color="blue")
        emb1 = extractor.extract_single(img)
        emb2 = extractor.extract_single(img)
        assert np.isclose(np.dot(emb1, emb2), 1.0, atol=1e-5)

    def test_different_images_produce_different_embeddings(self, extractor):
        """
        Structurally different images should produce distinguishable embeddings.

        Note: ResNet50 interprets uniform random noise as statistically similar
        textures, so we use solid colors from different color domains instead.
        Red vs green activates different early-layer filters.
        """
        img_red = Image.new("RGB", (224, 224), color="red")
        img_green = Image.new("RGB", (224, 224), color="green")
        emb1 = extractor.extract_single(img_red)
        emb2 = extractor.extract_single(img_green)

        similarity = np.dot(emb1, emb2)
        assert similarity < 0.95, (
            f"Expected distinct embeddings for red vs green, "
            f"got similarity={similarity:.4f}"
        )


class TestBatchExtraction:
    def test_batch_returns_tuple(self, extractor, tmp_path):
        """extract_batch should return (embeddings, valid_indices) tuple."""
        # Create test images
        paths = []
        for i in range(3):
            path = tmp_path / f"test_{i}.jpg"
            img = Image.new("RGB", (224, 224), color=(i * 80, 0, 0))
            img.save(str(path))
            paths.append(str(path))

        embeddings, valid_indices = extractor.extract_batch(paths, batch_size=2)

        assert embeddings.shape == (3, 2048)
        assert valid_indices == [0, 1, 2]

    def test_batch_skips_invalid_images(self, extractor, tmp_path):
        """Invalid images should be skipped, not zero-filled."""
        # Create 2 valid + 1 invalid
        valid1 = tmp_path / "valid1.jpg"
        Image.new("RGB", (224, 224), color="red").save(str(valid1))

        invalid = tmp_path / "invalid.jpg"
        invalid.write_text("not an image")

        valid2 = tmp_path / "valid2.jpg"
        Image.new("RGB", (224, 224), color="blue").save(str(valid2))

        embeddings, valid_indices = extractor.extract_batch(
            [str(valid1), str(invalid), str(valid2)], batch_size=10
        )

        assert embeddings.shape[0] == 2  # Only 2 valid
        assert 1 not in valid_indices  # Index 1 (invalid) skipped
        assert all(np.isclose(np.linalg.norm(e), 1.0, atol=1e-5) for e in embeddings)

    def test_batch_all_invalid(self, extractor, tmp_path):
        """All-invalid batch should return empty arrays."""
        invalid = tmp_path / "bad.jpg"
        invalid.write_text("garbage")

        embeddings, valid_indices = extractor.extract_batch([str(invalid)])

        assert embeddings.shape == (0, 2048)
        assert valid_indices == []