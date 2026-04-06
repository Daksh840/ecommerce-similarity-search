"""
Tests for the similarity search service (FAISS).

Covers:
- Index creation and vector addition
- Search accuracy and ranking
- Dimension validation
- Persistence (save/load round-trip with JSON metadata)
- Edge cases (empty index, dimension mismatch, etc.)
- Batch search
- Vector removal
- Stats reporting

Run: pytest tests/test_search_service.py -v
"""

import numpy as np
import pytest
import tempfile
import shutil
from pathlib import Path

from app.services.search_service import SimilaritySearch


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def dimension():
    """Use small dimension for fast tests."""
    return 128


@pytest.fixture
def search_engine(dimension):
    """Fresh SimilaritySearch instance (no persistence)."""
    return SimilaritySearch(dimension=dimension)


@pytest.fixture
def tmp_index_dir():
    """Temporary directory for index persistence tests."""
    path = tempfile.mkdtemp()
    yield path
    shutil.rmtree(path, ignore_errors=True)


def make_random_vectors(n: int, dim: int, normalize: bool = True) -> np.ndarray:
    """Generate random L2-normalized vectors (simulates real embeddings)."""
    vectors = np.random.randn(n, dim).astype("float32")
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


# ── Index Creation ───────────────────────────────────────────────────────────

class TestIndexCreation:
    def test_creates_empty_index(self, search_engine, dimension):
        assert search_engine.index is not None
        assert search_engine.index.ntotal == 0
        assert search_engine.dimension == dimension
        assert search_engine.is_trained is True

    def test_default_dimension(self):
        engine = SimilaritySearch()
        assert engine.dimension == 2048


# ── Adding Vectors ───────────────────────────────────────────────────────────

class TestAddVectors:
    def test_add_single_batch(self, search_engine, dimension):
        n = 10
        vectors = make_random_vectors(n, dimension)
        metadata = make_metadata(n)

        search_engine.add_vectors(vectors, metadata)

        assert search_engine.index.ntotal == n
        assert len(search_engine.metadata) == n

    def test_add_incremental(self, search_engine, dimension):
        """Verify incremental addition (two separate batches)."""
        batch1 = make_random_vectors(5, dimension)
        batch2 = make_random_vectors(3, dimension)

        search_engine.add_vectors(batch1, make_metadata(5))
        search_engine.add_vectors(batch2, make_metadata(3))

        assert search_engine.index.ntotal == 8
        assert len(search_engine.metadata) == 8

    def test_metadata_mapping_correct(self, search_engine, dimension):
        """Verify metadata maps to correct index positions."""
        vectors = make_random_vectors(3, dimension)
        meta = [
            {"product_id": "A"},
            {"product_id": "B"},
            {"product_id": "C"},
        ]

        search_engine.add_vectors(vectors, meta)

        assert search_engine.metadata[0]["product_id"] == "A"
        assert search_engine.metadata[1]["product_id"] == "B"
        assert search_engine.metadata[2]["product_id"] == "C"

    def test_shape_mismatch_raises(self, search_engine, dimension):
        """Different count of vectors vs metadata should raise."""
        vectors = make_random_vectors(5, dimension)
        metadata = make_metadata(3)  # Mismatch!

        with pytest.raises(ValueError, match="Shape mismatch"):
            search_engine.add_vectors(vectors, metadata)

    def test_wrong_dimension_raises(self, search_engine, dimension):
        """Wrong embedding dimension should raise ValueError."""
        wrong_dim = dimension + 10
        vectors = make_random_vectors(5, wrong_dim)
        metadata = make_metadata(5)

        with pytest.raises(ValueError, match="Expected embeddings"):
            search_engine.add_vectors(vectors, metadata)


# ── Search ───────────────────────────────────────────────────────────────────

class TestSearch:
    def test_search_returns_results(self, search_engine, dimension):
        vectors = make_random_vectors(20, dimension)
        search_engine.add_vectors(vectors, make_metadata(20))

        query = make_random_vectors(1, dimension).squeeze()
        results = search_engine.search(query, k=5)

        assert len(results) == 5
        assert all("product" in r for r in results)
        assert all("similarity_score" in r for r in results)
        assert all("index_id" in r for r in results)

    def test_search_self_is_top_result(self, search_engine, dimension):
        """Searching for an indexed vector should return itself as top match."""
        vectors = make_random_vectors(10, dimension)
        search_engine.add_vectors(vectors, make_metadata(10))

        # Search for the exact 5th vector
        results = search_engine.search(vectors[4], k=1)

        assert len(results) == 1
        assert results[0]["index_id"] == 4
        assert results[0]["similarity_score"] == pytest.approx(1.0, abs=1e-5)

    def test_results_sorted_by_score(self, search_engine, dimension):
        vectors = make_random_vectors(50, dimension)
        search_engine.add_vectors(vectors, make_metadata(50))

        query = make_random_vectors(1, dimension).squeeze()
        results = search_engine.search(query, k=10)

        scores = [r["similarity_score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_search_empty_index(self, search_engine, dimension):
        """Searching an empty index should return empty list, not crash."""
        query = make_random_vectors(1, dimension).squeeze()
        results = search_engine.search(query, k=5)
        assert results == []

    def test_k_larger_than_index(self, search_engine, dimension):
        """Requesting more results than indexed vectors should still work."""
        vectors = make_random_vectors(3, dimension)
        search_engine.add_vectors(vectors, make_metadata(3))

        results = search_engine.search(vectors[0], k=10)
        assert len(results) == 3  # Only 3 vectors exist

    def test_2d_query_input(self, search_engine, dimension):
        """Query shaped (1, dim) should work same as (dim,)."""
        vectors = make_random_vectors(10, dimension)
        search_engine.add_vectors(vectors, make_metadata(10))

        query_1d = vectors[0]
        query_2d = vectors[0].reshape(1, -1)

        results_1d = search_engine.search(query_1d, k=3)
        results_2d = search_engine.search(query_2d, k=3)

        assert len(results_1d) == len(results_2d)
        assert results_1d[0]["index_id"] == results_2d[0]["index_id"]

    def test_wrong_query_dimension_raises(self, search_engine, dimension):
        """Query with wrong dimension should raise ValueError."""
        vectors = make_random_vectors(10, dimension)
        search_engine.add_vectors(vectors, make_metadata(10))

        wrong_query = np.random.randn(dimension + 5).astype("float32")
        with pytest.raises(ValueError, match="Query dimension"):
            search_engine.search(wrong_query, k=3)


# ── Batch Search ─────────────────────────────────────────────────────────────

class TestBatchSearch:
    def test_batch_search(self, search_engine, dimension):
        vectors = make_random_vectors(20, dimension)
        search_engine.add_vectors(vectors, make_metadata(20))

        queries = make_random_vectors(3, dimension)
        results = search_engine.search_batch(queries, k=5)

        assert len(results) == 3
        assert all(len(r) == 5 for r in results)

    def test_batch_search_empty_index(self, search_engine, dimension):
        queries = make_random_vectors(2, dimension)
        results = search_engine.search_batch(queries, k=5)
        assert len(results) == 2
        assert all(r == [] for r in results)

    def test_batch_wrong_dimension_raises(self, search_engine, dimension):
        vectors = make_random_vectors(10, dimension)
        search_engine.add_vectors(vectors, make_metadata(10))

        wrong_queries = np.random.randn(3, dimension + 5).astype("float32")
        with pytest.raises(ValueError, match="Expected queries"):
            search_engine.search_batch(wrong_queries, k=3)


# ── Persistence ──────────────────────────────────────────────────────────────

class TestPersistence:
    def test_save_and_load(self, dimension, tmp_index_dir):
        """Full round-trip: create → add → save → load → search."""
        # Create and populate
        engine1 = SimilaritySearch(dimension=dimension)
        vectors = make_random_vectors(15, dimension)
        metadata = make_metadata(15)
        engine1.add_vectors(vectors, metadata)
        engine1.save(tmp_index_dir)

        # Load into fresh instance
        engine2 = SimilaritySearch(dimension=dimension, index_path=tmp_index_dir)

        assert engine2.index.ntotal == 15
        assert len(engine2.metadata) == 15
        assert engine2.metadata[0]["product_id"] == "prod_0"

        # Verify search still works after reload
        results = engine2.search(vectors[0], k=1)
        assert results[0]["index_id"] == 0
        assert results[0]["similarity_score"] == pytest.approx(1.0, abs=1e-5)

    def test_save_creates_directory(self, dimension):
        """Save should create the directory if it doesn't exist."""
        engine = SimilaritySearch(dimension=dimension)
        vectors = make_random_vectors(5, dimension)
        engine.add_vectors(vectors, make_metadata(5))

        save_path = Path(tempfile.mkdtemp()) / "nested" / "index"
        try:
            engine.save(str(save_path))
            assert (save_path / "faiss_index.bin").exists()
            assert (save_path / "metadata.json").exists()
            assert (save_path / "config.json").exists()
        finally:
            shutil.rmtree(save_path.parent, ignore_errors=True)

    def test_save_no_path_raises(self, dimension):
        engine = SimilaritySearch(dimension=dimension)
        with pytest.raises(ValueError, match="No save path"):
            engine.save()

    def test_metadata_is_json_not_pickle(self, dimension, tmp_index_dir):
        """Verify metadata is saved as JSON (not pickle) for security."""
        engine = SimilaritySearch(dimension=dimension)
        vectors = make_random_vectors(3, dimension)
        engine.add_vectors(vectors, make_metadata(3))
        engine.save(tmp_index_dir)

        meta_path = Path(tmp_index_dir) / "metadata.json"
        assert meta_path.exists()

        import json
        with open(meta_path) as f:
            data = json.load(f)
        assert "0" in data  # Keys are strings in JSON
        assert data["0"]["product_id"] == "prod_0"


# ── Vector Removal ───────────────────────────────────────────────────────────

class TestRemoveVectors:
    def test_remove_single_vector(self, search_engine, dimension):
        vectors = make_random_vectors(5, dimension)
        search_engine.add_vectors(vectors, make_metadata(5))

        search_engine.remove_vectors([2])

        assert search_engine.index.ntotal == 4
        assert len(search_engine.metadata) == 4

    def test_remove_multiple_vectors(self, search_engine, dimension):
        vectors = make_random_vectors(10, dimension)
        search_engine.add_vectors(vectors, make_metadata(10))

        search_engine.remove_vectors([0, 3, 7])

        assert search_engine.index.ntotal == 7
        assert len(search_engine.metadata) == 7

    def test_remove_empty_list(self, search_engine, dimension):
        vectors = make_random_vectors(5, dimension)
        search_engine.add_vectors(vectors, make_metadata(5))

        search_engine.remove_vectors([])
        assert search_engine.index.ntotal == 5


# ── Stats ────────────────────────────────────────────────────────────────────

class TestStats:
    def test_stats_empty(self, search_engine):
        stats = search_engine.get_stats()
        assert stats["total_vectors"] == 0
        assert stats["index_type"] == "IndexFlatIP"
        assert stats["memory_estimate_mb"] == 0

    def test_stats_after_add(self, search_engine, dimension):
        vectors = make_random_vectors(100, dimension)
        search_engine.add_vectors(vectors, make_metadata(100))

        stats = search_engine.get_stats()
        assert stats["total_vectors"] == 100
        assert stats["dimension"] == dimension
        assert stats["metadata_entries"] == 100
        assert stats["memory_estimate_mb"] > 0
