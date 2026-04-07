"""
FAISS-based similarity search with persistence.

Handles:
- Index building (IndexFlatIP for exact cosine similarity)
- Incremental updates (add new products without rebuilding)
- Metadata mapping (index_id → product info)
- GPU acceleration (automatic if available)
- Save/load with atomic writes for crash safety
- Vector removal via index rebuild

Index Type Reference:
┌──────────────┬───────────┬────────┬──────────┬───────────────────┐
│ Index Type   │ Time      │ Memory │ Accuracy │ Use Case          │
├──────────────┼───────────┼────────┼──────────┼───────────────────┤
│ IndexFlatIP  │ O(n)      │ 100%   │ Exact    │ < 100k vectors    │
│ IndexIVFFlat │ O(√n)     │ 100%   │ Approx   │ 100k - 10M        │
│ IndexIVFPQ   │ O(√n)     │ 5-10%  │ Approx   │ > 10M (mobile)    │
└──────────────┴───────────┴────────┴──────────┴───────────────────┘

We use IndexFlatIP here because:
- After L2 normalization, Inner Product == Cosine Similarity
- No approximation error, simple implementation
- Suitable for our catalog size (< 100k products)
"""

import json
import logging
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Optional

import faiss
import numpy as np

logger = logging.getLogger(__name__)


class SimilaritySearch:
    """
    FAISS-based vector similarity search engine.

    Thread-safety: FAISS search is thread-safe for reads.
    Writes (add_vectors, remove_vectors) are NOT thread-safe —
    callers must synchronize externally.
    """

    def __init__(self, dimension: int = 2048, index_path: Optional[str] = None):
        self.dimension = dimension
        self.index_path = Path(index_path) if index_path else None
        self.index = None
        self.metadata: Dict[int, Dict] = {}  # index_id → product_info
        self.is_trained = False

        if self.index_path and self.index_path.exists():
            self.load()
        else:
            self._create_index()

    def _create_index(self):
        """
        Initialize FAISS index.

        IndexFlatIP: Exact search using Inner Product.
        After L2 normalization (done in EmbeddingExtractor),
        Inner Product equals Cosine Similarity.
        """
        self.index = faiss.IndexFlatIP(self.dimension)

        # GPU acceleration (~10x speedup on search)
        if faiss.get_num_gpus() > 0:
            logger.info("GPU detected — moving index to GPU for acceleration")
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)

        self.is_trained = True
        logger.info("Created IndexFlatIP with dimension=%d", self.dimension)

    def add_vectors(self, embeddings: np.ndarray, metadata_list: List[Dict]):
        """
        Add vectors to index with associated metadata.

        Args:
            embeddings: (n, dimension) numpy array — must be L2-normalized
            metadata_list: List of dicts with product_id, name, category, etc.

        Raises:
            ValueError: If index not initialized, shape mismatch, or wrong dimension
        """
        if not self.is_trained:
            raise ValueError("Index not initialized. Call _create_index() first.")

        if len(embeddings) != len(metadata_list):
            raise ValueError(
                f"Shape mismatch: {len(embeddings)} embeddings vs "
                f"{len(metadata_list)} metadata entries"
            )

        # Validate embedding dimension
        if embeddings.ndim != 2 or embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Expected embeddings of shape (n, {self.dimension}), "
                f"got {embeddings.shape}"
            )

        # FAISS requires float32 and C-contiguous memory layout.
        # Non-contiguous arrays cause silent errors or segfaults.
        embeddings = np.ascontiguousarray(embeddings.astype("float32"))

        # Track starting position for metadata mapping
        start_idx = self.index.ntotal

        # Add vectors to FAISS index
        self.index.add(embeddings)

        # Map FAISS internal indices to product metadata
        for i, meta in enumerate(metadata_list):
            self.metadata[start_idx + i] = meta

        logger.info(
            "Added %d vectors. Total index size: %d",
            len(embeddings),
            self.index.ntotal,
        )

    def search(self, query_vector: np.ndarray, k: int = 5) -> List[Dict]:
        """
        Search for top-k most similar products.

        Args:
            query_vector: (dimension,) or (1, dimension) L2-normalized embedding
            k: Number of results to return

        Returns:
            List of dicts: [{product, similarity_score, index_id}]
            Scores are cosine similarity values (after L2 normalization).

        Raises:
            ValueError: If query vector has wrong dimension
        """
        if self.index.ntotal == 0:
            logger.warning("Search called on empty index")
            return []

        # Handle 1D input — reshape to (1, dim) for FAISS
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)

        # Validate query dimension
        if query_vector.shape[1] != self.dimension:
            raise ValueError(
                f"Query dimension {query_vector.shape[1]} != "
                f"index dimension {self.dimension}"
            )

        query_vector = np.ascontiguousarray(query_vector.astype("float32"))

        # FAISS search returns (scores, indices) arrays of shape (n_queries, k)
        scores, indices = self.index.search(query_vector, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # FAISS returns -1 for empty/unfilled slots
                continue

            meta = self.metadata.get(int(idx), {})
            results.append({
                "product": meta,
                "similarity_score": float(score),
                "index_id": int(idx),
            })

        return results

    def search_batch(self, query_vectors: np.ndarray, k: int = 5) -> List[List[Dict]]:
        """
        Batch search for multiple query vectors at once.
        More efficient than calling search() in a loop.

        Args:
            query_vectors: (n, dimension) L2-normalized embeddings
            k: Number of results per query

        Returns:
            List of result lists, one per query vector.
        """
        if self.index.ntotal == 0:
            logger.warning("Batch search called on empty index")
            return [[] for _ in range(len(query_vectors))]

        # Validate dimension
        if query_vectors.ndim != 2 or query_vectors.shape[1] != self.dimension:
            raise ValueError(
                f"Expected queries of shape (n, {self.dimension}), "
                f"got {query_vectors.shape}"
            )

        query_vectors = np.ascontiguousarray(query_vectors.astype("float32"))
        scores_batch, indices_batch = self.index.search(query_vectors, k)

        all_results = []
        for scores, indices in zip(scores_batch, indices_batch):
            results = []
            for score, idx in zip(scores, indices):
                if idx == -1:
                    continue
                meta = self.metadata.get(int(idx), {})
                results.append({
                    "product": meta,
                    "similarity_score": float(score),
                    "index_id": int(idx),
                })
            all_results.append(results)

        return all_results

    def save(self, path: str = None):
        """
        Persist index and metadata to disk with atomic writes.

        Saves two files:
        - faiss_index.bin: The FAISS index (binary)
        - metadata.json: Product metadata mapping (JSON, not pickle)
        - config.json: Index configuration for reconstruction

        Atomic write: writes to temp dir first, then renames.
        This prevents corruption if the process is killed mid-write.

        Note: GPU indices must be converted to CPU before saving.
        """
        save_path = Path(path) if path else self.index_path
        if not save_path:
            raise ValueError("No save path specified (pass path or set index_path)")

        save_path.mkdir(parents=True, exist_ok=True)

        # GPU index cannot be serialized directly — convert to CPU first
        if faiss.get_num_gpus() > 0 and hasattr(self.index, "copyToCpu"):
            index_cpu = faiss.index_gpu_to_cpu(self.index)
        else:
            index_cpu = self.index

        # Atomic write: write to temp, then rename
        tmp_dir = Path(tempfile.mkdtemp(dir=save_path.parent))
        try:
            # Save FAISS index binary
            faiss.write_index(index_cpu, str(tmp_dir / "faiss_index.bin"))

            # Save metadata as JSON (safer than pickle — no code execution risk)
            # Convert int keys to strings for JSON compatibility
            json_metadata = {str(k): v for k, v in self.metadata.items()}
            with open(tmp_dir / "metadata.json", "w", encoding="utf-8") as f:
                json.dump(json_metadata, f, indent=2)

            # Save config for reproducibility
            config = {
                "dimension": self.dimension,
                "ntotal": int(self.index.ntotal),
                "index_type": "IndexFlatIP",
            }
            with open(tmp_dir / "config.json", "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2)

            # Atomic swap: move files from temp to final location
            for fname in ["faiss_index.bin", "metadata.json", "config.json"]:
                src = tmp_dir / fname
                dst = save_path / fname
                if dst.exists():
                    dst.unlink()
                shutil.move(str(src), str(dst))

        finally:
            # Clean up temp dir
            shutil.rmtree(tmp_dir, ignore_errors=True)

        logger.info(
            "Index saved to %s (%d vectors, %dd)",
            save_path,
            self.index.ntotal,
            self.dimension,
        )

    def load(self, path: str = None):
        """
        Load index and metadata from disk.
        Automatically moves to GPU if available.
        Supports both JSON (preferred) and legacy pickle metadata.
        """
        load_path = Path(path) if path else self.index_path
        if not load_path or not load_path.exists():
            raise FileNotFoundError(f"Index path does not exist: {load_path}")

        # Load FAISS index
        index_file = load_path / "faiss_index.bin"
        if not index_file.exists():
            raise FileNotFoundError(f"FAISS index file not found: {index_file}")
        self.index = faiss.read_index(str(index_file))

        # Load metadata — prefer JSON, fallback to pickle for backward compat
        json_meta = load_path / "metadata.json"
        pkl_meta = load_path / "metadata.pkl"

        if json_meta.exists():
            with open(json_meta, "r", encoding="utf-8") as f:
                raw = json.load(f)
            # Convert string keys back to int
            self.metadata = {int(k): v for k, v in raw.items()}
        elif pkl_meta.exists():
            import pickle
            logger.warning("Loading legacy pickle metadata — consider re-saving as JSON")
            with open(pkl_meta, "rb") as f:
                self.metadata = pickle.load(f)
        else:
            logger.warning("No metadata file found — index will have no product info")
            self.metadata = {}

        # Load config for validation
        config_json = load_path / "config.json"
        config_pkl = load_path / "config.pkl"

        if config_json.exists():
            with open(config_json, "r", encoding="utf-8") as f:
                config = json.load(f)
            self.dimension = config.get("dimension", self.dimension)
        elif config_pkl.exists():
            import pickle
            with open(config_pkl, "rb") as f:
                config = pickle.load(f)
            self.dimension = config.get("dimension", self.dimension)

        # Move to GPU if available
        if faiss.get_num_gpus() > 0:
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)

        self.is_trained = True
        logger.info(
            "Loaded index from %s (%d vectors, %dd)",
            load_path,
            self.index.ntotal,
            self.dimension,
        )

    def get_stats(self) -> Dict:
        """Return index statistics for monitoring/debugging."""
        return {
            "total_vectors": int(self.index.ntotal) if self.index else 0,
            "dimension": self.dimension,
            "index_type": "IndexFlatIP",
            "metadata_entries": len(self.metadata),
            "is_trained": self.is_trained,
            "gpu_enabled": faiss.get_num_gpus() > 0,
            "memory_estimate_mb": round(
                (self.index.ntotal * self.dimension * 4) / (1024 ** 2), 2
            )
            if self.index
            else 0,
        }

    def remove_vectors(self, index_ids: List[int]):
        """
        Remove vectors by rebuilding the index without them.

        Note: FAISS IndexFlatIP doesn't support direct deletion.
        This uses reconstruct_n() to extract all vectors, filters out
        the ones to remove, and rebuilds the index.

        For frequent deletions at scale, consider using IndexIDMap.
        """
        if not index_ids:
            return

        ids_to_remove = set(index_ids)
        n_total = self.index.ntotal

        # Get CPU index for reconstruction
        if faiss.get_num_gpus() > 0 and hasattr(self.index, "copyToCpu"):
            cpu_index = faiss.index_gpu_to_cpu(self.index)
        else:
            cpu_index = self.index

        # Extract ALL vectors at once using reconstruct_n (single FAISS call)
        # Much faster than per-vector reconstruct() in a Python loop
        all_vectors = cpu_index.reconstruct_n(0, n_total)

        # Filter: keep only vectors NOT in the removal set
        keep_mask = np.array([i not in ids_to_remove for i in range(n_total)])
        keep_embeddings = all_vectors[keep_mask]
        keep_metadata = [
            self.metadata.get(i, {})
            for i in range(n_total)
            if i not in ids_to_remove
        ]

        # Rebuild index from scratch
        self._create_index()
        self.metadata = {}

        if len(keep_embeddings) > 0:
            self.add_vectors(keep_embeddings, keep_metadata)

        logger.info(
            "Removed %d vectors. New total: %d",
            len(ids_to_remove),
            self.index.ntotal,
        )