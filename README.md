# E-Commerce Visual Similarity Search

> Given a product image, retrieve the **top-K most visually similar products** from a jewellery e-commerce catalog using deep learning embeddings and vector search.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.135-green)
![FAISS](https://img.shields.io/badge/FAISS-1.13-orange)
![PyTorch](https://img.shields.io/badge/PyTorch-2.11-red)
[![CI](https://github.com/Daksh840/ecommerce-similarity-search/actions/workflows/ci.yml/badge.svg)](https://github.com/Daksh840/ecommerce-similarity-search/actions/workflows/ci.yml)

---

## Architecture

```
┌─────────────┐     ┌──────────────────────────────────────────────┐
│   Client    │     │              FastAPI Server                  │
│  (Browser/  │────▶│                                              │
│   curl)     │     │  ┌─────────┐  ┌───────────┐  ┌───────────┐  │
│             │◀────│  │ Prepro- │─▶│ Embedding │─▶│   FAISS   │  │
└─────────────┘     │  │ cessor  │  │ Extractor │  │  Search   │  │
                    │  │(Pillow) │  │(ResNet50) │  │(IndexFlat)│  │
                    │  └─────────┘  └───────────┘  └───────────┘  │
                    │                                              │
                    │  ┌─────────┐  ┌───────────┐                  │
                    │  │  Redis  │  │  Product   │                 │
                    │  │ (cache) │  │  Metadata  │                 │
                    │  └─────────┘  └───────────┘                  │
                    └──────────────────────────────────────────────┘
```

### Pipeline

1. **Preprocessing** — Validate image format, fix EXIF orientation, convert color modes (RGBA→RGB, CMYK→RGB), check dimensions
2. **Embedding** — Extract 2048-dim feature vector using ResNet50 (ImageNet V2 weights), L2-normalize
3. **Search** — Find nearest neighbors via FAISS `IndexFlatIP` (inner product = cosine similarity after normalization)
4. **Response** — Return top-K products with similarity scores and metadata

---

## Quick Start

### Local Setup

```bash
# Clone and enter project
cd ecommerce-similarity-search

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Copy dataset to data directory
cp ../jewellery_products_subset.json data/products.json

# Start the server
python -m app.main
```

Server starts at `http://localhost:8000`. API docs at `http://localhost:8000/docs`.

### Docker Setup

```bash
# Build and run with Docker Compose (includes Redis)
docker-compose up --build

# Or build manually
docker build -t similarity-search .
docker run -p 8000:8000 similarity-search
```

---

## API Reference

### Build Index (required first)

```bash
curl -X POST http://localhost:8000/api/v1/index/build
```

Response:
```json
{
  "status": "building",
  "products_indexed": 0,
  "index_stats": {
    "total_vectors": 0,
    "dimension": 2048,
    "index_type": "IndexFlatIP"
  }
}
```

### Search by Image Upload

```bash
curl -X POST http://localhost:8000/api/v1/search/image \
  -F "file=@ring.jpg" \
  -G -d "top_k=3"
```

Response:
```json
{
  "query": "image upload: ring.jpg",
  "results": [
    {
      "product": {
        "product_id": "9883604582685",
        "name": "Classic Charm Gold Chain",
        "category": "Chains"
      },
      "similarity_score": 0.9234,
      "index_id": 42
    }
  ],
  "total_indexed": 200
}
```

### Search by Image URL

```bash
curl -X POST "http://localhost:8000/api/v1/search/url?image_url=https://example.com/product.jpg&top_k=3"
```

### Search by Product ID

Find products similar to an already-indexed product (no image upload needed):

```bash
curl http://localhost:8000/api/v1/search/product/9883604582685?top_k=3
```

### Category Filtering

All search endpoints support optional `category` filtering:

```bash
# Find similar rings only
curl -X POST http://localhost:8000/api/v1/search/image \
  -F "file=@ring.jpg" \
  -G -d "top_k=5" -d "category=Rings"

# Filter URL search by category
curl -X POST "http://localhost:8000/api/v1/search/url?image_url=https://example.com/product.jpg&category=Chains"

# Filter product similarity by category
curl http://localhost:8000/api/v1/search/product/9883604582685?category=Rings
```

### Health Check

```bash
curl http://localhost:8000/health
```

### Index Statistics

```bash
curl http://localhost:8000/api/v1/index/stats
```

---

## ML Approach

### Why ResNet50?

| Factor | Choice | Rationale |
|--------|--------|-----------|
| **Model** | ResNet50 (V2 weights) | Strong transfer learning features, well-studied, fast inference |
| **Layer** | Pre-FC average pool | 2048-dim captures high-level visual semantics |
| **Normalization** | L2 normalize embeddings | Enables cosine similarity via inner product |
| **Index** | FAISS IndexFlatIP | Exact search, no approximation error, O(n) — suitable for <100k products |

### Why Not Other Approaches?

- **CLIP**: Better for text-image matching, but overkill for image-only similarity. Larger model = slower inference.
- **DINO/DINOv2**: Superior features for fine-grained similarity, but heavier compute. Good upgrade path.
- **IndexIVFPQ**: Approximate search — only needed at >100k scale. Adds complexity without benefit here.

### Similarity Metric

After L2 normalization, vectors are unit-length. The inner product of two unit vectors equals their cosine similarity:

```
cos(θ) = (a · b) / (||a|| × ||b||) = a · b    (when ||a|| = ||b|| = 1)
```

This is why we use `IndexFlatIP` instead of `IndexFlatL2`.

---

## Design Decisions & Trade-offs

| Decision | Trade-off | Rationale |
|----------|-----------|-----------|
| Background index build | Progress not visible via API | Prevents HTTP timeout for large datasets |
| Module-level singletons | Not DI-friendly | Avoids complexity for a single-service app |
| Redis caching (optional) | Extra dependency | Graceful degradation — works without Redis |
| JSON metadata (not pickle) | Slightly slower | Security: pickle allows arbitrary code execution |
| SSRF protection | Adds latency (DNS lookup) | Security: prevents internal network scanning |
| Center crop (not resize) | May crop edges | Preserves aspect ratio, focuses on center object |
| Post-search category filter | May return fewer results | Simpler than per-category indices; request more results and filter |
| run_in_executor for CPU ops | Small scheduling overhead | Prevents blocking the asyncio event loop |
| Single worker (Dockerfile) | No parallelism | Avoids duplicating ResNet50 in memory per worker |

---

## Project Structure

```
ecommerce-similarity-search/
├── app/
│   ├── main.py                  # FastAPI application & endpoints
│   ├── config.py                # Pydantic settings (env vars)
│   ├── core/
│   │   ├── exceptions.py        # Custom exception hierarchy
│   │   ├── logger.py            # Structured logging setup
│   │   └── security.py          # SSRF prevention, URL validation
│   ├── models/
│   │   └── schemas.py           # Pydantic request/response models
│   └── services/
│       ├── data_loader.py       # Dataset loading (JSON/CSV) & image download
│       ├── embedding_service.py # ResNet50 feature extraction
│       ├── preprocessing_service.py # Image validation & normalization
│       └── search_service.py    # FAISS index management & similarity search
├── tests/
│   ├── conftest.py              # Shared fixtures
│   ├── test_embedding.py        # Embedding extraction tests
│   ├── test_search_service.py   # FAISS search tests
│   ├── test_preprocessing_service.py # Image preprocessing tests
│   ├── test_data_loader.py      # Data loading tests
│   └── test_integration.py      # End-to-end API tests
├── scripts/
│   └── benchmark.py             # Search latency benchmarks
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   └── 02_embedding_analysis.ipynb
├── data/                        # Dataset & FAISS index (gitignored)
├── Dockerfile                   # Multi-stage production build
├── docker-compose.yml           # Full stack (API + Redis)
├── requirements.txt             # Pinned dependencies
├── .env.example                 # Environment variable template
└── README.md                    # This file
```

---

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test suites
pytest tests/test_search_service.py -v       # FAISS search (fast, no GPU)
pytest tests/test_preprocessing_service.py -v # Image preprocessing
pytest tests/test_embedding.py -v            # ResNet50 extraction (loads model)
pytest tests/test_data_loader.py -v          # Data loading
pytest tests/test_integration.py -v          # Full API integration

# Run with coverage
pytest tests/ --cov=app --cov-report=term-missing
```

Or use the Makefile shortcuts:

```bash
make test              # All tests
make test-unit         # Unit tests only (fast, no model loading)
make test-integration  # API integration tests
make test-coverage     # Tests + coverage report
```

---

## Configuration

All settings can be overridden via environment variables or a `.env` file.
See [`.env.example`](.env.example) for the full list.

| Variable | Default | Description |
|----------|---------|-------------|
| `EMBEDDING_DIMENSION` | 2048 | ResNet50 output dimension |
| `INDEX_PATH` | `data/faiss_index` | FAISS index storage path |
| `DATA_PATH` | `data/products.json` | Product dataset file |
| `REDIS_URL` | `redis://localhost:6379` | Redis URL (optional) |
| `DEFAULT_TOP_K` | 5 | Default number of results |
| `MAX_TOP_K` | 20 | Maximum allowed top_k |
| `LOG_LEVEL` | INFO | Logging verbosity |

---

## Performance

Benchmarked on CPU (Intel i7, 16GB RAM):

| Metric | Value |
|--------|-------|
| Embedding extraction | ~60-100ms per image |
| FAISS search (200 vectors) | <1ms |
| FAISS search (10k vectors) | ~4.77ms |
| End-to-end API latency | ~70-120ms |
| Index memory (200 products) | ~1.6 MB |

---

## License

MIT
