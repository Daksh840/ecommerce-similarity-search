"""
FastAPI application — the API layer for E-Commerce Similarity Search.

Architecture:
    Client → FastAPI (validation, routing) → Services (business logic) → FAISS (search)

Endpoints:
    POST /api/v1/search/image        → Upload image, find similar products
    POST /api/v1/search/url          → Provide image URL, find similar products
    GET  /api/v1/search/product/{id} → Find similar products by product ID
    POST /api/v1/index/build         → Build index from dataset
    GET  /api/v1/index/stats         → Index statistics
    GET  /health                     → Health check

Design Decisions:
- Lifespan events (startup/shutdown) for resource management
- Domain exceptions mapped to HTTP status codes at the API boundary
- Response models for automatic documentation
- Redis caching on URL endpoint (optional, graceful degradation)
- SSRF protection on URL endpoint
- Background task support for index building
- Category filtering on all search endpoints (post-search filter)
- CPU-bound ops (preprocessing, embedding) run in executor to avoid blocking asyncio
"""

import asyncio
import hashlib
import logging
import threading
from contextlib import asynccontextmanager
from typing import Optional

import faiss
import httpx
from fastapi import FastAPI, File, UploadFile, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.core.exceptions import (
    ImageValidationError,
    ImageProcessingError,
    EmbeddingExtractionError,
    IndexNotFoundError,
    IndexEmptyError,
)
from app.core.logger import setup_logging
from app.core.security import validate_url, check_content_length
from app.models.schemas import (
    SearchResponse,
    SearchResult,
    ProductInfo,
    IndexStatsResponse,
    IndexBuildResponse,
    HealthResponse,
    ErrorResponse,
)
from app.services.data_loader import ProductDataset
from app.services.embedding_service import EmbeddingExtractor
from app.services.preprocessing_service import ImagePreprocessor
from app.services.search_service import SimilaritySearch

logger = logging.getLogger(__name__)
settings = get_settings()


# ---------------------------------------------------------------------------
# Service singletons (initialized at startup, shared across requests)
# ---------------------------------------------------------------------------
embedding_extractor: Optional[EmbeddingExtractor] = None
similarity_search: Optional[SimilaritySearch] = None
image_preprocessor: Optional[ImagePreprocessor] = None

# Redis is optional — import with guard
try:
    import redis
    _redis_available = True
except ImportError:
    redis = None  # type: ignore
    _redis_available = False

redis_client: Optional["redis.Redis"] = None  # type: ignore

# Lock to prevent concurrent index builds
_build_lock = threading.Lock()
_build_in_progress = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup/shutdown lifecycle.

    Why lifespan instead of @app.on_event?
    - @app.on_event is deprecated in modern FastAPI
    - Lifespan provides a clean context manager pattern
    - Resources allocated before yield, cleaned up after
    """
    global embedding_extractor, similarity_search, image_preprocessor, redis_client

    # --- Startup ---
    setup_logging()
    logger.info("Starting %s v%s", settings.app_name, settings.app_version)

    # Initialize Redis (optional — graceful degradation)
    if _redis_available:
        try:
            redis_client = redis.Redis.from_url(
                settings.redis_url, decode_responses=True
            )
            redis_client.ping()
            logger.info("Redis cache connected via %s", settings.redis_url)
        except Exception as e:
            logger.warning("Redis not available, caching disabled: %s", e)
            redis_client = None
    else:
        logger.info("Redis package not installed — caching disabled")

    # Initialize services
    image_preprocessor = ImagePreprocessor()
    logger.info("ImagePreprocessor initialized")

    embedding_extractor = EmbeddingExtractor()
    logger.info("EmbeddingExtractor initialized")

    similarity_search = SimilaritySearch(
        dimension=settings.embedding_dimension,
        index_path=settings.index_path,
    )
    logger.info(
        "SimilaritySearch initialized (%d vectors loaded)",
        similarity_search.index.ntotal,
    )

    yield  # ← App is running and serving requests

    # --- Shutdown ---
    logger.info("Shutting down — saving index if modified")
    if similarity_search and similarity_search.index.ntotal > 0:
        try:
            similarity_search.save()
        except Exception as e:
            logger.error("Failed to save index on shutdown: %s", e)

    logger.info("Shutdown complete")


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "Visual similarity search for e-commerce products. "
        "Upload a product image and find visually similar items "
        "using ResNet50 embeddings + FAISS vector search."
    ),
    lifespan=lifespan,
)

# CORS — allow frontend access during development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Lock down in production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Exception handlers — map domain exceptions to HTTP responses
# ---------------------------------------------------------------------------
@app.exception_handler(ImageValidationError)
async def image_validation_handler(request, exc: ImageValidationError):
    return JSONResponse(
        status_code=422,
        content={"error": exc.message, "detail": exc.detail},
    )


@app.exception_handler(ImageProcessingError)
async def image_processing_handler(request, exc: ImageProcessingError):
    return JSONResponse(
        status_code=500,
        content={"error": exc.message, "detail": exc.detail},
    )


@app.exception_handler(EmbeddingExtractionError)
async def embedding_extraction_handler(request, exc: EmbeddingExtractionError):
    return JSONResponse(
        status_code=500,
        content={"error": exc.message, "detail": exc.detail},
    )


@app.exception_handler(IndexNotFoundError)
async def index_not_found_handler(request, exc: IndexNotFoundError):
    return JSONResponse(
        status_code=404,
        content={"error": exc.message, "detail": exc.detail},
    )


@app.exception_handler(IndexEmptyError)
async def index_empty_handler(request, exc: IndexEmptyError):
    return JSONResponse(
        status_code=503,
        content={"error": exc.message, "detail": exc.detail},
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Health check — used by load balancers and monitoring."""
    return HealthResponse(
        status="healthy",
        version=settings.app_version,
        index_loaded=similarity_search is not None and similarity_search.is_trained,
        total_products=similarity_search.index.ntotal if similarity_search else 0,
    )


@app.post(
    "/api/v1/search/image",
    response_model=SearchResponse,
    responses={422: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    tags=["Search"],
)
async def search_by_image(
    file: UploadFile = File(..., description="Product image to search for"),
    top_k: int = Query(
        default=settings.default_top_k,
        ge=1,
        le=settings.max_top_k,
        description="Number of similar products to return",
    ),
    category: str = Query(
        default=None,
        description="Filter results by product category (optional)",
    ),
):
    """
    Upload a product image and find visually similar products.

    Pipeline:
    1. Validate MIME type & content
    2. Preprocess uploaded image (CPU-bound → run_in_executor)
    3. Extract 2048-dim embedding via ResNet50 (CPU-bound → run_in_executor)
    4. Search FAISS index for nearest neighbors
    5. Optionally filter by category
    6. Return ranked results with cosine similarity scores
    """
    # MIME type validation (early rejection before reading full body)
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=422,
            detail=f"Expected image file, got {file.content_type}",
        )

    if not similarity_search or similarity_search.index.ntotal == 0:
        raise HTTPException(
            status_code=503,
            detail="Index is empty. Build the index first via POST /api/v1/index/build",
        )

    # 1. Read upload (async I/O)
    image_bytes = await file.read()
    logger.info("Received image: %s (%d bytes)", file.filename, len(image_bytes))

    # 2-3. Preprocessing + embedding extraction are CPU-bound.
    # Wrap in run_in_executor to avoid blocking the asyncio event loop.
    loop = asyncio.get_running_loop()
    image = await loop.run_in_executor(
        None, image_preprocessor.validate_and_process, image_bytes
    )
    embedding = await loop.run_in_executor(
        None, embedding_extractor.extract_single, image
    )

    # 4. Search
    # Request extra results if filtering by category (to compensate for filtered-out items)
    search_k = top_k * 3 if category else top_k
    results = similarity_search.search(embedding, k=search_k)

    # 5. Category filter (post-search)
    if category:
        results = [
            r for r in results
            if r["product"].get("category", "").lower() == category.lower()
        ][:top_k]

    # 6. Format response
    return SearchResponse(
        query=f"image upload: {file.filename}",
        results=[
            SearchResult(
                product=ProductInfo(**r["product"]),
                similarity_score=round(r["similarity_score"], 4),
                index_id=r["index_id"],
            )
            for r in results
        ],
        total_indexed=similarity_search.index.ntotal,
    )


@app.post(
    "/api/v1/search/url",
    response_model=SearchResponse,
    responses={422: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    tags=["Search"],
)
async def search_by_url(
    image_url: str = Query(..., description="URL of the product image"),
    top_k: int = Query(
        default=settings.default_top_k,
        ge=1,
        le=settings.max_top_k,
        description="Number of similar products to return",
    ),
    category: str = Query(
        default=None,
        description="Filter results by product category (optional)",
    ),
):
    """
    Provide an image URL and find visually similar products.

    Includes:
    - SSRF protection (blocks internal/private IPs)
    - Content-length validation (rejects oversized downloads)
    - Redis caching (optional, graceful degradation)
    """
    # SSRF protection — validate URL before making any request
    try:
        validate_url(image_url)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if not similarity_search or similarity_search.index.ntotal == 0:
        raise HTTPException(
            status_code=503,
            detail="Index is empty. Build the index first.",
        )

    # Check cache first
    cache_key = ""
    if redis_client:
        url_hash = hashlib.sha256(image_url.encode()).hexdigest()
        cat_suffix = f":cat:{category.lower()}" if category else ""
        cache_key = f"search:url:{url_hash}:k{top_k}{cat_suffix}"
        try:
            cached = redis_client.get(cache_key)
            if cached:
                logger.info("Cache hit for URL: %s", image_url)
                # Deserialize back into proper Pydantic model
                return SearchResponse.model_validate_json(cached)
        except Exception as e:
            logger.warning("Redis cache read error: %s", e)

    # Download image with async HTTP client
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36"
                )
            }
            resp = await client.get(image_url, headers=headers, follow_redirects=True)
            resp.raise_for_status()

            # Content-length check
            try:
                check_content_length(dict(resp.headers))
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))

    except httpx.HTTPError as e:
        raise HTTPException(
            status_code=400, detail=f"Failed to download image: {e}"
        )

    # Preprocess → embed → search (CPU-bound ops wrapped in executor)
    loop = asyncio.get_running_loop()
    image = await loop.run_in_executor(
        None, image_preprocessor.validate_and_process, resp.content
    )
    embedding = await loop.run_in_executor(
        None, embedding_extractor.extract_single, image
    )

    search_k = top_k * 3 if category else top_k
    results = similarity_search.search(embedding, k=search_k)

    # Category filter (post-search)
    if category:
        results = [
            r for r in results
            if r["product"].get("category", "").lower() == category.lower()
        ][:top_k]

    response = SearchResponse(
        query=f"url: {image_url}",
        results=[
            SearchResult(
                product=ProductInfo(**r["product"]),
                similarity_score=round(r["similarity_score"], 4),
                index_id=r["index_id"],
            )
            for r in results
        ],
        total_indexed=similarity_search.index.ntotal,
    )

    # Cache the response (expire after 1 hour)
    if redis_client and cache_key:
        try:
            redis_client.setex(cache_key, 3600, response.model_dump_json())
        except Exception as e:
            logger.warning("Redis cache write error: %s", e)

    return response


def _run_index_build():
    """Background task: build FAISS index from dataset."""
    global _build_in_progress

    try:
        logger.info("Starting index build...")

        # Load dataset
        dataset = ProductDataset(
            data_path=settings.data_path,
            image_dir=settings.image_dir,
        )

        # Download images
        products = dataset.download_images(max_workers=4)
        if not products:
            logger.error("No images downloaded successfully")
            return

        logger.info("Downloaded %d product images", len(products))

        # Extract embeddings in batch (returns only valid embeddings)
        image_paths = [p["path"] for p in products]
        embeddings, valid_indices = embedding_extractor.extract_batch(
            image_paths, batch_size=settings.batch_size
        )

        # Build metadata only for successfully embedded products
        metadata_list = [
            {
                "product_id": products[i]["product_id"],
                "name": products[i].get("name", ""),
                "category": products[i].get("category", "unknown"),
            }
            for i in valid_indices
        ]

        # Add to index
        similarity_search.add_vectors(embeddings, metadata_list)

        # Persist
        similarity_search.save()

        # Flush Redis cache — stale search results point to old index IDs
        if redis_client:
            try:
                # Delete only search-related keys, not all Redis data
                cursor = 0
                while True:
                    cursor, keys = redis_client.scan(
                        cursor=cursor, match="search:*", count=100
                    )
                    if keys:
                        redis_client.delete(*keys)
                    if cursor == 0:
                        break
                logger.info("Flushed stale search cache after re-indexing")
            except Exception as e:
                logger.warning("Failed to flush Redis cache: %s", e)

        logger.info(
            "Index build complete: %d products indexed", len(metadata_list)
        )

    except Exception as e:
        logger.error("Index build failed: %s", e, exc_info=True)
    finally:
        _build_in_progress = False


@app.post(
    "/api/v1/index/build",
    response_model=IndexBuildResponse,
    tags=["Index Management"],
)
async def build_index(background_tasks: BackgroundTasks):
    """
    Build the FAISS index from the product dataset.

    Steps:
    1. Load product data (JSON/CSV)
    2. Download product images in parallel
    3. Extract embeddings in batches via ResNet50
    4. Add to FAISS index and persist to disk

    Note: Runs synchronously for simplicity. In production, use a
    background task queue (Celery) for very large datasets.
    """
    global _build_in_progress

    # Prevent concurrent builds — atomic check-and-set via lock
    with _build_lock:
        if _build_in_progress:
            raise HTTPException(
                status_code=409,
                detail="Index build already in progress. Please wait.",
            )
        _build_in_progress = True

    background_tasks.add_task(_run_index_build)

    return IndexBuildResponse(
        status="building",
        products_indexed=similarity_search.index.ntotal,
        index_stats=similarity_search.get_stats(),
    )


@app.get(
    "/api/v1/index/stats",
    response_model=IndexStatsResponse,
    tags=["Index Management"],
)
async def index_stats():
    """Return current FAISS index statistics."""
    if not similarity_search:
        raise HTTPException(status_code=503, detail="Search service not initialized")

    stats = similarity_search.get_stats()
    return IndexStatsResponse(**stats)


@app.get(
    "/api/v1/search/product/{product_id}",
    response_model=SearchResponse,
    responses={404: {"model": ErrorResponse}},
    tags=["Search"],
)
async def search_by_product_id(
    product_id: str,
    top_k: int = Query(
        default=settings.default_top_k,
        ge=1,
        le=settings.max_top_k,
        description="Number of similar products to return",
    ),
    category: str = Query(
        default=None,
        description="Filter results by product category (optional)",
    ),
):
    """
    Find products similar to an already-indexed product.

    Looks up the product by ID in the index metadata, retrieves its
    embedding vector, and searches for nearest neighbors.
    This avoids re-uploading an image for products already in the catalog.
    """
    if not similarity_search or similarity_search.index.ntotal == 0:
        raise HTTPException(
            status_code=503,
            detail="Index is empty. Build the index first.",
        )

    # Find the index position for this product_id
    target_idx = None
    for idx, meta in similarity_search.metadata.items():
        if meta.get("product_id") == product_id:
            target_idx = idx
            break

    if target_idx is None:
        raise HTTPException(
            status_code=404,
            detail=f"Product '{product_id}' not found in index",
        )

    # Reconstruct the embedding vector from the FAISS index
    # (avoids needing the original image)
    if faiss.get_num_gpus() > 0 and hasattr(similarity_search.index, "copyToCpu"):
        cpu_index = faiss.index_gpu_to_cpu(similarity_search.index)
    else:
        cpu_index = similarity_search.index

    query_vector = cpu_index.reconstruct(target_idx)

    # Search (request top_k+1 because the product itself will be in results)
    search_k = (top_k + 1) * 3 if category else top_k + 1
    raw_results = similarity_search.search(query_vector, k=search_k)

    # Exclude the query product itself from results
    results = [r for r in raw_results if r["index_id"] != target_idx]

    # Category filter (post-search)
    if category:
        results = [
            r for r in results
            if r["product"].get("category", "").lower() == category.lower()
        ]

    results = results[:top_k]

    return SearchResponse(
        query=f"product_id: {product_id}",
        results=[
            SearchResult(
                product=ProductInfo(**r["product"]),
                similarity_score=round(r["similarity_score"], 4),
                index_id=r["index_id"],
            )
            for r in results
        ],
        total_indexed=similarity_search.index.ntotal,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
    )
