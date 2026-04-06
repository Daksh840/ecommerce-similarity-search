"""
Integration tests for the FastAPI application.

Tests the full request pipeline:
    HTTP Request → FastAPI → Preprocessing → Embedding → FAISS → Response

Uses FastAPI's TestClient (backed by Starlette's ASGI test client).
These tests load the actual model — they are slower than unit tests
but verify the entire system works end-to-end.
"""

import io
import numpy as np
import pytest
from PIL import Image
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    """
    Create a test client with the full app.
    scope='module' to avoid re-loading the model per test.
    """
    from app.main import app
    
    with TestClient(app) as c:
        yield c


class TestHealthCheck:
    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "index_loaded" in data
        assert "total_products" in data

    def test_health_response_format(self, client):
        response = client.get("/health")
        data = response.json()
        assert isinstance(data["total_products"], int)
        assert isinstance(data["index_loaded"], bool)


class TestIndexStats:
    def test_stats_returns_200(self, client):
        response = client.get("/api/v1/index/stats")
        assert response.status_code == 200
        data = response.json()
        assert "total_vectors" in data
        assert "dimension" in data
        assert "index_type" in data
        assert data["index_type"] == "IndexFlatIP"
        assert data["dimension"] == 2048


class TestSearchByImage:
    def test_valid_image_upload(self, client):
        """Upload a valid image — should return search results or 503 (empty index)."""
        img = Image.new("RGB", (256, 256), color="red")
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        buf.seek(0)

        response = client.post(
            "/api/v1/search/image",
            files={"file": ("test.jpg", buf, "image/jpeg")},
        )

        # Either 200 (index has data) or 503 (empty index) — both are valid
        assert response.status_code in (200, 503)

        if response.status_code == 200:
            data = response.json()
            assert "results" in data
            assert "total_indexed" in data
            assert isinstance(data["results"], list)

    def test_non_image_upload_rejected(self, client):
        """Non-image content type should be rejected."""
        buf = io.BytesIO(b"this is not an image")

        response = client.post(
            "/api/v1/search/image",
            files={"file": ("test.txt", buf, "text/plain")},
        )

        assert response.status_code == 422

    def test_invalid_image_bytes_rejected(self, client):
        """Invalid image bytes should return 422 or 500."""
        buf = io.BytesIO(b"fake image data that is not valid")

        response = client.post(
            "/api/v1/search/image",
            files={"file": ("test.jpg", buf, "image/jpeg")},
        )

        assert response.status_code in (422, 500, 503)

    def test_top_k_parameter(self, client):
        """top_k query parameter should be validated."""
        img = Image.new("RGB", (256, 256), color="red")
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        buf.seek(0)

        # top_k = 0 should fail validation (ge=1)
        response = client.post(
            "/api/v1/search/image?top_k=0",
            files={"file": ("test.jpg", buf, "image/jpeg")},
        )
        assert response.status_code == 422


class TestSearchByUrl:
    def test_invalid_url_rejected(self, client):
        """Non-HTTP URLs should be rejected."""
        response = client.post(
            "/api/v1/search/url?image_url=ftp://example.com/image.jpg"
        )
        assert response.status_code == 400

    def test_localhost_blocked(self, client):
        """SSRF: localhost should be blocked."""
        response = client.post(
            "/api/v1/search/url?image_url=http://localhost/secret"
        )
        assert response.status_code == 400
        assert "not allowed" in response.json()["detail"].lower()

    def test_private_ip_blocked(self, client):
        """SSRF: private IPs should be blocked."""
        response = client.post(
            "/api/v1/search/url?image_url=http://192.168.1.1/image.jpg"
        )
        assert response.status_code == 400

    def test_metadata_endpoint_blocked(self, client):
        """SSRF: AWS metadata endpoint should be blocked."""
        response = client.post(
            "/api/v1/search/url?image_url=http://169.254.169.254/latest/meta-data/"
        )
        assert response.status_code == 400


class TestSearchByProductId:
    def test_product_not_found(self, client):
        """Non-existent product_id should return 404."""
        response = client.get("/api/v1/search/product/nonexistent_id")
        # Either 404 (not found) or 503 (empty index) are valid
        assert response.status_code in (404, 503)

    def test_invalid_top_k(self, client):
        """top_k=0 should fail validation."""
        response = client.get(
            "/api/v1/search/product/some_id?top_k=0"
        )
        assert response.status_code == 422


class TestSwaggerDocs:
    def test_openapi_schema(self, client):
        """OpenAPI schema should be accessible and include all endpoints."""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        schema = response.json()
        assert "paths" in schema
        assert "/health" in schema["paths"]
        assert "/api/v1/search/image" in schema["paths"]
        assert "/api/v1/search/url" in schema["paths"]
        assert "/api/v1/search/product/{product_id}" in schema["paths"]
        assert "/api/v1/index/build" in schema["paths"]
        assert "/api/v1/index/stats" in schema["paths"]
