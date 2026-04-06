"""
Tests for the product data loader service.

Covers:
- JSON loading and cleaning
- CSV loading fallback
- Data deduplication
- Image URL extraction from list field
- Error handling for missing files
"""

import json
import tempfile
import shutil
from pathlib import Path

import pytest

from app.services.data_loader import ProductDataset


@pytest.fixture
def sample_json_data():
    """Sample product data matching the jewellery dataset structure."""
    return [
        {
            "product_id": "prod_001",
            "name": "Gold Ring",
            "category": "Rings",
            "images": [
                "https://example.com/ring1.jpg",
                "https://example.com/ring1_alt.jpg",
            ],
        },
        {
            "product_id": "prod_002",
            "name": "Silver Necklace",
            "category": "Necklaces",
            "images": ["https://example.com/necklace1.jpg"],
        },
        {
            "product_id": "prod_003",
            "name": "Diamond Earrings",
            "category": "Earrings",
            "images": ["https://example.com/earring1.jpg"],
        },
    ]


@pytest.fixture
def json_dataset_path(sample_json_data, tmp_path):
    """Create a temporary JSON dataset file."""
    path = tmp_path / "products.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(sample_json_data, f)
    return str(path)


@pytest.fixture
def image_dir(tmp_path):
    """Temporary image directory."""
    d = tmp_path / "images"
    d.mkdir()
    return str(d)


class TestDataLoading:
    def test_load_json(self, json_dataset_path, image_dir):
        dataset = ProductDataset(data_path=json_dataset_path, image_dir=image_dir)
        assert len(dataset.df) == 3
        assert "image_url" in dataset.df.columns

    def test_json_image_url_extraction(self, json_dataset_path, image_dir):
        """Should extract first URL from images list."""
        dataset = ProductDataset(data_path=json_dataset_path, image_dir=image_dir)
        first_url = dataset.df.iloc[0]["image_url"]
        assert first_url == "https://example.com/ring1.jpg"

    def test_deduplication(self, tmp_path, image_dir):
        """Duplicate product_ids should be removed."""
        data = [
            {
                "product_id": "dup_001",
                "name": "Item A",
                "category": "Rings",
                "images": ["https://example.com/a.jpg"],
            },
            {
                "product_id": "dup_001",  # duplicate
                "name": "Item A Copy",
                "category": "Rings",
                "images": ["https://example.com/a_copy.jpg"],
            },
            {
                "product_id": "dup_002",
                "name": "Item B",
                "category": "Earrings",
                "images": ["https://example.com/b.jpg"],
            },
        ]
        path = tmp_path / "dupes.json"
        with open(path, "w") as f:
            json.dump(data, f)

        dataset = ProductDataset(data_path=str(path), image_dir=image_dir)
        assert len(dataset.df) == 2

    def test_invalid_urls_filtered(self, tmp_path, image_dir):
        """Non-HTTP URLs should be removed."""
        data = [
            {
                "product_id": "good",
                "name": "Valid",
                "category": "Rings",
                "images": ["https://example.com/valid.jpg"],
            },
            {
                "product_id": "bad",
                "name": "Invalid",
                "category": "Rings",
                "images": ["ftp://example.com/invalid.jpg"],
            },
        ]
        path = tmp_path / "urls.json"
        with open(path, "w") as f:
            json.dump(data, f)

        dataset = ProductDataset(data_path=str(path), image_dir=image_dir)
        assert len(dataset.df) == 1
        assert dataset.df.iloc[0]["product_id"] == "good"

    def test_missing_file_raises(self, image_dir):
        with pytest.raises(FileNotFoundError):
            ProductDataset(data_path="/nonexistent/file.json", image_dir=image_dir)

    def test_unsupported_format_raises(self, tmp_path, image_dir):
        path = tmp_path / "data.xml"
        path.write_text("<data/>")
        with pytest.raises(ValueError, match="Unsupported"):
            ProductDataset(data_path=str(path), image_dir=image_dir)

    def test_null_images_filtered(self, tmp_path, image_dir):
        """Products with null/empty images should be removed."""
        data = [
            {
                "product_id": "has_image",
                "name": "Good",
                "category": "Rings",
                "images": ["https://example.com/good.jpg"],
            },
            {
                "product_id": "no_image",
                "name": "Bad",
                "category": "Rings",
                "images": [],
            },
        ]
        path = tmp_path / "nulls.json"
        with open(path, "w") as f:
            json.dump(data, f)

        dataset = ProductDataset(data_path=str(path), image_dir=image_dir)
        assert len(dataset.df) == 1

    def test_csv_fallback(self, tmp_path, image_dir):
        """Should also support CSV format."""
        csv_path = tmp_path / "products.csv"
        csv_path.write_text(
            "product_id,name,category,image_url\n"
            "csv_001,CSV Product,Rings,https://example.com/csv.jpg\n"
        )
        dataset = ProductDataset(data_path=str(csv_path), image_dir=image_dir)
        assert len(dataset.df) == 1
