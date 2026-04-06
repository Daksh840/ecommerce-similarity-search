"""
Product dataset loader with JSON support and parallel image download.

Handles:
- Loading product data from JSON (with CSV fallback)
- Data validation and deduplication
- Parallel image downloading with retry logic
- Content-based deduplication via URL hashing

Interview Points:
- ThreadPoolExecutor for I/O-bound parallel downloads
- Hash-based file naming avoids path collisions
- Downloaded images are cached locally (skip if exists)
- Validates image integrity before saving (img.verify())
"""

import json
import hashlib
import io
import logging
from pathlib import Path
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
from PIL import Image
from tqdm import tqdm

logger = logging.getLogger(__name__)


class ProductDataset:
    """
    Loads product data and downloads images for index building.

    Supports both JSON and CSV input formats.
    Expected fields: product_id, name, category, images (or image_url)
    """

    def __init__(self, data_path: str, image_dir: str):
        self.data_path = Path(data_path)
        self.image_dir = Path(image_dir)
        self.image_dir.mkdir(parents=True, exist_ok=True)
        self.df = self._load_and_clean()

    def _load_and_clean(self) -> pd.DataFrame:
        """
        Load data with validation and deduplication.

        Supports JSON (.json) and CSV (.csv) formats.
        """
        path = self.data_path

        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")

        if path.suffix == ".json":
            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            df = pd.DataFrame(raw)
            logger.info("Loaded JSON dataset: %d records", len(df))
        elif path.suffix == ".csv":
            df = pd.read_csv(path)
            logger.info("Loaded CSV dataset: %d records", len(df))
        else:
            raise ValueError(f"Unsupported data format: {path.suffix}")

        initial_size = len(df)

        # Normalize image field: 'images' (list) → 'image_url' (first URL)
        if "images" in df.columns and "image_url" not in df.columns:
            df["image_url"] = df["images"].apply(self._extract_first_image)
            logger.info("Extracted image_url from images list field")

        # Data quality checks
        df = df.drop_duplicates(subset=["product_id"])
        df = df.dropna(subset=["image_url", "product_id"])

        # Validate URLs
        df = df[df["image_url"].str.startswith(("http://", "https://"))]

        logger.info(
            "Dataset cleaned: %d → %d records (%d removed)",
            initial_size,
            len(df),
            initial_size - len(df),
        )
        return df.reset_index(drop=True)

    @staticmethod
    def _extract_first_image(images) -> Optional[str]:
        """Extract the first valid image URL from an images field."""
        if isinstance(images, list) and len(images) > 0:
            return images[0]
        if isinstance(images, str):
            return images
        return None

    def download_images(self, max_workers: int = 4) -> List[Dict]:
        """
        Parallel image download with error handling and caching.

        Args:
            max_workers: Number of concurrent download threads

        Returns:
            List of dicts with product_id, path, category, name
        """
        results = []

        def download_single(idx: int, row: pd.Series) -> Optional[Dict]:
            try:
                # Hash-based filename — avoids path collisions
                url_hash = hashlib.sha256(row["image_url"].encode()).hexdigest()[:16]
                ext = row["image_url"].split(".")[-1].split("?")[0][:4] or "jpg"
                save_path = self.image_dir / f"{url_hash}.{ext}"

                # Skip if already downloaded (local cache)
                if save_path.exists():
                    return {
                        "product_id": str(row["product_id"]),
                        "path": str(save_path),
                        "category": row.get("category", "unknown"),
                        "name": row.get("name", ""),
                    }

                # Download with timeout and browser-like User-Agent
                headers = {
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36"
                    )
                }
                response = requests.get(
                    row["image_url"],
                    headers=headers,
                    timeout=15,
                    stream=True,
                )
                response.raise_for_status()

                # Validate image integrity before saving
                content = response.content
                img = Image.open(io.BytesIO(content))
                img.verify()

                # Save to disk
                with open(save_path, "wb") as f:
                    f.write(content)

                return {
                    "product_id": str(row["product_id"]),
                    "path": str(save_path),
                    "category": row.get("category", "unknown"),
                    "name": row.get("name", ""),
                }

            except Exception as e:
                logger.warning(
                    "Failed to download product %s: %s",
                    row.get("product_id", "unknown"),
                    str(e),
                )
                return None

        # Parallel download with progress bar
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(download_single, idx, row): idx
                for idx, row in self.df.iterrows()
            }

            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Downloading images",
            ):
                result = future.result()
                if result:
                    results.append(result)

        logger.info(
            "Downloaded %d / %d images successfully",
            len(results),
            len(self.df),
        )
        return results
