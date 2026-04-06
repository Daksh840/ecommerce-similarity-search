"""
Feature extraction service using ResNet50.

Architecture:
    PIL Image → Resize(256) → CenterCrop(224) → Normalize → ResNet50 → 2048-dim → L2 Normalize

Why ResNet50?
- Strong ImageNet V2 pretrained features generalize well to product images
- 2048-dim output provides rich representation without being too large
- Well-studied, reproducible, widely available

Interview Points:
- We remove the final FC layer to get feature vectors, not class predictions
- L2 normalization is CRITICAL: it makes inner product == cosine similarity
- AMP (automatic mixed precision) gives ~2x speedup on GPU with no accuracy loss
- Batch processing amortizes GPU kernel launch overhead
"""

import logging
from typing import Union, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from torchvision.models import resnet50, ResNet50_Weights

logger = logging.getLogger(__name__)


class EmbeddingExtractor:
    """
    Production-grade feature extraction with:
    - Automatic mixed precision (AMP) for speed
    - Batch processing for throughput
    - Center cropping (focus on object, not background)
    - Robust error handling (no zero-vector poisoning)
    """

    def __init__(self, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Using device: %s", self.device)

        # Load pretrained ResNet50 with V2 weights (better accuracy than V1)
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

        # Remove final classification layer (FC)
        # We want the 2048-dim feature vector from the average pooling layer
        self.model = nn.Sequential(*list(model.children())[:-1])
        self.model = self.model.to(self.device)
        self.model.eval()

        # Standard ImageNet preprocessing pipeline
        self.transform = T.Compose([
            T.Resize(256),             # Resize shorter side to 256
            T.CenterCrop(224),         # Square crop (focus on center object)
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet channel means
                std=[0.229, 0.224, 0.225],   # ImageNet channel stds
            ),
        ])

        # Half-precision speedup on GPU
        self.use_amp = self.device == "cuda"

    def extract_single(self, image_input: Union[str, Image.Image]) -> np.ndarray:
        """
        Extract embedding from a single image.

        Args:
            image_input: File path (str) or PIL Image

        Returns:
            L2-normalized 2048-dimensional numpy array (float32)

        Raises:
            RuntimeError: If feature extraction fails
        """
        try:
            # Load image
            if isinstance(image_input, str):
                image = Image.open(image_input).convert("RGB")
            else:
                image = image_input.convert("RGB")

            # Transform to tensor
            tensor = self.transform(image).unsqueeze(0).to(self.device)

            # Inference with no gradient tracking (saves memory)
            with torch.no_grad():
                if self.use_amp:
                    with torch.amp.autocast("cuda"):
                        embedding = self.model(tensor)
                else:
                    embedding = self.model(tensor)

            # Flatten (1, 2048, 1, 1) → (2048,) and move to CPU
            # Use reshape(-1) instead of squeeze() for consistency with extract_batch
            # squeeze() would also work here, but reshape is explicit about intent
            embedding = embedding.cpu().numpy().reshape(-1)

            # L2 normalize — CRITICAL for cosine similarity via inner product
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm

            return embedding.astype("float32")

        except Exception as e:
            logger.error("Extraction failed for %s: %s", image_input, e)
            raise

    def extract_batch(
        self, image_paths: List[str], batch_size: int = 32
    ) -> Tuple[np.ndarray, List[int]]:
        """
        Batch feature extraction for index building.

        Unlike extract_single, this method skips failed images entirely
        instead of inserting zero vectors (which would poison the index).

        Args:
            image_paths: List of image file paths
            batch_size: Number of images to process per GPU batch

        Returns:
            Tuple of:
            - embeddings: (n_valid, 2048) numpy array of L2-normalized vectors
            - valid_indices: List of original indices that succeeded
        """
        all_embeddings = []
        valid_indices = []

        for batch_start in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[batch_start : batch_start + batch_size]
            batch_tensors = []
            batch_valid = []

            # Load and transform batch — skip failures entirely
            for j, path in enumerate(batch_paths):
                try:
                    img = Image.open(path).convert("RGB")
                    tensor = self.transform(img)
                    batch_tensors.append(tensor)
                    batch_valid.append(batch_start + j)
                except Exception as e:
                    logger.warning("Skipping %s: %s", path, e)
                    continue  # Skip — do NOT insert zero vector

            if not batch_tensors:
                continue

            batch = torch.stack(batch_tensors).to(self.device)

            with torch.no_grad():
                if self.use_amp:
                    with torch.amp.autocast("cuda"):
                        batch_emb = self.model(batch)
                else:
                    batch_emb = self.model(batch)

            batch_emb = batch_emb.cpu().numpy().reshape(len(batch_tensors), -1)

            # L2 normalize each vector
            norms = np.linalg.norm(batch_emb, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
            batch_emb = batch_emb / norms

            all_embeddings.append(batch_emb)
            valid_indices.extend(batch_valid)

            logger.info(
                "Batch %d-%d: extracted %d / %d embeddings",
                batch_start,
                batch_start + len(batch_paths),
                len(batch_tensors),
                len(batch_paths),
            )

        if not all_embeddings:
            return np.empty((0, 2048), dtype="float32"), []

        return np.vstack(all_embeddings).astype("float32"), valid_indices
