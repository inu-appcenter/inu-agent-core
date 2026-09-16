"""
Embedding Client Module for inu-agent-core
Supports local in-memory FastEmbed (ONNX), OpenAI-compatible HTTP embeddings,
and deterministic Mock embeddings for offline/testing scenarios.
"""
from typing import List, Optional
import asyncio
import numpy as np

from app.core.config import settings
from app.core.logging import logger


class FastEmbedBackend:
    """
    Local in-memory embedding backend using FastEmbed (ONNX runtime).
    Fast, CPU-efficient, completely free, with no external network calls after model caching.
    """
    _instance: Optional["FastEmbedBackend"] = None
    _model = None

    @classmethod
    def get_instance(cls) -> "FastEmbedBackend":
        if cls._instance is None:
            cls._instance = FastEmbedBackend()
        return cls._instance

    def _get_model(self):
        if self._model is None:
            from fastembed import TextEmbedding
            model_name = settings.EMBEDDING_MODEL_NAME
            logger.info(f"Initializing FastEmbed TextEmbedding model: {model_name}")
            self._model = TextEmbedding(model_name=model_name)
        return self._model

    def embed(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        model = self._get_model()
        raw_embs = list(model.embed(texts))
        return [e.tolist() if hasattr(e, "tolist") else list(e) for e in raw_embs]


class OpenAIBackend:
    """
    Remote embedding backend using OpenAI-compatible /v1/embeddings endpoint (e.g. LiteLLM, vLLM, OpenAI).
    """
    def __init__(self):
        import httpx
        self._base_url = settings.LLM_BASE_URL.rstrip("/")
        self._api_key = settings.LLM_API_KEY
        self._timeout = httpx.Timeout(15.0)

    async def embed(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        import httpx
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "inu-agent-core/0.1.0",
        }
        if self._api_key and self._api_key != "not-set":
            headers["Authorization"] = f"Bearer {self._api_key}"

        payload = {
            "model": settings.EMBEDDING_MODEL_NAME,
            "input": texts,
        }
        url = f"{self._base_url}/embeddings"

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(url, json=payload, headers=headers)
            if resp.status_code != 200:
                raise RuntimeError(f"OpenAI embedding request failed ({resp.status_code}): {resp.text}")
            data = resp.json()
            items = sorted(data.get("data", []), key=lambda x: x.get("index", 0))
            return [item.get("embedding", []) for item in items]


class MockEmbeddingBackend:
    """
    Deterministic pseudo-embedding backend for offline unit tests.
    Encodes token n-grams and character hashes into a normalized 384-d vector
    so semantic overlap produces positive cosine similarity without model files.
    """
    DIM = 384

    def embed(self, texts: List[str]) -> List[List[float]]:
        results: List[List[float]] = []
        for text in texts:
            vec = np.zeros(self.DIM, dtype=np.float32)
            words = text.lower().split()
            for idx, word in enumerate(words):
                h = hash(word) % self.DIM
                vec[h] += 1.0 / (idx + 1)
                for i in range(len(word) - 1):
                    bh = hash(word[i:i+2]) % self.DIM
                    vec[bh] += 0.5

            norm = np.linalg.norm(vec)
            if norm > 0:
                vec /= norm
            else:
                vec[0] = 1.0
            results.append(vec.tolist())
        return results


async def get_embeddings(texts: List[str], backend: Optional[str] = None) -> List[List[float]]:
    """
    Generate embeddings for a list of texts using the configured backend.
    """
    if not texts:
        return []

    target_backend = (backend or settings.EMBEDDING_BACKEND).lower()

    if target_backend == "fastembed":
        try:
            fast_backend = FastEmbedBackend.get_instance()
            return await asyncio.to_thread(fast_backend.embed, texts)
        except Exception as e:
            logger.warning(f"FastEmbed embedding failed ({e}), attempting fallback...")
            raise e

    elif target_backend == "openai":
        openai_backend = OpenAIBackend()
        return await openai_backend.embed(texts)

    elif target_backend == "mock":
        mock_backend = MockEmbeddingBackend()
        return mock_backend.embed(texts)

    else:
        raise ValueError(f"Unknown embedding backend: {target_backend}")
