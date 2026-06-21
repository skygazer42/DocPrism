from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import Any, Protocol

import httpx


class EmbeddingProvider(Protocol):
    provider: str
    model: str
    dim: int

    def embed(self, texts: list[str]) -> list[list[float]]:
        ...


_PROVIDER_CACHE: dict[tuple[Any, ...], EmbeddingProvider] = {}


@dataclass(frozen=True)
class HashEmbeddingProvider:
    dim: int = 32
    provider: str = "hash"
    model: str = "sha256-deterministic"

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        vector: list[float] = []
        counter = 0
        seed = text.encode("utf-8")
        while len(vector) < self.dim:
            digest = hashlib.sha256(seed + counter.to_bytes(4, "big")).digest()
            for offset in range(0, len(digest), 4):
                value = int.from_bytes(digest[offset : offset + 4], "big", signed=False)
                vector.append((value / 2**31) - 1.0)
                if len(vector) == self.dim:
                    break
            counter += 1
        return vector


@dataclass
class TransformersEmbeddingProvider:
    model_path: str
    dim: int = 0
    device: str | None = None
    max_length: int = 512
    provider: str = "transformers"
    tokenizer: Any | None = None
    model_obj: Any | None = None

    @property
    def model(self) -> str:
        return self.model_path

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        tokenizer, model = self._components()
        import torch
        import torch.nn.functional as functional

        batch = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        device = self._device()
        if hasattr(batch, "to"):
            batch = batch.to(device)
        else:
            batch = {key: value.to(device) if hasattr(value, "to") else value for key, value in batch.items()}

        with torch.no_grad():
            output = model(**batch)
            hidden = output.last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1).expand(hidden.size()).float()
            summed = torch.sum(hidden * mask, dim=1)
            counts = torch.clamp(mask.sum(dim=1), min=1e-9)
            pooled = summed / counts
            normalized = functional.normalize(pooled, p=2, dim=1)
        vectors = [[round(float(value), 6) for value in row] for row in normalized.detach().cpu().tolist()]
        if vectors:
            self.dim = len(vectors[0])
        return vectors

    def _components(self) -> tuple[Any, Any]:
        if self.tokenizer is None or self.model_obj is None:
            from transformers import AutoModel, AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
            self.model_obj = AutoModel.from_pretrained(self.model_path, trust_remote_code=True)
        device = self._device()
        if hasattr(self.model_obj, "to"):
            self.model_obj = self.model_obj.to(device)
        if hasattr(self.model_obj, "eval"):
            self.model_obj.eval()
        return self.tokenizer, self.model_obj

    def _device(self) -> str:
        if self.device:
            return self.device
        requested = os.getenv("EMBEDDING_DEVICE", "auto")
        if requested != "auto":
            self.device = requested
            return self.device
        import torch

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        return self.device


@dataclass(frozen=True)
class OpenAICompatibleEmbeddingProvider:
    base_url: str
    api_key: str
    model: str
    dim: int
    provider: str = "openai_compatible"

    def embed(self, texts: list[str]) -> list[list[float]]:
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {"model": self.model, "input": texts}
        with httpx.Client(timeout=60.0, trust_env=False) as client:
            response = client.post(f"{self.base_url.rstrip('/')}/embeddings", headers=headers, json=payload)
            response.raise_for_status()
        data = response.json()["data"]
        vectors = [item["embedding"] for item in sorted(data, key=lambda item: item["index"])]
        if vectors and len(vectors[0]) != self.dim:
            object.__setattr__(self, "dim", len(vectors[0]))
        return vectors


def build_embedding_provider(provider_name: str, dim: int) -> EmbeddingProvider:
    if provider_name == "hash":
        return HashEmbeddingProvider(dim=dim)
    if provider_name == "transformers":
        model_path = os.getenv("EMBEDDING_MODEL_PATH") or os.getenv("EMBEDDING_MODEL")
        if not model_path:
            raise RuntimeError("EMBEDDING_MODEL_PATH or EMBEDDING_MODEL is required for transformers embedding")
        key = (
            provider_name,
            model_path,
            os.getenv("EMBEDDING_DEVICE"),
            int(os.getenv("EMBEDDING_MAX_LENGTH", "512")),
        )
        if key not in _PROVIDER_CACHE:
            _PROVIDER_CACHE[key] = TransformersEmbeddingProvider(
                model_path=model_path,
                dim=dim,
                device=os.getenv("EMBEDDING_DEVICE"),
                max_length=int(os.getenv("EMBEDDING_MAX_LENGTH", "512")),
            )
        return _PROVIDER_CACHE[key]
    if provider_name == "openai_compatible":
        api_key = os.getenv("EMBEDDING_API_KEY")
        if not api_key:
            raise RuntimeError("EMBEDDING_API_KEY is required for openai_compatible embedding")
        key = (
            provider_name,
            os.getenv("EMBEDDING_BASE_URL", "http://127.0.0.1:8000/v1"),
            os.getenv("EMBEDDING_MODEL", "bge-m3"),
            dim,
        )
        if key not in _PROVIDER_CACHE:
            _PROVIDER_CACHE[key] = OpenAICompatibleEmbeddingProvider(
                base_url=os.getenv("EMBEDDING_BASE_URL", "http://127.0.0.1:8000/v1"),
                api_key=api_key,
                model=os.getenv("EMBEDDING_MODEL", "bge-m3"),
                dim=dim,
            )
        return _PROVIDER_CACHE[key]
    raise RuntimeError(f"Unsupported EMBEDDING_PROVIDER={provider_name!r}")
