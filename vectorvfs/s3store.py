from pathlib import Path
from typing import Optional

import torch

from vectorvfs.config import S3Config, S3Mode

try:
    import boto3
except ImportError:  # pragma: no cover - optional dependency
    boto3 = None


class S3VectorStore:
    def __init__(self, config: S3Config, client=None) -> None:
        self.config = config
        self.client = client or self._build_client()

    def _build_client(self):
        if self.config.mode == S3Mode.DISABLED:
            return None
        if boto3 is None:
            raise ImportError("boto3 is required for S3 vector support")
        client_kwargs = {}
        if self.config.region:
            client_kwargs["region_name"] = self.config.region
        return boto3.client("s3vectors", **client_kwargs)

    def _ready(self) -> bool:
        return (
            self.config.mode != S3Mode.DISABLED
            and self.client is not None
            and self.config.bucket
            and self.config.index
        )

    def _key(self, file_path: Path) -> str:
        return str(file_path)

    def write(self, file_path: Path, tensor: torch.Tensor) -> bool:
        if not self._ready():
            return False

        key = self._key(file_path)
        vector = {
            "key": key,
            "data": {"float32": tensor.detach().float().cpu().view(-1).tolist()},
            "metadata": {"source_path": key},
        }
        try:
            self.client.put_vectors(
                vectorBucketName=self.config.bucket,
                indexName=self.config.index,
                vectors=[vector],
            )
            return True
        except Exception:
            return False

    def read(self, file_path: Path, map_location=None) -> Optional[torch.Tensor]:
        if not self._ready():
            return None

        key = self._key(file_path)
        try:
            response = self.client.get_vectors(
                vectorBucketName=self.config.bucket,
                indexName=self.config.index,
                keys=[key],
                returnData=True,
            )
        except Exception:
            return None

        vectors = response.get("vectors") or []
        if not vectors:
            return None

        data = vectors[0].get("data", {})
        floats = data.get("float32") or data.get("float64") or data.get("FLOAT32")
        if floats is None:
            return None

        tensor = torch.tensor(floats, dtype=torch.float32)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        if map_location is not None:
            tensor = tensor.to(map_location)
        return tensor

    def query(self, query_tensor: torch.Tensor, top_k: int) -> list[tuple[str, float]]:
        if not self._ready():
            return []

        vector = query_tensor.detach().float().cpu().view(-1).tolist()
        try:
            response = self.client.query_vectors(
                vectorBucketName=self.config.bucket,
                indexName=self.config.index,
                queryVector={"float32": vector},
                topK=top_k,
                returnMetadata=True,
                returnData=False,
            )
        except Exception:
            return []

        matches = (
            response.get("matches")
            or response.get("vectors")
            or response.get("results")
            or []
        )

        results: list[tuple[str, float]] = []
        for item in matches:
            key = item.get("key") or item.get("id") or item.get("vectorId")
            score = item.get("score") or item.get("similarity") or item.get("distance")
            if key is None or score is None:
                continue
            try:
                results.append((str(key), float(score)))
            except Exception:
                continue
        return results
