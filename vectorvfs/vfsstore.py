import io
import os
from enum import Enum
from pathlib import Path
from typing import Optional

import torch

try:
    import boto3
except ImportError:  # pragma: no cover - optional dependency
    boto3 = None


class S3Mode(str, Enum):
    DISABLED = "disabled"
    LOCAL_PRIMARY = "local_primary"
    S3_PRIMARY = "s3_primary"

    @classmethod
    def from_value(cls, value: Optional[str]) -> "S3Mode":
        normalized = (value or "disabled").lower()
        if normalized in {"local", "local_primary", "local_with_s3", "local_with_sync"}:
            return cls.LOCAL_PRIMARY
        if normalized in {"s3", "s3_primary", "remote"}:
            return cls.S3_PRIMARY
        return cls.DISABLED


class XAttrFile:
    def __init__(self, file_path: Path) -> None:
        """
        Initialize an XAttrFile for managing extended attributes on a file.
        :param file_path: Path to the target file.
        """
        self.file_path = file_path

    def list(self) -> list[str]:
        """
        List all extended attribute names set on the file.
        :return: List of attribute names.
        """
        return os.listxattr(str(self.file_path))

    def write(self, key: str, data: bytes) -> None:
        """
        Write or replace an extended attribute on the file.
        :param key: Name of the attribute (e.g., 'user.comment').
        :param data: Bytes to store in the attribute.
        """
        os.setxattr(str(self.file_path), key, data)

    def read(self, key: str) -> bytes:
        """
        Read the value of an extended attribute from the file.
        :param key: Name of the attribute to read.
        :return: Bytes stored in the attribute.
        """
        return os.getxattr(str(self.file_path), key)

    def remove(self, key: str) -> None:
        """
        Remove an extended attribute from the file.
        :param key: Name of the attribute to remove.
        """
        os.removexattr(str(self.file_path), key)


class VFSStore:
    def __init__(
        self,
        xattrfile: XAttrFile,
        *,
        s3_mode: Optional[str] = None,
        s3_bucket: Optional[str] = None,
        s3_index: Optional[str] = None,
        s3_region: Optional[str] = None,
        s3_client=None,
    ) -> None:
        self.xattrfile = xattrfile
        self.s3_mode = S3Mode.from_value(s3_mode or os.getenv("VECTORVFS_S3_MODE"))
        self.s3_bucket = s3_bucket or os.getenv("VECTORVFS_S3_BUCKET")
        self.s3_index = s3_index or os.getenv("VECTORVFS_S3_INDEX")
        self.s3_region = (
            s3_region or os.getenv("VECTORVFS_S3_REGION") or os.getenv("AWS_REGION")
        )
        self.s3_client = s3_client or self._build_s3_client()

    def _tensor_to_bytes(self, tensor: torch.Tensor) -> bytes:
        buffer = io.BytesIO()
        torch.save(tensor, buffer)
        return buffer.getvalue()
    
    def _bytes_to_tensor(self, b: bytes, map_location=None) -> torch.Tensor:
        buffer = io.BytesIO(b)
        return torch.load(buffer, map_location=map_location, weights_only=True)

    def _build_s3_client(self):
        if self.s3_mode == S3Mode.DISABLED:
            return None
        if boto3 is None:
            raise ImportError("boto3 is required for S3 vector support")
        client_kwargs = {}
        if self.s3_region:
            client_kwargs["region_name"] = self.s3_region
        return boto3.client("s3vectors", **client_kwargs)

    def _s3_ready(self) -> bool:
        return (
            self.s3_mode != S3Mode.DISABLED
            and self.s3_client is not None
            and self.s3_bucket
            and self.s3_index
        )

    def _s3_key(self) -> Optional[str]:
        file_path = getattr(self.xattrfile, "file_path", None)
        if file_path is None:
            return None
        return str(file_path)

    def _write_s3_vector(self, tensor: torch.Tensor) -> bool:
        if not self._s3_ready():
            return False

        key = self._s3_key()
        if key is None:
            return False

        vector = {
            "key": key,
            "data": {"float32": tensor.detach().float().cpu().view(-1).tolist()},
            "metadata": {"source_path": key},
        }
        try:
            self.s3_client.put_vectors(
                vectorBucketName=self.s3_bucket,
                indexName=self.s3_index,
                vectors=[vector],
            )
            return True
        except Exception:
            return False

    def _read_s3_vector(self, map_location=None) -> Optional[torch.Tensor]:
        if not self._s3_ready():
            return None

        key = self._s3_key()
        if key is None:
            return None

        try:
            response = self.s3_client.get_vectors(
                vectorBucketName=self.s3_bucket,
                indexName=self.s3_index,
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

    def write_tensor(self, tensor: torch.Tensor) -> int:
        btensor = self._tensor_to_bytes(tensor)
        wrote_local = False
        if self.s3_mode != S3Mode.S3_PRIMARY:
            self.xattrfile.write("user.vectorvfs", btensor)
            wrote_local = True

        s3_success = False
        if self.s3_mode != S3Mode.DISABLED:
            s3_success = self._write_s3_vector(tensor)

        if self.s3_mode == S3Mode.S3_PRIMARY and not s3_success and not wrote_local:
            self.xattrfile.write("user.vectorvfs", btensor)
        return len(btensor)

    def read_tensor(self, map_location=None) -> torch.Tensor:
        if self.s3_mode == S3Mode.S3_PRIMARY:
            tensor = self._read_s3_vector(map_location=map_location)
            if tensor is not None:
                return tensor

        local_error: Optional[OSError] = None
        try:
            btensor = self.xattrfile.read("user.vectorvfs")
            return self._bytes_to_tensor(btensor, map_location=map_location)
        except OSError as exc:
            local_error = exc

        if self.s3_mode != S3Mode.DISABLED:
            tensor = self._read_s3_vector(map_location=map_location)
            if tensor is not None:
                if self.s3_mode == S3Mode.LOCAL_PRIMARY:
                    try:
                        self.xattrfile.write("user.vectorvfs", self._tensor_to_bytes(tensor))
                    except OSError:
                        pass
                return tensor

        if local_error is not None:
            raise local_error
        raise
