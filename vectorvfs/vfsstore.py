import io
import os
from pathlib import Path
from typing import Optional

import torch

from vectorvfs.config import S3Mode, load_s3_config
from vectorvfs.s3store import S3VectorStore


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
        self.s3_config = load_s3_config(
            mode=s3_mode, bucket=s3_bucket, index=s3_index, region=s3_region
        )
        self.s3_store = (
            S3VectorStore(self.s3_config, client=s3_client)
            if self.s3_config.mode != S3Mode.DISABLED
            else None
        )

    def _tensor_to_bytes(self, tensor: torch.Tensor) -> bytes:
        buffer = io.BytesIO()
        torch.save(tensor, buffer)
        return buffer.getvalue()
    
    def _bytes_to_tensor(self, b: bytes, map_location=None) -> torch.Tensor:
        buffer = io.BytesIO(b)
        return torch.load(buffer, map_location=map_location, weights_only=True)

    def write_tensor(self, tensor: torch.Tensor) -> int:
        btensor = self._tensor_to_bytes(tensor)
        wrote_local = False
        if self.s3_config.mode != S3Mode.S3_PRIMARY:
            self.xattrfile.write("user.vectorvfs", btensor)
            wrote_local = True

        s3_success = False
        if self.s3_store is not None:
            s3_success = self.s3_store.write(self.xattrfile.file_path, tensor)

        if (
            self.s3_config.mode == S3Mode.S3_PRIMARY
            and not s3_success
            and not wrote_local
        ):
            self.xattrfile.write("user.vectorvfs", btensor)
        return len(btensor)

    def read_tensor(self, map_location=None) -> torch.Tensor:
        if self.s3_config.mode == S3Mode.S3_PRIMARY and self.s3_store is not None:
            tensor = self.s3_store.read(
                self.xattrfile.file_path, map_location=map_location
            )
            if tensor is not None:
                return tensor

        local_error: Optional[OSError] = None
        try:
            btensor = self.xattrfile.read("user.vectorvfs")
            return self._bytes_to_tensor(btensor, map_location=map_location)
        except OSError as exc:
            local_error = exc

        if self.s3_store is not None:
            tensor = self.s3_store.read(
                self.xattrfile.file_path, map_location=map_location
            )
            if tensor is not None:
                if self.s3_config.mode == S3Mode.LOCAL_PRIMARY:
                    try:
                        self.xattrfile.write("user.vectorvfs", self._tensor_to_bytes(tensor))
                    except OSError:
                        pass
                return tensor

        if local_error is not None:
            raise local_error
        raise
