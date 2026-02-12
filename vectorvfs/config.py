import os
from dataclasses import dataclass
from enum import Enum
from typing import Optional


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


@dataclass
class S3Config:
    mode: S3Mode
    bucket: Optional[str]
    index: Optional[str]
    region: Optional[str]


def load_s3_config(
    *,
    mode: Optional[str] = None,
    bucket: Optional[str] = None,
    index: Optional[str] = None,
    region: Optional[str] = None,
) -> S3Config:
    return S3Config(
        mode=S3Mode.from_value(mode or os.getenv("VECTORVFS_S3_MODE")),
        bucket=bucket or os.getenv("VECTORVFS_S3_BUCKET"),
        index=index or os.getenv("VECTORVFS_S3_INDEX"),
        region=region
        or os.getenv("VECTORVFS_S3_REGION")
        or os.getenv("AWS_REGION"),
    )
