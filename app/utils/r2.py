import uuid
from pathlib import Path

import boto3
from botocore.config import Config

from app.core.config import (
    R2_ACCESS_KEY_ID,
    R2_BUCKET_TEMP,
    R2_ENDPOINT_URL,
    R2_SECRET_ACCESS_KEY,
)


def _s3():
    if not R2_ENDPOINT_URL:
        raise RuntimeError("R2 not configured — set R2_ACCOUNT_ID in environment")
    return boto3.client(
        "s3",
        endpoint_url=R2_ENDPOINT_URL,
        aws_access_key_id=R2_ACCESS_KEY_ID,
        aws_secret_access_key=R2_SECRET_ACCESS_KEY,
        config=Config(signature_version="s3v4"),
        region_name="auto",
    )


def generate_upload_key(filename: str) -> str:
    ext = Path(filename).suffix.lower() or ".bin"
    return f"{uuid.uuid4()}{ext}"


def presigned_put_url(object_key: str, content_type: str, expires_in: int = 300) -> str:
    ct = content_type or "application/octet-stream"
    return _s3().generate_presigned_url(
        "put_object",
        Params={"Bucket": R2_BUCKET_TEMP, "Key": object_key, "ContentType": ct},
        ExpiresIn=expires_in,
    )


def download_object(object_key: str, dest_path: str) -> None:
    _s3().download_file(R2_BUCKET_TEMP, object_key, dest_path)
