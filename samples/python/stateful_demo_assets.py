# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Checked asset primitives shared by the local stateful demonstrations."""

import hashlib
import tempfile
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlopen


@dataclass(frozen=True)
class Asset:
    name: str
    size: int
    digest: str
    git_blob: bool = False
    repository_root: bool = False


def digest(path: Path, *, git_blob: bool = False) -> str:
    value = hashlib.sha1(usedforsecurity=False) if git_blob else hashlib.sha256()
    if git_blob:
        value.update(f"blob {path.stat().st_size}\0".encode("ascii"))
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def verify_asset(path: Path, asset: Asset) -> None:
    if path.stat().st_size != asset.size:
        raise ValueError(f"Incorrect size for {path}; expected {asset.size} bytes")
    if digest(path, git_blob=asset.git_blob) != asset.digest:
        raise ValueError(f"Checksum mismatch: {path}")


def acquire_file(destination: Path, asset: Asset, url: str) -> str:
    """Reuse verified files or publish a bounded, checked download atomically."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        verify_asset(destination, asset)
    else:
        with tempfile.TemporaryDirectory(prefix=".download-", dir=destination.parent) as temporary:
            pending = Path(temporary) / destination.name
            with urlopen(url, timeout=60) as response, pending.open("wb") as output:
                received = 0
                while chunk := response.read(1024 * 1024):
                    received += len(chunk)
                    if received > asset.size:
                        raise ValueError(f"Download exceeds expected size: {asset.name}")
                    output.write(chunk)
            verify_asset(pending, asset)
            pending.replace(destination)
    return digest(destination)
