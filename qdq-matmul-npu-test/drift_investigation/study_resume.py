"""Allocate non-overwriting case directories for resumable EP experiments."""

from pathlib import Path


def next_attempt_dir(output_dir: Path, name: str) -> Path:
    attempt = 0
    while True:
        suffix = f".retry-{attempt}" if attempt else ""
        folder = output_dir / f"{name}{suffix}"
        try:
            folder.mkdir()
        except FileExistsError:
            attempt += 1
            continue
        return folder


def completed(result: dict | None) -> bool:
    return result is not None and "error" not in result
