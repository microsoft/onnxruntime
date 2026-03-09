# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import base64
import functools
import os
import platform
import traceback
from pathlib import Path
from types import TracebackType
from typing import Optional

ORT_SUPPORT_DIR = r"Microsoft/DeveloperTools/.onnxruntime"


@property
@functools.lru_cache(maxsize=1)
def get_telemetry_base_dir() -> Path:
    os_name = platform.system()
    if os_name == "Windows":
        base_dir = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA")
        if not base_dir:
            base_dir = str(Path.home() / "AppData" / "Local")
        return Path(base_dir) / "Microsoft" / ".onnxruntime"

    if os_name == "Darwin":
        home = os.getenv("HOME")
        if home is None:
            raise ValueError("HOME environment variable not set")
        return Path(home) / "Library" / "Application Support" / ORT_SUPPORT_DIR

    home = os.getenv("XDG_CACHE_HOME", f"{os.getenv('HOME')}/.cache")
    if not home:
        raise ValueError("HOME environment variable not set")

    return Path(home) / ORT_SUPPORT_DIR


def _format_exception_message(ex: BaseException, tb: Optional[TracebackType] = None) -> str:
    """Format an exception and trim local paths for readability."""
    folder = "Olive"
    file_line = 'File "'
    formatted = traceback.format_exception(type(ex), ex, tb, limit=5)
    lines = []
    for line in formatted:
        line_trunc = line.strip()
        if line_trunc.startswith(file_line) and folder in line_trunc:
            idx = line_trunc.find(folder)
            if idx != -1:
                line_trunc = line_trunc[idx + len(folder) :]
        elif line_trunc.startswith(file_line):
            idx = line_trunc[len(file_line) :].find('"')
            line_trunc = line_trunc[idx + len(file_line) :]
        lines.append(line_trunc)
    return "\n".join(lines)


class _ExclusiveFileLock:
    """Cross-platform exclusive file lock context manager.

    Uses fcntl on Unix/Linux/macOS, msvcrt on Windows.
    Prevents cache corruption when multiple processes access the same file.

    Design decisions:
    - Lock is held for the entire duration of file access (prevents partial reads/writes)
    - Lock is released automatically on close (even on exceptions)
    - Platform-specific implementation (fcntl for POSIX, msvcrt for Windows)

    Assumptions:
    - File locking is supported on the platform
    - Lock is advisory on some systems (cooperative locking)
    """

    def __init__(self, file_path: Path, mode: str):
        self.file_path = file_path
        self.mode = mode
        self.file = None

    def __enter__(self):
        self.file = open(self.file_path, self.mode, encoding="utf-8")

        # Platform-specific locking
        if os.name == "posix":
            import fcntl

            fcntl.flock(self.file.fileno(), fcntl.LOCK_EX)
        elif os.name == "nt":
            import msvcrt

            # Lock 1 byte at position 0
            msvcrt.locking(self.file.fileno(), msvcrt.LK_LOCK, 1)

        return self.file

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file:
            # Unlock happens automatically on close
            self.file.close()


def _exclusive_file_lock(file_path: Path, mode: str):
    """Create an exclusive file lock context manager.

    :param file_path: Path to the file to lock.
    :param mode: File open mode ('r', 'a', 'w', etc.).
    :return: Context manager that returns an open file handle.
    """
    return _ExclusiveFileLock(file_path, mode)


def _encode_cache_line(plaintext: str) -> str:
    """Encode a single cache line using base64.

    :param plaintext: The plaintext string to encode.
    :return: Base64-encoded string (safe for a single text line).
    """
    return base64.b64encode(plaintext.encode("utf-8")).decode("ascii")


def _decode_cache_line(encoded: str) -> str:
    """Decode a single base64-encoded cache line.

    :param encoded: The base64-encoded string.
    :return: The decoded plaintext string.
    """
    return base64.b64decode(encoded.encode("ascii")).decode("utf-8")
