# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import contextlib
import logging
import os
import platform
import re
import shutil
import stat
import subprocess
import tempfile
import urllib.parse
import urllib.request

AZCOPY_VERSION = "10.4.3"

# See here for instructions on getting stable download links:
# https://docs.microsoft.com/en-us/azure/storage/common/storage-use-azcopy-v10#obtain-a-static-download-link
_AZCOPY_DOWNLOAD_URLS = {
    "Linux": "https://azcopyvnext.azureedge.net/release20200501/azcopy_linux_amd64_10.4.3.tar.gz",
    "Darwin": "https://azcopyvnext.azureedge.net/release20200501/azcopy_darwin_amd64_10.4.3.zip",
    "Windows": "https://azcopyvnext.azureedge.net/release20200501/azcopy_windows_amd64_10.4.3.zip",
}

_log = logging.getLogger("util.get_azcopy")


def _check_version(azcopy_path):
    proc = subprocess.run([azcopy_path, "--version"], stdout=subprocess.PIPE, text=True)  # noqa: PLW1510
    match = re.search(r"\d+(?:\.\d+)+", proc.stdout)

    if not match:
        raise RuntimeError("Failed to determine azcopy version.")

    return match.group(0) == AZCOPY_VERSION


def _find_azcopy(start_dir):
    for root, _, file_names in os.walk(start_dir):
        for file_name in file_names:
            if file_name == "azcopy" or file_name == "azcopy.exe":
                return os.path.join(root, file_name)
    raise RuntimeError(f"Failed to azcopy in '{start_dir}'.")


@contextlib.contextmanager
def get_azcopy(local_azcopy_path="azcopy"):
    """
    Creates a context manager that returns a path to a particular version of
    azcopy (specified in AZCOPY_VERSION). Downloads a temporary copy if needed.

    :param local_azcopy_path: Path to a local azcopy to try first.

    Example usage:
        with get_azcopy() as azcopy_path:
            subprocess.run([azcopy_path, "--version"])
    """
    with contextlib.ExitStack() as context_stack:
        azcopy_path = shutil.which(local_azcopy_path)

        if azcopy_path is None or not _check_version(azcopy_path):
            temp_dir = context_stack.enter_context(tempfile.TemporaryDirectory())

            download_url = _AZCOPY_DOWNLOAD_URLS[platform.system()]
            download_basename = urllib.parse.urlsplit(download_url).path.rsplit("/", 1)[-1]
            assert len(download_basename) > 0
            downloaded_path = os.path.join(temp_dir, download_basename)

            _log.info(f"Downloading azcopy from '{download_url}'...")
            urllib.request.urlretrieve(download_url, downloaded_path)

            extracted_path = os.path.join(temp_dir, "azcopy")
            shutil.unpack_archive(downloaded_path, extracted_path)

            azcopy_path = _find_azcopy(extracted_path)

            os.chmod(azcopy_path, stat.S_IXUSR)

        yield azcopy_path
