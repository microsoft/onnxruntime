#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request  # noqa: F401
import zipfile  # noqa: F401

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
REPO_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "..", ".."))

sys.path.append(os.path.join(REPO_DIR, "tools", "python"))

from util import get_azcopy  # noqa: E402


def _download(azcopy_path, url, local_path):
    subprocess.run([azcopy_path, "cp", "--log-level", "NONE", url, local_path], check=True)


def _get_sha256_digest(file_path):
    alg = hashlib.sha256()
    read_bytes_length = 8192

    with open(file_path, mode="rb") as archive:
        while True:
            read_bytes = archive.read(read_bytes_length)
            if len(read_bytes) == 0:
                break
            alg.update(read_bytes)

    return alg.hexdigest()


def _check_file_sha256_digest(path, expected_digest):
    actual_digest = _get_sha256_digest(path)
    match = actual_digest.lower() == expected_digest.lower()
    if not match:
        raise RuntimeError(
            f"SHA256 digest mismatch, expected: {expected_digest.lower()}, actual: {actual_digest.lower()}"
        )


def main():
    parser = argparse.ArgumentParser(description="Downloads an Azure blob archive.")
    parser.add_argument("--azure_blob_url", required=True, help="The Azure blob URL.")
    parser.add_argument("--target_dir", required=True, help="The destination directory.")
    parser.add_argument("--archive_sha256_digest", help="The SHA256 digest of the archive. Verified if provided.")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as temp_dir, get_azcopy() as azcopy_path:
        archive_path = os.path.join(temp_dir, "archive.zip")
        print(f"Downloading archive from '{args.azure_blob_url}'...")

        azure_blob_url = args.azure_blob_url
        azure_blob_sas_token = os.getenv("AZURE_BLOB_SAS_TOKEN", None)
        if azure_blob_sas_token:
            azure_blob_url = azure_blob_url + "?" + azure_blob_sas_token

        _download(azcopy_path, azure_blob_url, archive_path)
        if args.archive_sha256_digest:
            _check_file_sha256_digest(archive_path, args.archive_sha256_digest)
        print(f"Extracting to '{args.target_dir}'...")
        shutil.unpack_archive(archive_path, args.target_dir)
        print("Done.")


if __name__ == "__main__":
    sys.exit(main())
