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
import urllib.request
import zipfile

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
REPO_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "..", ".."))

sys.path.append(os.path.join(REPO_DIR, "tools", "python"))

import get_azcopy  # noqa: E402

# update these if the E2E test data changes
ARCHIVE_BLOB_URL = "https://onnxruntimetestdata.blob.core.windows.net/training/onnxruntime_training_data.zip?snapshot=2020-06-15T23:17:35.8314853Z"
ARCHIVE_SHA256_DIGEST = "B01C169B6550D1A0A6F1B4E2F34AE2A8714B52DBB70AC04DA85D371F691BDFF9"

def _download(azcopy_path, url, local_path):
  subprocess.run([azcopy_path, "cp", "--log-level", "NONE", url, local_path], check=True)

def _get_sha256_digest(file_path):
  alg = hashlib.sha256()
  read_bytes_length = 8192

  with open(file_path, mode="rb") as archive:
    while True:
      read_bytes = archive.read(read_bytes_length)
      if len(read_bytes) == 0: break
      alg.update(read_bytes)

  return alg.hexdigest()

def _check_file_sha256_digest(path, expected_digest):
  actual_digest = _get_sha256_digest(path)
  match = actual_digest.lower() == expected_digest.lower()
  if not match:
    raise RuntimeError(
        "SHA256 digest mismatch, expected: {}, actual: {}".format(expected_digest.lower(), actual_digest.lower()))

def main():
  parser = argparse.ArgumentParser(description="Downloads training end-to-end test data.")
  parser.add_argument("target_dir", help="The test data destination directory.")
  args = parser.parse_args()

  with tempfile.TemporaryDirectory() as temp_dir, \
       get_azcopy.get_azcopy() as azcopy_path:
    archive_path = os.path.join(temp_dir, "archive.zip")
    print("Downloading E2E test data from '{}'...".format(ARCHIVE_BLOB_URL))
    _download(azcopy_path, ARCHIVE_BLOB_URL, archive_path)
    _check_file_sha256_digest(archive_path, ARCHIVE_SHA256_DIGEST)
    print("Extracting to '{}'...".format(args.target_dir))
    shutil.unpack_archive(archive_path, args.target_dir)
    print("Done.")

if __name__ == "__main__":
  sys.exit(main())
