#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import hashlib
import os
import sys
import tempfile
import urllib.request
import zipfile

# update these if the E2E test data changes
ARCHIVE_BLOB_URL = "https://onnxruntimetestdata.blob.core.windows.net/training/onnxruntime_training_data.zip?snapshot=2019-12-12T18:53:12.3943496Z"
ARCHIVE_SHA256_DIGEST = "066abe039a544d4f100ad2047f5983b65dc27d0b24f9f448cf4474076b486500"

def _download(url, local_path):
  urllib.request.urlretrieve(url, local_path)

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

def _extract_archive(archive_path, target_dir):
  with zipfile.ZipFile(archive_path) as archive:
    archive.extractall(target_dir)

def main():
  parser = argparse.ArgumentParser(description="Downloads training end-to-end test data.")
  parser.add_argument("target_dir", help="The test data destination directory.")
  args = parser.parse_args()

  with tempfile.TemporaryDirectory() as temp_dir:
    archive_path = os.path.join(temp_dir, "archive.zip")
    print("Downloading E2E test data from '{}'...".format(ARCHIVE_BLOB_URL))
    _download(ARCHIVE_BLOB_URL, archive_path)
    _check_file_sha256_digest(archive_path, ARCHIVE_SHA256_DIGEST)
    print("Extracting to '{}'...".format(args.target_dir))
    _extract_archive(archive_path, args.target_dir)
    print("Done.")

if __name__ == "__main__":
  sys.exit(main())
