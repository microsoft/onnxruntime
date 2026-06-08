#!/usr/bin/env python
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import io
import tarfile
import zipfile

import pytest
from parity_utilities import find_transformers_source

# The source module is not copied into the test build output directory, use a helper function to find it.
if find_transformers_source():
    from convert_tf_models_to_pytorch import safe_extract_archive
else:
    from onnxruntime.transformers.convert_tf_models_to_pytorch import safe_extract_archive


def test_safe_extract_archive_rejects_tar_path_traversal(tmp_path):
    archive_path = tmp_path / "malicious.tar.gz"
    with tarfile.open(archive_path, "w:gz") as tar_ref:
        data = b"malicious"
        member = tarfile.TarInfo(name="../escape.txt")
        member.size = len(data)
        tar_ref.addfile(member, io.BytesIO(data))

    with pytest.raises(ValueError, match="outside"):
        safe_extract_archive(archive_path, tmp_path)


def test_safe_extract_archive_rejects_zip_path_traversal(tmp_path):
    archive_path = tmp_path / "malicious.zip"
    with zipfile.ZipFile(archive_path, "w") as zip_ref:
        zip_ref.writestr("../escape.txt", "malicious")

    with pytest.raises(ValueError, match="outside"):
        safe_extract_archive(archive_path, tmp_path)


def test_safe_extract_archive_rejects_zip_backslash_traversal(tmp_path):
    archive_path = tmp_path / "malicious_backslash.zip"
    with zipfile.ZipFile(archive_path, "w") as zip_ref:
        zip_ref.writestr("..\\escape.txt", "malicious")

    with pytest.raises(ValueError, match="outside"):
        safe_extract_archive(archive_path, tmp_path)


def test_safe_extract_archive_rejects_tar_symlink(tmp_path):
    archive_path = tmp_path / "malicious_link.tar.gz"
    with tarfile.open(archive_path, "w:gz") as tar_ref:
        member = tarfile.TarInfo(name="link")
        member.type = tarfile.SYMTYPE
        member.linkname = "/etc/passwd"
        tar_ref.addfile(member)

    with pytest.raises(ValueError, match="link"):
        safe_extract_archive(archive_path, tmp_path)
