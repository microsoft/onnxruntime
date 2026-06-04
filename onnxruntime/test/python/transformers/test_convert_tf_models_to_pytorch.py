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

# The source module is not copied into the test build output directory, so skip the
# whole module when it cannot be located instead of failing during collection.
if find_transformers_source():
    import convert_tf_models_to_pytorch
else:
    pytest.skip("convert_tf_models_to_pytorch.py not found", allow_module_level=True)


def test_safe_extract_archive_rejects_tar_path_traversal(tmp_path):
    archive_path = tmp_path / "malicious.tar.gz"
    with tarfile.open(archive_path, "w:gz") as tar_ref:
        data = b"malicious"
        member = tarfile.TarInfo(name="../escape.txt")
        member.size = len(data)
        tar_ref.addfile(member, io.BytesIO(data))

    with pytest.raises(ValueError, match="outside"):
        convert_tf_models_to_pytorch.safe_extract_archive(archive_path, tmp_path)


def test_safe_extract_archive_rejects_zip_path_traversal(tmp_path):
    archive_path = tmp_path / "malicious.zip"
    with zipfile.ZipFile(archive_path, "w") as zip_ref:
        zip_ref.writestr("../escape.txt", "malicious")

    with pytest.raises(ValueError, match="outside"):
        convert_tf_models_to_pytorch.safe_extract_archive(archive_path, tmp_path)


def test_safe_extract_archive_rejects_zip_backslash_traversal(tmp_path):
    archive_path = tmp_path / "malicious_backslash.zip"
    with zipfile.ZipFile(archive_path, "w") as zip_ref:
        zip_ref.writestr("..\\escape.txt", "malicious")

    with pytest.raises(ValueError, match="outside"):
        convert_tf_models_to_pytorch.safe_extract_archive(archive_path, tmp_path)


def test_safe_extract_archive_rejects_tar_symlink(tmp_path):
    archive_path = tmp_path / "malicious_link.tar.gz"
    with tarfile.open(archive_path, "w:gz") as tar_ref:
        member = tarfile.TarInfo(name="link")
        member.type = tarfile.SYMTYPE
        member.linkname = "/etc/passwd"
        tar_ref.addfile(member)

    with pytest.raises(ValueError, match="link"):
        convert_tf_models_to_pytorch.safe_extract_archive(archive_path, tmp_path)
