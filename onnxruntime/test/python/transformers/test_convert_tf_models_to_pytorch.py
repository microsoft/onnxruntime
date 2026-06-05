import importlib.util
import io
import tarfile
import zipfile
from pathlib import Path

import pytest

# Resolve the module path relative to this file so the test works regardless of CWD.
MODULE_PATH = (
    Path(__file__).resolve().parents[4]
    / "onnxruntime"
    / "python"
    / "tools"
    / "transformers"
    / "convert_tf_models_to_pytorch.py"
)
SPEC = importlib.util.spec_from_file_location("convert_tf_models_to_pytorch", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
convert_tf_models_to_pytorch = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(convert_tf_models_to_pytorch)


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
