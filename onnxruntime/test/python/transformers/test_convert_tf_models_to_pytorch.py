import importlib
import importlib.util
import io
import tarfile
import zipfile
from pathlib import Path

import pytest

MODULE_NAME = "convert_tf_models_to_pytorch"


def _load_convert_module():
    # Prefer the installed package. In CI this test file is copied into the build/staging directory,
    # so resolving the module path relative to __file__ no longer points at the source tree.
    try:
        return importlib.import_module(f"onnxruntime.transformers.{MODULE_NAME}")
    except ImportError:
        pass

    module_path = (
        Path(__file__).resolve().parents[4]
        / "onnxruntime"
        / "python"
        / "tools"
        / "transformers"
        / f"{MODULE_NAME}.py"
    )
    if not module_path.is_file():
        pytest.skip(f"Could not locate {MODULE_NAME}.py at {module_path}", allow_module_level=True)

    spec = importlib.util.spec_from_file_location(MODULE_NAME, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


convert_tf_models_to_pytorch = _load_convert_module()


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
