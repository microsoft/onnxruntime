# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Check SDK contents and load only the packaged runtime after extraction."""

import ctypes
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path, PurePosixPath


def verify_runtime(library, version):
    runtime = ctypes.CDLL(library)
    runtime.OrtGetApiBase.restype = ctypes.POINTER(ctypes.c_void_p)
    base = runtime.OrtGetApiBase()
    if not base:
        raise AssertionError("Packaged OrtGetApiBase returned a null API base.")
    get_version = ctypes.CFUNCTYPE(ctypes.c_char_p)(base[1])
    actual = get_version().decode("utf-8")
    if actual != version:
        raise AssertionError(f"Packaged runtime reports {actual!r}; expected {version!r}.")


def main():
    archive_path, version_path, provider = sys.argv[1:]
    version = Path(version_path).read_text(encoding="utf-8").strip()
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        if len(set(names)) != len(names):
            raise AssertionError(f"SDK archive {archive_path} contains duplicate members.")
        for name in names:
            path = PurePosixPath(name)
            if path.is_absolute() or ".." in path.parts or path.parts[0] != "onnxruntime":
                raise AssertionError(f"SDK archive has an invalid member path: {name!r}.")
        required = {
            "onnxruntime/LICENSE",
            "onnxruntime/ThirdPartyNotices.txt",
            "onnxruntime/VERSION_NUMBER",
            "onnxruntime/include/onnxruntime_c_api.h",
            "onnxruntime/include/onnxruntime_cxx_api.h",
            "onnxruntime/include/onnxruntime_cxx_inline.h",
            "onnxruntime/include/onnxruntime_float16.h",
            "onnxruntime/include/onnxruntime_ep_c_api.h",
            "onnxruntime/examples/sdk_inference_example.cc",
            "onnxruntime/examples/add_f32.onnx",
        }
        if provider == "cuda":
            required.update(
                {
                    "onnxruntime/lib/libonnxruntime_providers_cuda.so",
                    "onnxruntime/lib/libonnxruntime_providers_shared.so",
                }
            )
        missing = required.difference(names)
        if missing:
            raise AssertionError(f"SDK archive is missing required files: {sorted(missing)}.")
        if sys.platform == "win32":
            library = "bin/onnxruntime.dll"
            if "onnxruntime/lib/onnxruntime.lib" not in names:
                raise AssertionError("Windows SDK must include lib/onnxruntime.lib for native consumers.")
        elif sys.platform == "darwin":
            library = "lib/libonnxruntime.dylib"
        else:
            library = "lib/libonnxruntime.so"
        with tempfile.TemporaryDirectory(prefix="ort-bazel-sdk-") as directory:
            archive.extractall(directory)
            root = Path(directory) / "onnxruntime"
            if (root / "VERSION_NUMBER").read_text(encoding="utf-8").strip() != version:
                raise AssertionError("Packaged VERSION_NUMBER does not match the source version.")
            # A subprocess releases the DLL before Windows removes the extraction directory.
            subprocess.run(
                [sys.executable, __file__, "--load-library", str(root / library), version],
                check=True,
            )
    print(f"SDK archive layout and extracted runtime load passed for ONNX Runtime {version}.")


if __name__ == "__main__":
    if sys.argv[1] == "--load-library":
        verify_runtime(sys.argv[2], sys.argv[3])
    else:
        main()
