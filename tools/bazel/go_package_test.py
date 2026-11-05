# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Validate Go bundle structure and exact Bazel-built native payloads."""

import sys
import zipfile
from pathlib import Path


def main():
    package, runtime, bridge, version_file = map(Path, sys.argv[1:5])
    version = version_file.read_text(encoding="utf-8").strip()
    root = "onnxruntime-go/"
    with zipfile.ZipFile(package) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise AssertionError("Go release bundle contains duplicate members.")
        files = {name for name in names if not name.endswith("/")}
        required = {
            root + "source/go.mod",
            root + "source/RELEASE_PACKAGE.md",
            root + "source/onnxruntime/doc.go",
            root + "source/onnxruntime/cshim.c",
            root + "source/onnxruntime/cshim.h",
            root + "source/onnxruntime/onnxruntime_c_api.h",
            root + "native/VERSION_NUMBER",
            root + "native/LICENSE",
            root + "native/ThirdPartyNotices.txt",
            root + f"native/lib/{runtime.name}",
            root + f"native/lib/{bridge.name}",
        }
        missing = required - files
        if missing:
            raise AssertionError(f"Go release bundle is missing: {sorted(missing)}")
        if archive.read(root + f"native/lib/{runtime.name}") != runtime.read_bytes():
            raise AssertionError("Go bundle runtime differs from the Bazel-built runtime.")
        if archive.read(root + f"native/lib/{bridge.name}") != bridge.read_bytes():
            raise AssertionError("Go bundle provider bridge differs from the Bazel-built bridge.")
        if archive.read(root + "native/VERSION_NUMBER").decode().strip() != version:
            raise AssertionError("Go bundle version differs from VERSION_NUMBER.")
        module = archive.read(root + "source/go.mod").decode()
        if "module github.com/microsoft/onnxruntime/go" not in module:
            raise AssertionError("Go bundle has an unexpected module path.")
        instructions = archive.read(root + "source/RELEASE_PACKAGE.md").decode()
        for value in ("replace github.com/microsoft/onnxruntime/go", "SetSharedLibraryPath", "native/lib"):
            if value not in instructions:
                raise AssertionError(f"Go instructions do not document {value}.")
    print(f"Go release bundle structure and exact native payload passed for ORT {version}.")


if __name__ == "__main__":
    main()
