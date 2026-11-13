# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Validate Rust bundle structure and exact Bazel-built native payloads."""

import sys
import zipfile
from pathlib import Path


def main():
    package, runtime, bridge, version_file = map(Path, sys.argv[1:5])
    version = version_file.read_text(encoding="utf-8").strip()
    root = "onnxruntime-rust/"
    with zipfile.ZipFile(package) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise AssertionError("Rust release bundle contains duplicate members.")
        files = {name for name in names if not name.endswith("/")}
        required = {
            root + "source/Cargo.toml",
            root + "source/Cargo.lock",
            root + "source/README.md",
            root + "source/RELEASE_PACKAGE.md",
            root + "source/onnxruntime/Cargo.toml",
            root + "source/onnxruntime/src/lib.rs",
            root + "source/onnxruntime-sys/Cargo.toml",
            root + "source/onnxruntime-sys/build.rs",
            root + "source/onnxruntime-sys/src/lib.rs",
            root + "native/VERSION_NUMBER",
            root + "native/LICENSE",
            root + "native/ThirdPartyNotices.txt",
            root + f"native/lib/{runtime.name}",
            root + f"native/lib/{bridge.name}",
        }
        missing = required - files
        if missing:
            raise AssertionError(f"Rust release bundle is missing: {sorted(missing)}")
        api_headers = [name for name in files if name.endswith("/onnxruntime_c_api.h")]
        if len(api_headers) != 1:
            raise AssertionError(f"Rust release bundle must contain exactly one C API header: {api_headers}")
        if archive.read(root + f"native/lib/{runtime.name}") != runtime.read_bytes():
            raise AssertionError("Rust bundle runtime differs from the Bazel-built runtime.")
        if archive.read(root + f"native/lib/{bridge.name}") != bridge.read_bytes():
            raise AssertionError("Rust bundle provider bridge differs from the Bazel-built bridge.")
        if archive.read(root + "native/VERSION_NUMBER").decode().strip() != version:
            raise AssertionError("Rust bundle version differs from VERSION_NUMBER.")
        cargo = archive.read(root + "source/onnxruntime/Cargo.toml").decode()
        if 'onnxruntime-sys = { version = "0.0.14", path = "../onnxruntime-sys" }' not in cargo:
            raise AssertionError("Rust high-level crate must retain its local low-level crate dependency.")
        instructions = archive.read(root + "source/RELEASE_PACKAGE.md").decode()
        for value in ("ORT_RUST_STRATEGY=system", "ORT_RUST_LIB_LOCATION", "ORT_RUST_HEADER_LOCATION"):
            if value not in instructions:
                raise AssertionError(f"Rust instructions do not document {value}.")
    print(f"Rust release bundle structure and exact native payload passed for ORT {version}.")


if __name__ == "__main__":
    main()
