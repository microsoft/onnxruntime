# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Validate first-party .crate files without requiring a native build."""

import re
import sys
import tarfile
from pathlib import Path


def _value(manifest: str, key: str) -> str:
    match = re.search(rf'(?m)^{key}\s*=\s*"([^"]+)"\s*$', manifest)
    if not match:
        raise AssertionError(f"Missing {key} in Cargo.toml.")
    return match.group(1)


def _check(path: Path):
    if path.suffix != ".crate":
        raise AssertionError(f"Unexpected crate extension: {path}")
    with tarfile.open(path, "r:gz") as archive:
        files = {member.name for member in archive if member.isfile()}
        roots = {name.split("/", 1)[0] for name in files}
        if len(roots) != 1:
            raise AssertionError(f"{path.name} does not have one package root: {roots}")
        root = roots.pop()
        manifest = archive.extractfile(f"{root}/Cargo.toml").read().decode()
        name = _value(manifest, "name")
        version = _value(manifest, "version")
        if root != f"{name}-{version}" or path.name != f"{name}-{version}.crate":
            raise AssertionError(f"{path.name} is not version-derived from its manifest.")
        required = {
            f"{root}/Cargo.toml.orig",
            f"{root}/README.md",
            f"{root}/LICENSE-APACHE",
            f"{root}/LICENSE-MIT",
            f"{root}/ORT_NATIVE_SDK.md",
            f"{root}/src/lib.rs",
        }
        missing = required - files
        if missing:
            raise AssertionError(f"{path.name} is missing {sorted(missing)}")
        if 'readme = "README.md"' not in manifest:
            raise AssertionError(f"{path.name} does not normalize its readme path.")
        if "path = " in manifest:
            raise AssertionError(f"{path.name} contains a local path dependency.")
        native_notes = archive.extractfile(f"{root}/ORT_NATIVE_SDK.md").read().decode()
        for requirement in ("ORT_RUST_STRATEGY=system", "ORT_RUST_LIB_LOCATION", "never downloads"):
            if requirement not in native_notes:
                raise AssertionError(f"{path.name} does not document external native SDK use: {requirement}")
        if any("/native/" in name or name.endswith((".dll", ".so", ".dylib", ".lib")) for name in files):
            raise AssertionError(f"{path.name} contains a platform-native payload.")
        if any(name.endswith("/bindings.rs") for name in files):
            raise AssertionError(f"{path.name} incorrectly contains target-specific generated C bindings.")
        if name == "onnxruntime-sys" and "consumer target" not in native_notes:
            raise AssertionError("onnxruntime-sys does not document consumer-target binding generation.")


def main():
    for argument in sys.argv[1:]:
        _check(Path(argument))
    print("Portable Cargo crate format validation passed.")


if __name__ == "__main__":
    main()
