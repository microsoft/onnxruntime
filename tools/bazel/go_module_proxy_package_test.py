# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Validate Go module proxy artifacts without a native build."""

import json
import re
import sys
import zipfile
from pathlib import Path


def main():
    proxy_root = Path(sys.argv[1])
    artifacts = list(proxy_root.glob("github.com/microsoft/onnxruntime/go/@v/*"))
    mod_files = [path for path in artifacts if path.suffix == ".mod"]
    if len(mod_files) != 1:
        raise AssertionError(f"Expected one versioned .mod file, found {mod_files}.")
    mod_path = mod_files[0]
    info_path = mod_path.with_suffix(".info")
    zip_path = mod_path.with_suffix(".zip")
    module_text = mod_path.read_text(encoding="utf-8")
    module = re.search(r"(?m)^module\s+(\S+)\s*$", module_text).group(1)
    version = mod_path.stem
    if info_path.name != f"{version}.info" or zip_path.name != f"{version}.zip":
        raise AssertionError("Go proxy artifacts do not share a version-derived basename.")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    if info != {"Version": version, "Time": "1970-01-01T00:00:00Z"}:
        raise AssertionError(f"Unexpected Go proxy info: {info}")
    root = f"{module}@{version}/"
    with zipfile.ZipFile(zip_path) as archive:
        names = archive.namelist()
        if not names or any(not name.startswith(root) for name in names):
            raise AssertionError("Go module ZIP does not use the canonical module@version root.")
        required = {
            root + "go.mod",
            root + "LICENSE",
            root + "ThirdPartyNotices.txt",
            root + "onnxruntime/doc.go",
            root + "onnxruntime/cshim.c",
            root + "onnxruntime/cshim.h",
            root + "onnxruntime/onnxruntime_c_api.h",
        }
        missing = required - set(names)
        if missing:
            raise AssertionError(f"Go module ZIP is missing {sorted(missing)}")
        if archive.read(root + "go.mod").decode() != module_text:
            raise AssertionError("The proxy .mod and ZIP go.mod differ.")
        if any(name.endswith((".dll", ".so", ".dylib", ".lib")) for name in names):
            raise AssertionError("Go module proxy ZIP contains a platform-native payload.")
    print("Portable Go module proxy format validation passed.")


if __name__ == "__main__":
    main()
