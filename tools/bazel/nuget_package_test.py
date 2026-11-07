# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Validate managed and honest per-RID NuGet metadata and exact source-built payload bytes."""

import sys
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path


def read_spec(archive):
    names = [name for name in archive.namelist() if name.endswith(".nuspec")]
    if len(names) != 1:
        raise AssertionError(f"Expected one nuspec, got {names}.")
    namespace = {"n": "http://schemas.microsoft.com/packaging/2013/05/nuspec.xsd"}
    spec = ET.fromstring(archive.read(names[0]))
    metadata = spec.find("n:metadata", namespace)
    if metadata is None:
        raise AssertionError("NuGet nuspec is missing its metadata section.")
    return metadata, namespace


def main():
    managed_package, native_package, version_file = map(Path, sys.argv[1:4])
    assembly_outputs = {
        "net8.0": [Path(sys.argv[4]), Path(sys.argv[5])],
        "netstandard2.0": [Path(sys.argv[6]), Path(sys.argv[7])],
        "net9.0-android35.0": [Path(sys.argv[8]), Path(sys.argv[9])],
        "net9.0-ios18.0": [Path(sys.argv[10]), Path(sys.argv[11])],
        "net9.0-maccatalyst18.0": [Path(sys.argv[12]), Path(sys.argv[13])],
    }
    runtime = Path(sys.argv[14])
    bridge = Path(sys.argv[15])
    remaining = list(map(Path, sys.argv[16:-2]))
    rid, flavor = sys.argv[-2:]
    version = version_file.read_text(encoding="utf-8").strip()
    with zipfile.ZipFile(managed_package) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise AssertionError(f"Managed NuGet package {managed_package} contains duplicate members.")
        metadata, namespace = read_spec(archive)
        if metadata.findtext("n:id", namespaces=namespace) != "Microsoft.ML.OnnxRuntime.Managed.Bazel":
            raise AssertionError("Managed NuGet package ID is incorrect.")
        if metadata.findtext("n:version", namespaces=namespace) != version:
            raise AssertionError(f"Managed NuGet version must match VERSION_NUMBER ({version}).")
        groups = {
            group.get("targetFramework"): {
                dependency.get("id"): dependency.get("version")
                for dependency in group.findall("n:dependency", namespace)
            }
            for group in metadata.findall("n:dependencies/n:group", namespace)
        }
        expected_groups = {
            "netstandard2.0": {"System.Memory": "[4.5.5]"},
            "net8.0": {
                "System.Memory": "[4.5.5]",
                "System.Numerics.Tensors": "[9.0.0]",
            },
        }
        for framework in ("net9.0-android35.0", "net9.0-ios18.0", "net9.0-maccatalyst18.0"):
            expected_groups[framework] = {
                "System.Memory": "[4.5.5]",
                "System.Numerics.Tensors": "[9.0.0]",
            }
        if groups != expected_groups:
            raise AssertionError(f"Managed NuGet dependency groups must be {expected_groups}, got {groups}.")
        for framework, sources in assembly_outputs.items():
            for source in sources:
                name = f"lib/{framework}/{source.name}"
                if archive.read(name) != source.read_bytes():
                    raise AssertionError(f"Managed member {name} differs from source-built output {source}.")
    with zipfile.ZipFile(native_package) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise AssertionError(f"Native NuGet package {native_package} contains duplicate members.")
        metadata, namespace = read_spec(archive)
        expected_id = f"Microsoft.ML.OnnxRuntime.Bazel.{flavor.capitalize()}.{rid}"
        if metadata.findtext("n:id", namespaces=namespace) != expected_id:
            raise AssertionError(f"Native package ID must identify flavor and RID ({expected_id}).")
        if metadata.findtext("n:version", namespaces=namespace) != version:
            raise AssertionError(f"Native NuGet version must match VERSION_NUMBER ({version}).")
        groups = {
            group.get("targetFramework"): {
                dependency.get("id"): dependency.get("version")
                for dependency in group.findall("n:dependency", namespace)
            }
            for group in metadata.findall("n:dependencies/n:group", namespace)
        }
        expected_groups = {
            framework: {"Microsoft.ML.OnnxRuntime.Managed.Bazel": f"[{version}]"}
            for framework in ("netstandard2.0", "net8.0")
        }
        if groups != expected_groups:
            raise AssertionError(
                f"Native NuGet must depend on the exact matching managed package for both frameworks; got {groups}."
            )
        for source in [runtime, bridge, *remaining]:
            name = f"runtimes/{rid}/native/{source.name}"
            if archive.read(name) != source.read_bytes():
                raise AssertionError(f"Native member {name} differs from source-built output {source}.")
        required = {
            "LICENSE",
            "ThirdPartyNotices.txt",
            "Privacy.md",
            "README.md",
            "ORT_icon_for_light_bg.png",
            "build/native/Microsoft.ML.OnnxRuntime.Bazel.props",
            "build/native/Microsoft.ML.OnnxRuntime.Bazel.targets",
        }
        missing = required.difference(names)
        if missing:
            raise AssertionError(f"Native package is missing release metadata files: {sorted(missing)}.")
        if not any(name.startswith("build/native/include/onnxruntime/core/session/") for name in names):
            raise AssertionError("Native package is missing public C API headers.")
    print(f"Managed and {flavor} per-RID NuGet packages passed for {rid}, version {version}.")


if __name__ == "__main__":
    main()
