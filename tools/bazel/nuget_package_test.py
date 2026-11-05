# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Validate NuGet metadata, RID paths, and exact source-built payload bytes."""

import sys
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path


def main():
    package, version_file, assembly, runtime, bridge = map(Path, sys.argv[1:6])
    rid = sys.argv[6]
    version = version_file.read_text(encoding="utf-8").strip()
    with zipfile.ZipFile(package) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise AssertionError(f"NuGet package {package} contains duplicate members.")
        expected = {
            "lib/net8.0/Microsoft.ML.OnnxRuntime.dll": assembly,
            f"runtimes/{rid}/native/{runtime.name}": runtime,
            f"runtimes/{rid}/native/{bridge.name}": bridge,
        }
        for name, source in expected.items():
            if archive.read(name) != source.read_bytes():
                raise AssertionError(f"NuGet member {name} differs from source-built output {source}.")
        for name in ("[Content_Types].xml", "_rels/.rels"):
            ET.fromstring(archive.read(name))
        namespace = {"n": "http://schemas.microsoft.com/packaging/2013/05/nuspec.xsd"}
        spec = ET.fromstring(archive.read("Microsoft.ML.OnnxRuntime.Bazel.nuspec"))
        metadata = spec.find("n:metadata", namespace)
        if metadata is None:
            raise AssertionError("NuGet nuspec is missing its metadata section.")
        if metadata.findtext("n:id", namespaces=namespace) != "Microsoft.ML.OnnxRuntime.Bazel":
            raise AssertionError("NuGet must use the experimental Microsoft.ML.OnnxRuntime.Bazel package ID.")
        if metadata.findtext("n:version", namespaces=namespace) != version:
            raise AssertionError(f"NuGet version must match VERSION_NUMBER ({version}).")
        group = metadata.find("n:dependencies/n:group", namespace)
        if group is None:
            raise AssertionError("NuGet nuspec must declare its net8.0 dependency group.")
        dependencies = [(item.get("id"), item.get("version")) for item in group]
        if group.get("targetFramework") != "net8.0" or dependencies != [("System.Numerics.Tensors", "[9.0.0]")]:
            raise AssertionError("NuGet must declare its net8.0 System.Numerics.Tensors 9.0.0 dependency.")
        required = set(expected) | {
            "[Content_Types].xml",
            "_rels/.rels",
            "Microsoft.ML.OnnxRuntime.Bazel.nuspec",
            "LICENSE",
            "ThirdPartyNotices.txt",
        }
        files = {name for name in names if not name.endswith("/")}
        if files != required:
            raise AssertionError(f"Unexpected NuGet members: {sorted(files.symmetric_difference(required))}.")
    print(f"Native NuGet metadata and exact source-built payload passed for {rid}, version {version}.")


if __name__ == "__main__":
    main()
