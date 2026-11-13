# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Validate optimized portable symbols remain separate from the managed NuGet package."""

import sys
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path


def package_metadata(archive):
    names = [name for name in archive.namelist() if name.endswith(".nuspec")]
    if len(names) != 1:
        raise AssertionError(f"Expected one nuspec, got {names}.")
    namespace = {"n": "http://schemas.microsoft.com/packaging/2013/05/nuspec.xsd"}
    metadata = ET.fromstring(archive.read(names[0])).find("n:metadata", namespace)
    if metadata is None:
        raise AssertionError("NuGet package is missing nuspec metadata.")
    return metadata, namespace


def main():
    managed_package, symbols_package, version_file = map(Path, sys.argv[1:4])
    pdbs = {
        "net8.0": Path(sys.argv[4]),
        "netstandard2.0": Path(sys.argv[5]),
        "net9.0-android35.0": Path(sys.argv[10]),
        "net9.0-ios18.0": Path(sys.argv[13]),
        "net9.0-maccatalyst18.0": Path(sys.argv[16]),
    }
    managed_outputs = {
        "net8.0": [Path(sys.argv[6]), Path(sys.argv[7])],
        "netstandard2.0": [Path(sys.argv[8]), Path(sys.argv[9])],
        "net9.0-android35.0": [Path(sys.argv[11]), Path(sys.argv[12])],
        "net9.0-ios18.0": [Path(sys.argv[14]), Path(sys.argv[15])],
        "net9.0-maccatalyst18.0": [Path(sys.argv[17]), Path(sys.argv[18])],
    }
    version = version_file.read_text(encoding="utf-8").strip()
    for pdb in pdbs.values():
        if not pdb.read_bytes().startswith(b"BSJB"):
            raise AssertionError(f"{pdb} is not a portable PDB (missing BSJB metadata signature).")
    with zipfile.ZipFile(managed_package) as archive:
        if any(name.endswith(".pdb") for name in archive.namelist()):
            raise AssertionError("The normal managed NuGet package must remain PDB-free.")
        metadata, namespace = package_metadata(archive)
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
            raise AssertionError(f"Managed package dependency groups must be {expected_groups}, got {groups}.")
        for framework, outputs in managed_outputs.items():
            for output in outputs:
                member = f"lib/{framework}/{output.name}"
                if archive.read(member) != output.read_bytes():
                    raise AssertionError(f"Managed member {member} differs from the rules_dotnet output {output}.")
    with zipfile.ZipFile(symbols_package) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise AssertionError(f"Symbols package {symbols_package} contains duplicate members.")
        metadata, namespace = package_metadata(archive)
        if metadata.findtext("n:id", namespaces=namespace) != "Microsoft.ML.OnnxRuntime.Managed.Bazel":
            raise AssertionError("Symbols package ID must match the managed package ID.")
        if metadata.findtext("n:version", namespaces=namespace) != version:
            raise AssertionError(f"Symbols package version must match VERSION_NUMBER ({version}).")
        for field in ("license", "readme", "icon"):
            if metadata.find(f"n:{field}", namespace) is not None:
                raise AssertionError(f"Symbols package must not reference omitted {field} metadata assets.")
        omitted_docs = {"LICENSE", "README.md", "ORT_icon_for_light_bg.png"}
        if omitted_docs.intersection(names):
            raise AssertionError("Symbols package must keep documentation and images in the normal NuGet package.")
        members = {f"lib/{framework}/Microsoft.ML.OnnxRuntime.pdb": pdb for framework, pdb in pdbs.items()}
        for member, pdb in members.items():
            if archive.read(member) != pdb.read_bytes():
                raise AssertionError(f"Symbols member {member} differs from the Roslyn output {pdb}.")
        payloads = {name for name in names if name.startswith("lib/")}
        if payloads != set(members):
            raise AssertionError(
                f"Symbols package must contain only the framework portable PDB payloads, got {sorted(payloads)}."
            )
    print(f"Separate optimized portable symbols package passed for version {version}.")


if __name__ == "__main__":
    main()
