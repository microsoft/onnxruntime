# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Validate Android or Apple native NuGet payload provenance and buildTransitive wiring."""

import os
import plistlib
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path, PurePosixPath


def main():
    kind = sys.argv[1]
    package, version_file, native_asset = map(Path, sys.argv[2:5])
    version = version_file.read_text(encoding="utf-8").strip()
    expected = {
        "android": {
            "id": "Microsoft.ML.OnnxRuntime.Bazel.Android.Arm64",
            "frameworks": ["net9.0-android35.0"],
            "asset": f"runtimes/android/native/{native_asset.name}",
        },
        "apple": {
            "id": "Microsoft.ML.OnnxRuntime.Bazel.Apple",
            "frameworks": ["net9.0-ios18.0", "net9.0-maccatalyst18.0"],
            "asset": f"runtimes/apple/native/{native_asset.name}",
        },
    }[kind]
    namespace = {"n": "http://schemas.microsoft.com/packaging/2013/05/nuspec.xsd"}
    with zipfile.ZipFile(package) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise AssertionError(f"{package} contains duplicate members.")
        specs = [name for name in names if name.endswith(".nuspec")]
        if len(specs) != 1:
            raise AssertionError(f"Expected one nuspec, got {specs}.")
        metadata = ET.fromstring(archive.read(specs[0])).find("n:metadata", namespace)
        if metadata is None:
            raise AssertionError(f"{package} is missing nuspec metadata.")
        if metadata.findtext("n:id", namespaces=namespace) != expected["id"]:
            raise AssertionError(f"{package} has an incorrect package ID.")
        if metadata.findtext("n:version", namespaces=namespace) != version:
            raise AssertionError(f"{package} version must match VERSION_NUMBER ({version}).")
        release_notes = metadata.findtext("n:releaseNotes", namespaces=namespace)
        readme = archive.read("README.md").decode()
        groups = {
            group.get("targetFramework"): {
                dependency.get("id"): dependency.get("version")
                for dependency in group.findall("n:dependency", namespace)
            }
            for group in metadata.findall("n:dependencies/n:group", namespace)
        }
        expected_groups = {
            framework: {"Microsoft.ML.OnnxRuntime.Managed.Bazel": f"[{version}]"}
            for framework in expected["frameworks"]
        }
        if groups != expected_groups:
            raise AssertionError(f"{package} dependency groups must be {expected_groups}, got {groups}.")
        if archive.read(expected["asset"]) != native_asset.read_bytes():
            raise AssertionError(f"{expected['asset']} differs from source-built asset {native_asset}.")
        for framework in expected["frameworks"]:
            for root in ("build", "buildTransitive"):
                target = f"{root}/{framework}/"
                matches = [name for name in names if name.startswith(target) and name.endswith(".targets")]
                if len(matches) != 1:
                    raise AssertionError(f"Expected one targets file under {target}, got {matches}.")
                content = archive.read(matches[0]).decode()
                if kind == "android":
                    if "AndroidLibrary" not in content or "runtimes\\android\\native\\*" not in content:
                        raise AssertionError(f"{matches[0]} does not map the packaged Android AAR.")
                elif (
                    "NativeReference" not in content
                    or f"runtimes\\apple\\native\\{native_asset.name}" not in content
                    or "<Kind>Static</Kind>" not in content
                    or "<ForceLoad>True</ForceLoad>" not in content
                ):
                    raise AssertionError(f"{matches[0]} does not statically link the packaged XCFramework.")
    if kind == "android":
        if release_notes != "Android API 27 or later; arm64-v8a native runtime.":
            raise AssertionError(f"{package} has incorrect Android platform metadata: {release_notes!r}.")
        if expected["id"] not in readme or release_notes not in readme:
            raise AssertionError(f"{package} README does not state its Android package/platform scope.")
        with zipfile.ZipFile(native_asset) as archive:
            native_libraries = {name for name in archive.namelist() if name.endswith("/libonnxruntime.so")}
            expected_library = {"jni/arm64-v8a/libonnxruntime.so"}
            if native_libraries != expected_library:
                raise AssertionError(
                    f"Android package must contain exactly the ARM64 runtime {expected_library}, got {native_libraries}."
                )
    else:
        with zipfile.ZipFile(native_asset) as archive:
            names = set(archive.namelist())
            plist_names = [name for name in names if name.endswith(".xcframework/Info.plist")]
            if len(plist_names) != 1:
                raise AssertionError(f"Expected one XCFramework Info.plist, got {plist_names}.")
            info = plistlib.loads(archive.read(plist_names[0]))
            libraries_by_slice = {
                (
                    library["SupportedPlatform"],
                    library.get("SupportedPlatformVariant", ""),
                    tuple(sorted(library["SupportedArchitectures"])),
                ): library
                for library in info["AvailableLibraries"]
            }
            required_slices = {
                ("ios", "", ("arm64",)),
                ("ios", "simulator", ("arm64", "x86_64")),
                ("ios", "maccatalyst", ("arm64", "x86_64")),
            }
            if not required_slices.issubset(libraries_by_slice):
                raise AssertionError(
                    f"Apple XCFramework is missing required iOS/Mac Catalyst slices: "
                    f"{sorted(required_slices.difference(libraries_by_slice))}."
                )
            root = PurePosixPath(plist_names[0]).parent
            minimum_versions = {}
            for slice_key in required_slices:
                library = libraries_by_slice[slice_key]
                framework = root / library["LibraryIdentifier"] / library["LibraryPath"]
                framework_plist = str(framework / "Info.plist")
                if framework_plist not in names:
                    raise AssertionError(f"Apple XCFramework slice {slice_key} is missing {framework_plist}.")
                minimum_versions[slice_key] = plistlib.loads(archive.read(framework_plist))["MinimumOSVersion"]
            ios_versions = {
                minimum_versions[slice_key] for slice_key in required_slices if slice_key[1] != "maccatalyst"
            }
            if len(ios_versions) != 1:
                raise AssertionError(f"Apple XCFramework has inconsistent iOS deployment minimums {ios_versions}.")
            ios_minimum = ios_versions.pop()
            catalyst_minimum = minimum_versions[("ios", "maccatalyst", ("arm64", "x86_64"))]
            expected_release_notes = (
                f"iOS {ios_minimum} or later (device arm64; simulator arm64/x86_64); "
                f"Mac Catalyst {catalyst_minimum} or later (arm64/x86_64)."
            )
            if release_notes != expected_release_notes:
                raise AssertionError(f"{package} has incorrect Apple platform metadata: {release_notes!r}.")
            if expected["id"] not in readme or release_notes not in readme:
                raise AssertionError(f"{package} README does not state its Apple package/platform scope.")
            if sys.platform == "darwin":
                with tempfile.TemporaryDirectory() as directory:
                    archive.extractall(directory)
                    for library in info["AvailableLibraries"]:
                        slice_key = (
                            library["SupportedPlatform"],
                            library.get("SupportedPlatformVariant", ""),
                            tuple(sorted(library["SupportedArchitectures"])),
                        )
                        if slice_key not in required_slices:
                            continue
                        binary = (
                            Path(directory, plist_names[0]).parent
                            / library["LibraryIdentifier"]
                            / library["LibraryPath"]
                        )
                        if binary.suffix == ".framework":
                            binary /= binary.stem
                        result = subprocess.run(
                            ["xcrun", "nm", "-g", "-U", str(binary)],
                            capture_output=True,
                            check=False,
                            text=True,
                            env={**os.environ, "LC_ALL": "C"},
                        )
                        if result.returncode:
                            raise AssertionError(
                                f"Unable to inspect defined symbols in {binary}: xcrun nm failed with "
                                f"exit code {result.returncode}: {result.stderr.strip()}"
                            )
                        defined_symbols = {
                            line.split()[-1].removeprefix("_") for line in result.stdout.splitlines() if line
                        }
                        if "OrtGetApiBase" not in defined_symbols:
                            raise AssertionError(
                                f"{binary} does not define the ONNX Runtime C API symbol OrtGetApiBase."
                            )
    print(f"{kind} native NuGet package passed for version {version}.")


if __name__ == "__main__":
    main()
