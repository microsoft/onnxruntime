#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Assembles locally installable SwiftPM and CocoaPods XCFramework archives."""

import argparse
import hashlib
import json
import pathlib
import plistlib
import re
import zipfile

PACKAGE_ROOT = "onnxruntime-objc"
XCFRAMEWORK_NAME = "onnxruntime_objc.xcframework"


def _read_xcframework(xcframework_zip: pathlib.Path) -> tuple[dict, dict[str, bytes]]:
    files = {}
    with zipfile.ZipFile(xcframework_zip) as archive:
        for info in archive.infolist():
            path = pathlib.PurePosixPath(info.filename)
            if info.is_dir():
                continue
            if path.is_absolute() or ".." in path.parts or "\\" in info.filename:
                raise ValueError(f"Unsafe XCFramework archive path: {info.filename}")
            try:
                framework_index = path.parts.index(XCFRAMEWORK_NAME)
            except ValueError:
                continue
            relative_path = pathlib.PurePosixPath(*path.parts[framework_index:])
            if str(relative_path) in files:
                raise ValueError(f"Duplicate XCFramework archive member: {relative_path}")
            files[str(relative_path)] = archive.read(info)

    plist_path = f"{XCFRAMEWORK_NAME}/Info.plist"
    if plist_path not in files:
        raise ValueError(f"{xcframework_zip} does not contain {plist_path}")
    info = plistlib.loads(files[plist_path])
    libraries = info.get("AvailableLibraries", [])
    if not libraries:
        raise ValueError("XCFramework Info.plist has no AvailableLibraries")
    for library in libraries:
        identifier = library.get("LibraryIdentifier")
        library_path = library.get("LibraryPath")
        if not identifier or not library_path or not library.get("SupportedArchitectures"):
            raise ValueError("Every XCFramework library must identify its slice and architectures")
        packaged_path = f"{XCFRAMEWORK_NAME}/{identifier}/{library_path}"
        if not any(path == packaged_path or path.startswith(packaged_path + "/") for path in files):
            raise ValueError(f"XCFramework slice {identifier} is missing {library_path}")
    return info, files


def _platforms(info: dict) -> tuple[bool, bool]:
    platforms = {library["SupportedPlatform"] for library in info["AvailableLibraries"]}
    unsupported = platforms - {"ios", "macos"}
    if unsupported:
        raise ValueError(f"Unsupported platforms in Objective-C package: {sorted(unsupported)}")
    return "ios" in platforms, "macos" in platforms


def _package_swift(version: str, info: dict, framework_files: dict[str, bytes]) -> dict[str, bytes]:
    has_ios, has_macos = _platforms(info)
    declared_platforms = []
    if has_ios:
        declared_platforms.append('        .iOS("15.1")')
    if has_macos:
        declared_platforms.append("        .macOS(.v12)")
    if any(library.get("SupportedPlatformVariant") == "maccatalyst" for library in info["AvailableLibraries"]):
        declared_platforms.append('        .macCatalyst("15.1")')
    platform_declarations = ",\n".join(declared_platforms)
    package_swift = f"""// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "onnxruntime-objc",
    platforms: [
{platform_declarations}
    ],
    products: [
        .library(name: "onnxruntime_objc", targets: ["onnxruntime_objc", "OnnxRuntimeLinker"])
    ],
    targets: [
        .binaryTarget(name: "onnxruntime_objc", path: "{XCFRAMEWORK_NAME}"),
        .target(
            name: "OnnxRuntimeLinker",
            dependencies: ["onnxruntime_objc"],
            linkerSettings: [.linkedLibrary("c++")]
        )
    ]
)
"""
    files = {f"{PACKAGE_ROOT}/Package.swift": package_swift.encode()}
    files[f"{PACKAGE_ROOT}/Sources/OnnxRuntimeLinker/Linker.swift"] = b"import onnxruntime_objc\n"
    files.update({f"{PACKAGE_ROOT}/{path}": content for path, content in framework_files.items()})
    files[f"{PACKAGE_ROOT}/VERSION"] = (version + "\n").encode()
    return files


def _package_cocoapods(version: str, info: dict, framework_files: dict[str, bytes]) -> dict[str, bytes]:
    has_ios, has_macos = _platforms(info)
    deployment_targets = []
    if has_ios:
        deployment_targets.append('  s.ios.deployment_target = "15.1"')
    if has_macos:
        deployment_targets.append('  s.osx.deployment_target = "12.0"')
    podspec = f"""Pod::Spec.new do |s|
  s.name = "onnxruntime-objc"
  s.version = "{version}"
  s.summary = "ONNX Runtime Objective-C API"
  s.description = "Source-built ONNX Runtime Objective-C XCFramework."
  s.homepage = "https://github.com/microsoft/onnxruntime"
  s.license = {{ :type => "MIT", :file => "LICENSE" }}
  s.author = {{ "ONNX Runtime" => "onnxruntime@microsoft.com" }}
  s.source = {{ :http => "file:///local/onnxruntime-objc.zip" }}
{chr(10).join(deployment_targets)}
  s.static_framework = true
  s.libraries = "c++"
  s.vendored_frameworks = "{XCFRAMEWORK_NAME}"
end
"""
    files = {f"{PACKAGE_ROOT}/onnxruntime-objc.podspec": podspec.encode()}
    files.update({f"{PACKAGE_ROOT}/{path}": content for path, content in framework_files.items()})
    return files


def _write_zip(output: pathlib.Path, files: dict[str, bytes]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path, content in sorted(files.items()):
            entry = zipfile.ZipInfo(path, date_time=(1980, 1, 1, 0, 0, 0))
            entry.external_attr = 0o644 << 16
            entry.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(entry, content)


def assemble(args: argparse.Namespace) -> None:
    version = pathlib.Path(args.version_file).read_text(encoding="utf-8").strip()
    if not re.fullmatch(r"\d+\.\d+\.\d+(?:[-.][A-Za-z0-9.]+)?", version):
        raise ValueError(f"Invalid release version {version!r}; supply a numeric VERSION_NUMBER.")
    info, framework_files = _read_xcframework(pathlib.Path(args.xcframework))
    common = {
        f"{PACKAGE_ROOT}/LICENSE": pathlib.Path(args.license).read_bytes(),
        f"{PACKAGE_ROOT}/ThirdPartyNotices.txt": pathlib.Path(args.notice).read_bytes(),
        f"{PACKAGE_ROOT}/README.md": (
            b"# Local ONNX Runtime Apple distribution\n\n"
            b"Extract this archive before use. Add the extracted directory as a local Swift "
            b"package, or use `pod 'onnxruntime-objc', :path => '/absolute/path/onnxruntime-objc'` "
            b"in your Podfile. The podspec source URL is not a published download location; "
            b"registry publication/signing are not performed by this target.\n"
        ),
    }
    common[f"{PACKAGE_ROOT}/onnxruntime-release.json"] = (
        json.dumps(
            {
                "schema_version": 1,
                "language": "objective-c",
                "version": version,
                "distribution": "local-xcframework",
                "slices": info["AvailableLibraries"],
                "files": {name: hashlib.sha256(data).hexdigest() for name, data in sorted(framework_files.items())},
            },
            indent=2,
            sort_keys=True,
        ).encode()
        + b"\n"
    )
    _write_zip(
        pathlib.Path(args.swift_output),
        _package_swift(version, info, framework_files) | common,
    )
    _write_zip(
        pathlib.Path(args.cocoapods_output),
        _package_cocoapods(version, info, framework_files) | common,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xcframework", required=True)
    parser.add_argument("--version-file", required=True)
    parser.add_argument("--license", required=True)
    parser.add_argument("--notice", required=True)
    parser.add_argument("--swift-output", required=True)
    parser.add_argument("--cocoapods-output", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    assemble(parse_args())
