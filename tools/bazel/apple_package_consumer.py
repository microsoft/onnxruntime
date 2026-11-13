# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Run the existing Swift inference test against an installed local release."""

import argparse
import pathlib
import platform
import re
import shutil
import subprocess
import tempfile
import zipfile


def stage_consumer(archive_path, source, model, directory):
    with zipfile.ZipFile(archive_path) as archive:
        seen = set()
        for entry in archive.infolist():
            path = pathlib.PurePosixPath(entry.filename)
            if (
                path.is_absolute()
                or ".." in path.parts
                or "\\" in entry.filename
                or not path.parts
                or path.parts[0] != "onnxruntime-objc"
                or (entry.external_attr >> 16) & 0o170000 == 0o120000
            ):
                raise ValueError(f"Unsafe installed Apple package member: {entry.filename}")
            if entry.filename in seen:
                raise ValueError(f"Duplicate installed Apple package member: {entry.filename}")
            seen.add(entry.filename)
            if entry.is_dir():
                continue
            destination = directory.joinpath(*path.parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(archive.read(entry))
    package = directory / "onnxruntime-objc"
    if not (package / "Package.swift").is_file():
        raise ValueError("Apple archive has no Package.swift; build the SwiftPM release ZIP before consuming it.")
    consumer = directory / "consumer"
    tests = consumer / "Tests" / "InstalledPackageTests"
    tests.mkdir(parents=True)
    shutil.copyfile(source, tests / "ort_swift_test.swift")
    shutil.copyfile(model, tests / "single_add.basic.ort")
    (consumer / "Package.swift").write_text(
        """// swift-tools-version: 5.9
import PackageDescription
let package = Package(
    name: "InstalledORTConsumer",
    platforms: [.macOS(.v12)],
    dependencies: [.package(path: "../onnxruntime-objc")],
    targets: [
        .testTarget(
            name: "InstalledPackageTests",
            dependencies: [.product(name: "onnxruntime_objc", package: "onnxruntime-objc")],
            resources: [.copy("single_add.basic.ort")]
        )
    ]
)
""",
        encoding="utf-8",
    )
    return consumer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=pathlib.Path)
    parser.add_argument("source", type=pathlib.Path)
    parser.add_argument("model", type=pathlib.Path)
    args = parser.parse_args()
    if platform.system() != "Darwin":
        raise RuntimeError("Installed SwiftPM inference requires a macOS Xcode host; run the Apple-only Bazel target.")
    swift = subprocess.check_output(["/usr/bin/xcrun", "--find", "swift"], text=True).strip()
    with tempfile.TemporaryDirectory(prefix="ort-installed-swift-") as temporary:
        directory = pathlib.Path(temporary)
        consumer = stage_consumer(args.archive, args.source, args.model, directory)
        result = subprocess.run(
            [
                swift,
                "test",
                "--package-path",
                str(consumer),
                "--scratch-path",
                str(directory / "build"),
                "--configuration",
                "release",
                "--jobs",
                "1",
                "--filter",
                "ORTSwiftInferenceTest.testAddInference",
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        print(result.stdout)
        result.check_returncode()
        if not re.search(r"Executed 1 test, with 0 failures", result.stdout):
            raise RuntimeError(
                "SwiftPM did not report one successful existing Add XCTest; inspect test discovery/filter."
            )


if __name__ == "__main__":
    main()
