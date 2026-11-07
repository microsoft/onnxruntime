#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import hashlib
import json
import pathlib
import plistlib
import sys
import tempfile
import unittest
import zipfile

sys.path.insert(0, str(pathlib.Path(__file__).parents[2]))

from tools.bazel import apple_package


class ApplePackageTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.temp_dir.name)
        self.xcframework = self.root / "input.zip"
        info = {
            "AvailableLibraries": [
                {
                    "LibraryIdentifier": "ios-arm64",
                    "LibraryPath": "onnxruntime_objc.framework",
                    "SupportedArchitectures": ["arm64"],
                    "SupportedPlatform": "ios",
                },
                {
                    "LibraryIdentifier": "macos-arm64_x86_64",
                    "LibraryPath": "onnxruntime_objc.framework",
                    "SupportedArchitectures": ["arm64", "x86_64"],
                    "SupportedPlatform": "macos",
                },
            ],
            "CFBundlePackageType": "XFWK",
            "XCFrameworkFormatVersion": "1.0",
        }
        with zipfile.ZipFile(self.xcframework, "w") as archive:
            archive.writestr("onnxruntime_objc.xcframework/Info.plist", plistlib.dumps(info))
            archive.writestr(
                "onnxruntime_objc.xcframework/ios-arm64/onnxruntime_objc.framework/onnxruntime_objc",
                b"native-library",
            )
            archive.writestr(
                "onnxruntime_objc.xcframework/macos-arm64_x86_64/onnxruntime_objc.framework/onnxruntime_objc",
                b"native-library",
            )
        for name, content in [
            ("VERSION_NUMBER", b"1.32.0\n"),
            ("LICENSE", b"MIT\n"),
            ("ThirdPartyNotices.txt", b"notices\n"),
        ]:
            (self.root / name).write_bytes(content)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_packages_reference_the_bundled_xcframework(self):
        swift_output = self.root / "swift.zip"
        pod_output = self.root / "pod.zip"
        apple_package.assemble(
            argparse.Namespace(
                xcframework=self.xcframework,
                version_file=self.root / "VERSION_NUMBER",
                license=self.root / "LICENSE",
                notice=self.root / "ThirdPartyNotices.txt",
                swift_output=swift_output,
                cocoapods_output=pod_output,
            )
        )

        with zipfile.ZipFile(swift_output) as archive:
            package = archive.read("onnxruntime-objc/Package.swift").decode()
            self.assertIn('.binaryTarget(name: "onnxruntime_objc", path: "onnxruntime_objc.xcframework")', package)
            self.assertIn('.iOS("15.1")', package)
            self.assertIn(".macOS(.v12)", package)
            self.assertIn('.linkedLibrary("c++")', package)
            self.assertEqual(
                archive.read("onnxruntime-objc/Sources/OnnxRuntimeLinker/Linker.swift"),
                b"import onnxruntime_objc\n",
            )
            self.assertEqual(archive.read("onnxruntime-objc/VERSION"), b"1.32.0\n")
            manifest = json.loads(archive.read("onnxruntime-objc/onnxruntime-release.json"))
            self.assertEqual(manifest["version"], "1.32.0")
            self.assertEqual(manifest["distribution"], "local-xcframework")
            for name, digest in manifest["files"].items():
                self.assertEqual(hashlib.sha256(archive.read("onnxruntime-objc/" + name)).hexdigest(), digest)
            self.assertEqual(
                archive.read(
                    "onnxruntime-objc/onnxruntime_objc.xcframework/ios-arm64/"
                    "onnxruntime_objc.framework/onnxruntime_objc"
                ),
                b"native-library",
            )

        with zipfile.ZipFile(pod_output) as archive:
            podspec = archive.read("onnxruntime-objc/onnxruntime-objc.podspec").decode()
            self.assertIn('s.version = "1.32.0"', podspec)
            self.assertIn('s.vendored_frameworks = "onnxruntime_objc.xcframework"', podspec)
            self.assertIn('s.libraries = "c++"', podspec)
            self.assertIn('s.ios.deployment_target = "15.1"', podspec)
            self.assertIn('s.osx.deployment_target = "12.0"', podspec)
            self.assertEqual(archive.read("onnxruntime-objc/LICENSE"), b"MIT\n")
            self.assertEqual(archive.read("onnxruntime-objc/ThirdPartyNotices.txt"), b"notices\n")

    def test_rejects_manifest_slice_without_a_library(self):
        broken = self.root / "broken.zip"
        with zipfile.ZipFile(self.xcframework) as source, zipfile.ZipFile(broken, "w") as destination:
            for entry in source.infolist():
                if "macos-arm64_x86_64/onnxruntime_objc.framework/onnxruntime_objc" not in entry.filename:
                    destination.writestr(entry, source.read(entry))

        with self.assertRaisesRegex(ValueError, "macos-arm64_x86_64"):
            apple_package._read_xcframework(broken)

    def test_swift_declares_catalyst_only_for_a_present_slice(self):
        info, files = apple_package._read_xcframework(self.xcframework)
        without_catalyst = apple_package._package_swift("1.32.0", info, files)
        self.assertNotIn(b".macCatalyst", without_catalyst["onnxruntime-objc/Package.swift"])
        info["AvailableLibraries"].append(
            {
                "SupportedPlatform": "ios",
                "SupportedPlatformVariant": "maccatalyst",
            }
        )
        with_catalyst = apple_package._package_swift("1.32.0", info, files)
        self.assertIn(b'.macCatalyst("15.1")', with_catalyst["onnxruntime-objc/Package.swift"])

    def test_rejects_unsafe_members(self):
        broken = self.root / "unsafe.zip"
        with zipfile.ZipFile(broken, "w") as archive:
            archive.writestr("../outside", b"invalid")
        with self.assertRaisesRegex(ValueError, "Unsafe"):
            apple_package._read_xcframework(broken)

    def test_rejects_version_injection(self):
        (self.root / "VERSION_NUMBER").write_text('1.0.0"; arbitrary-code', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Invalid release version"):
            apple_package.assemble(argparse.Namespace(version_file=self.root / "VERSION_NUMBER"))


if __name__ == "__main__":
    unittest.main()
