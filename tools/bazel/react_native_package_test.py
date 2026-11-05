# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Regression tests for the Android React Native package assembler."""

import argparse
import io
import json
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path

from react_native_package import package


class PackageTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        for name, data in {
            "classes.jar": b"compiled-java-payload",
            "libonnxruntime.so": b"ort-payload",
            "libonnxruntimejsi.so": b"adapter-payload",
            "libc++_shared.so": b"ndk-payload",
            "version": b"1.32.0\n",
            "index.ts": b'export const version = "1.32.0";',
            "index.js": b'exports.version = "1.32.0";',
            "LICENSE": b"license-payload",
        }.items():
            (self.root / name).write_bytes(data)
        self.args = argparse.Namespace(
            aar=str(self.root / "output.aar"),
            npm=str(self.root / "output.tgz"),
            classes=str(self.root / "classes.jar"),
            version=str(self.root / "version"),
            native=[
                str(self.root / name) for name in ("libonnxruntime.so", "libonnxruntimejsi.so", "libc++_shared.so")
            ],
            frontend=[str(self.root / "index.ts")],
            common=[str(self.root / "index.js")],
            notice=[str(self.root / "LICENSE")],
        )

    def test_exact_java_and_native_payloads(self):
        package(self.args)
        with zipfile.ZipFile(self.args.aar) as archive:
            self.assertEqual(archive.read("classes.jar"), b"compiled-java-payload")
            for path in self.args.native:
                self.assertEqual(archive.read("jni/arm64-v8a/" + Path(path).name), Path(path).read_bytes())
            self.assertIn(b"ai.onnxruntime.reactnative", archive.read("proguard.txt"))
        with tarfile.open(self.args.npm) as archive:
            self.assertEqual(
                archive.extractfile("package/android/onnxruntime-react-native.aar").read(),
                Path(self.args.aar).read_bytes(),
            )
            metadata = json.load(archive.extractfile("package/package.json"))
            self.assertNotIn("scripts", metadata)
            self.assertEqual(metadata["peerDependencies"]["react-native"], "0.73.11")
            self.assertIsNotNone(archive.extractfile("package/node_modules/onnxruntime-common/lib/index.js"))

    def test_reproducible_archives(self):
        package(self.args)
        first = Path(self.args.npm).read_bytes()
        package(self.args)
        self.assertEqual(first, Path(self.args.npm).read_bytes())
        with tarfile.open(fileobj=io.BytesIO(first)) as archive:
            self.assertTrue(all(member.mtime == 0 for member in archive))

    def test_missing_native_library_is_an_error(self):
        self.args.native.pop()
        with self.assertRaisesRegex(ValueError, "requires exactly"):
            package(self.args)


if __name__ == "__main__":
    unittest.main()
