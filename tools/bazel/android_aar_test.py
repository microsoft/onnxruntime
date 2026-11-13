# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Host-only AAR assembly regressions (not Android device execution)."""

import pathlib
import tempfile
import unittest
import zipfile
from xml.etree import ElementTree

from android_aar import package


class AndroidAarTest(unittest.TestCase):
    def test_archive_layout_and_reproducibility(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            for name in ("classes.jar", "runtime.so", "jni.so", "LICENSE", "api.h"):
                (root / name).write_bytes(name.encode())
            output = root / "runtime.aar"
            args = [root / name for name in ("classes.jar", "runtime.so", "jni.so")]
            assets = [root / "LICENSE", root / "api.h"]
            package(output, *args, assets)
            first = output.read_bytes()
            package(output, *args, assets)
            self.assertEqual(first, output.read_bytes())
            with zipfile.ZipFile(output) as archive:
                self.assertEqual(
                    set(archive.namelist()),
                    {
                        "AndroidManifest.xml",
                        "R.txt",
                        "proguard.txt",
                        "classes.jar",
                        "jni/arm64-v8a/libonnxruntime.so",
                        "jni/arm64-v8a/libonnxruntime4j_jni.so",
                        "META-INF/LICENSE",
                        "headers/api.h",
                    },
                )
                self.assertEqual(archive.read("classes.jar"), b"classes.jar")
                self.assertEqual(archive.read("jni/arm64-v8a/libonnxruntime.so"), b"runtime.so")
                self.assertEqual(archive.read("jni/arm64-v8a/libonnxruntime4j_jni.so"), b"jni.so")
                manifest = ElementTree.fromstring(archive.read("AndroidManifest.xml"))
                self.assertEqual(manifest.attrib["package"], "ai.onnxruntime")
                self.assertEqual(
                    manifest.find("uses-sdk").attrib["{http://schemas.android.com/apk/res/android}minSdkVersion"],
                    "27",
                )
                self.assertIn(b"ai.onnxruntime.**", archive.read("proguard.txt"))

    def test_missing_native_library_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            classes = root / "classes.jar"
            classes.write_bytes(b"classes")
            with self.assertRaises(FileNotFoundError):
                package(root / "runtime.aar", classes, root / "missing.so", root / "jni.so", [])
            self.assertFalse((root / "runtime.aar").exists())

    def test_duplicate_assets_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            source = root / "api.h"
            source.write_bytes(b"header")
            with self.assertRaisesRegex(ValueError, "Duplicate Android archive entry headers/api.h"):
                package(root / "runtime.aar", source, source, source, [source, source])


if __name__ == "__main__":
    unittest.main()
