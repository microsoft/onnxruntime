# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pathlib
import sys
import tempfile
import unittest
import zipfile

sys.path.insert(0, str(pathlib.Path(__file__).parents[2]))

from tools.bazel.apple_package_consumer import stage_consumer


class AppleInstalledConsumerContractTest(unittest.TestCase):
    def test_stages_existing_inference_source_against_only_the_installed_package(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            archive = root / "release.zip"
            with zipfile.ZipFile(archive, "w") as package:
                package.writestr("onnxruntime-objc/Package.swift", "// fixture package")
            source = root / "existing.swift"
            source.write_bytes(b"existing inference source")
            model = root / "existing.ort"
            model.write_bytes(b"existing model")
            consumer = stage_consumer(archive, source, model, root / "installed")
            self.assertEqual(
                (consumer / "Tests/InstalledPackageTests/ort_swift_test.swift").read_bytes(),
                source.read_bytes(),
            )
            manifest = (consumer / "Package.swift").read_text()
            self.assertIn('.package(path: "../onnxruntime-objc")', manifest)
            self.assertIn('resources: [.copy("single_add.basic.ort")]', manifest)
            self.assertNotIn("https:", manifest)

    def test_rejects_archive_path_escape(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            archive = root / "release.zip"
            with zipfile.ZipFile(archive, "w") as package:
                package.writestr("../escaped", b"invalid")
            with self.assertRaisesRegex(ValueError, "Unsafe"):
                stage_consumer(archive, root / "unused", root / "unused", root / "installed")


if __name__ == "__main__":
    unittest.main()
