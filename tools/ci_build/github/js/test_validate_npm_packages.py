# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import io
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path


class ValidateNpmPackagesTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for name in ("node", "web", "react-native"):
            (self.root / name).mkdir()
        self.version = "1.32.0"
        self.write_package("web", "onnxruntime-web", {})
        self.write_package("node", "onnxruntime-common", {})
        self.write_package("web", "onnxruntime-common", {})

    def write_package(self, directory, name, fields, version=None):
        version = version or self.version
        manifest = {"name": name, "version": version, **fields}
        content = json.dumps(manifest).encode()
        with tarfile.open(self.root / directory / f"{name}-{version}.tgz", "w:gz") as archive:
            member = tarfile.TarInfo("package/package.json")
            member.size = len(content)
            archive.addfile(member, io.BytesIO(content))

    def write_split_packages(self):
        dependencies = {}
        for platform, arch in (("linux", "x64"), ("darwin", "arm64"), ("win32", "x64")):
            name = f"onnxruntime-node-{platform}-{arch}"
            dependencies[name] = self.version
            fields = {"os": [platform], "cpu": [arch]}
            if platform == "linux":
                fields["libc"] = ["glibc"]
            self.write_package("node", name, fields)
        self.write_package("node", "onnxruntime-node", {"optionalDependencies": dependencies})

    def validate(self):
        return subprocess.run(
            [
                sys.executable,
                str(Path(__file__).with_name("validate-npm-packages.py")),
                str(self.root / "node"),
                str(self.root / "web"),
                str(self.root / "react-native"),
                "refs/heads/rel-1.32.0",
                "latest",
            ],
            env={**os.environ, "RELEASE_NODE": "1", "RELEASE_WEB": "1", "RELEASE_REACT_NATIVE": "0"},
            check=False,
            capture_output=True,
            text=True,
        )

    def test_validate_accepts_legacy_bundled_package(self):
        self.write_package("node", "onnxruntime-node", {})
        result = self.validate()
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_validate_accepts_complete_split_packages(self):
        self.write_split_packages()
        result = self.validate()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("ort_node_ver=1.32.0", result.stdout)

    def test_validate_rejects_missing_native_archive(self):
        self.write_split_packages()
        (self.root / "node" / f"onnxruntime-node-linux-x64-{self.version}.tgz").unlink()
        result = self.validate()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("do not match parent optional dependencies", result.stderr)

    def test_validate_rejects_all_missing_native_archives(self):
        self.write_split_packages()
        for filename in (self.root / "node").glob("onnxruntime-node-*-*-*.tgz"):
            filename.unlink()
        result = self.validate()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Native archives are missing", result.stderr)

    def test_validate_rejects_native_version_mismatch(self):
        self.write_split_packages()
        (self.root / "node" / f"onnxruntime-node-linux-x64-{self.version}.tgz").unlink()
        self.write_package("node", "onnxruntime-node-linux-x64", {"os": ["linux"], "cpu": ["x64"]}, "1.31.0")
        result = self.validate()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("version mismatch", result.stderr)

    def test_validate_rejects_native_platform_mismatch(self):
        self.write_split_packages()
        self.write_package("node", "onnxruntime-node-linux-x64", {"os": ["win32"], "cpu": ["x64"]})
        result = self.validate()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("platform metadata mismatch", result.stderr)

    def test_validate_rejects_native_libc_mismatch(self):
        self.write_split_packages()
        self.write_package("node", "onnxruntime-node-linux-x64", {"os": ["linux"], "cpu": ["x64"], "libc": ["musl"]})
        result = self.validate()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("libc metadata mismatch", result.stderr)


if __name__ == "__main__":
    unittest.main()
