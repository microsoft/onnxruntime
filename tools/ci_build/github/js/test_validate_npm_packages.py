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

NATIVE_LIBRARIES = {
    "linux": "libonnxruntime.so.1",
    "darwin": "libonnxruntime.1.dylib",
    "win32": "onnxruntime.dll",
}
NATIVE_TARGETS = tuple((platform, arch) for platform in NATIVE_LIBRARIES for arch in ("x64", "arm64"))


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

    def write_package(self, directory, name, fields, version=None, files=None):
        version = version or self.version
        manifest = {"name": name, "version": version, **fields}
        content = json.dumps(manifest).encode()
        with tarfile.open(self.root / directory / f"{name}-{version}.tgz", "w:gz") as archive:
            member = tarfile.TarInfo("package/package.json")
            member.size = len(content)
            archive.addfile(member, io.BytesIO(content))
            for filename, payload in (files or {}).items():
                if isinstance(payload, tarfile.TarInfo):
                    archive.addfile(payload)
                else:
                    member = tarfile.TarInfo(filename)
                    member.size = len(payload)
                    archive.addfile(member, io.BytesIO(payload))

    def native_files(self, platform, arch):
        directory = f"package/bin/napi-v6/{platform}/{arch}"
        return {
            f"{directory}/onnxruntime_binding.node": b"binding fixture",
            f"{directory}/{NATIVE_LIBRARIES[platform]}": b"runtime fixture",
        }

    def write_native_package(self, platform, arch, fields=None, version=None, files=None):
        metadata = {"os": [platform], "cpu": [arch]}
        if platform == "linux":
            metadata["libc"] = ["glibc"]
        metadata.update(fields or {})
        if files is None:
            files = self.native_files(platform, arch)
        self.write_package("node", f"onnxruntime-node-{platform}-{arch}", metadata, version, files)

    def write_split_packages(self):
        dependencies = {}
        for platform, arch in NATIVE_TARGETS:
            name = f"onnxruntime-node-{platform}-{arch}"
            dependencies[name] = self.version
            self.write_native_package(platform, arch)
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
        self.write_native_package("linux", "x64", version="1.31.0")
        result = self.validate()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("version mismatch", result.stderr)

    def test_validate_rejects_native_platform_mismatch(self):
        self.write_split_packages()
        self.write_native_package("linux", "x64", fields={"os": ["win32"]})
        result = self.validate()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("platform metadata mismatch", result.stderr)

    def test_validate_rejects_native_libc_mismatch(self):
        self.write_split_packages()
        self.write_native_package("linux", "x64", fields={"libc": ["musl"]})
        result = self.validate()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("libc metadata mismatch", result.stderr)

    def test_validate_rejects_manifest_only_native_archive(self):
        for platform, arch in NATIVE_TARGETS:
            with self.subTest(platform=platform, arch=arch):
                self.write_split_packages()
                self.write_native_package(platform, arch, files={})
                result = self.validate()
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("Native payload missing", result.stderr)
                self.assertIn(f"package/bin/napi-v6/{platform}/{arch}/onnxruntime_binding.node", result.stderr)

    def test_validate_rejects_missing_native_binding(self):
        for platform, arch in NATIVE_TARGETS:
            with self.subTest(platform=platform, arch=arch):
                self.write_split_packages()
                files = self.native_files(platform, arch)
                del files[f"package/bin/napi-v6/{platform}/{arch}/onnxruntime_binding.node"]
                self.write_native_package(platform, arch, files=files)
                result = self.validate()
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("Native payload missing", result.stderr)
                self.assertIn("onnxruntime_binding.node", result.stderr)

    def test_validate_rejects_missing_native_runtime_library(self):
        for platform, arch in NATIVE_TARGETS:
            with self.subTest(platform=platform, arch=arch):
                self.write_split_packages()
                files = self.native_files(platform, arch)
                del files[f"package/bin/napi-v6/{platform}/{arch}/{NATIVE_LIBRARIES[platform]}"]
                self.write_native_package(platform, arch, files=files)
                result = self.validate()
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("Native payload missing", result.stderr)
                self.assertIn(NATIVE_LIBRARIES[platform], result.stderr)

    def test_validate_rejects_native_payload_in_wrong_directory(self):
        for platform, arch in NATIVE_TARGETS:
            with self.subTest(platform=platform, arch=arch):
                self.write_split_packages()
                files = {
                    filename.replace("/napi-v6/", "/napi-v7/"): payload
                    for filename, payload in self.native_files(platform, arch).items()
                }
                self.write_native_package(platform, arch, files=files)
                result = self.validate()
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("Native payload missing", result.stderr)

    def test_validate_rejects_empty_or_non_regular_native_payload(self):
        for platform, arch in NATIVE_TARGETS:
            for filename in self.native_files(platform, arch):
                for kind in ("empty", "directory", "symlink", "hardlink"):
                    with self.subTest(platform=platform, arch=arch, filename=filename, kind=kind):
                        self.write_split_packages()
                        files = self.native_files(platform, arch)
                        if kind == "empty":
                            files[filename] = b""
                        else:
                            member = tarfile.TarInfo(filename)
                            member.type = {
                                "directory": tarfile.DIRTYPE,
                                "symlink": tarfile.SYMTYPE,
                                "hardlink": tarfile.LNKTYPE,
                            }[kind]
                            member.linkname = "package/package.json"
                            files[filename] = member
                        self.write_native_package(platform, arch, files=files)
                        result = self.validate()
                        self.assertNotEqual(result.returncode, 0)
                        self.assertIn("Native payload must be a non-empty regular file", result.stderr)
                        self.assertIn(filename, result.stderr)


if __name__ == "__main__":
    unittest.main()
