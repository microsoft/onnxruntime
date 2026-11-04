# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import contextlib
import io
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

import resolve_android_ndk


class ResolveAndroidNdkTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="ort android sdk ")
        self.addCleanup(self.directory.cleanup)
        self.sdk_root = Path(self.directory.name).resolve()
        self.run = mock.patch.object(resolve_android_ndk.subprocess, "run").start()
        self.addCleanup(mock.patch.stopall)

    def install_ndk(self, version):
        ndk = self.sdk_root / "ndk" / version
        ndk.mkdir(parents=True)
        (ndk / "source.properties").write_text(f"Pkg.Revision = {version}\n", encoding="utf-8")
        toolchain = ndk / "build" / "cmake" / "android.toolchain.cmake"
        toolchain.parent.mkdir(parents=True)
        toolchain.touch()
        host = {"linux": "linux-x86_64", "darwin": "darwin-x86_64", "win32": "windows-x86_64"}[sys.platform]
        compiler = (
            ndk
            / "toolchains"
            / "llvm"
            / "prebuilt"
            / host
            / "bin"
            / ("clang.exe" if sys.platform == "win32" else "clang")
        )
        compiler.parent.mkdir(parents=True)
        compiler.touch(mode=0o755)
        return ndk, compiler

    def test_selects_numerically_latest_installed_major_without_sdkmanager(self):
        self.install_ndk("28.2.999")
        expected, compiler = self.install_ndk("28.10.100")
        for ignored in ("27.9.999", "29.0.999", "28.11.123-beta1", "28.11.123 rc1", "latest"):
            (self.sdk_root / "ndk" / ignored).mkdir()
        self.assertEqual(resolve_android_ndk.resolve_ndk(self.sdk_root, "28"), expected)
        self.run.assert_called_once_with([str(compiler), "--version"], check=True, stdout=sys.stderr)

    def test_missing_ndk_directory_fails_without_online_lookup(self):
        with self.assertRaisesRegex(ValueError, "Provision NDK 28"):
            resolve_android_ndk.resolve_ndk(self.sdk_root, "28")
        self.run.assert_not_called()

    def test_numeric_preview_directory_is_not_selected(self):
        expected, compiler = self.install_ndk("28.2.123")
        for version, revision in (("28.3.456", "28.3.456-beta1"), ("28.4.567", "28.4.567 rc1")):
            ndk, _ = self.install_ndk(version)
            (ndk / "source.properties").write_text(f"Pkg.Revision = {revision}\n", encoding="utf-8")
        self.assertEqual(resolve_android_ndk.resolve_ndk(self.sdk_root, "28"), expected)
        self.run.assert_called_once_with([str(compiler), "--version"], check=True, stdout=sys.stderr)

    def test_only_other_majors_or_previews_fails(self):
        self.install_ndk("27.2.123")
        self.install_ndk("29.0.123")
        (self.sdk_root / "ndk" / "28.3.123-beta1").mkdir()
        with self.assertRaisesRegex(ValueError, "No stable NDK 28"):
            resolve_android_ndk.resolve_ndk(self.sdk_root, "28")
        self.run.assert_not_called()

    def test_partial_newest_installation_fails_instead_of_using_older_revision(self):
        self.install_ndk("28.0.123")
        (self.sdk_root / "ndk" / "28.2.456").mkdir()
        with self.assertRaises(FileNotFoundError):
            resolve_android_ndk.resolve_ndk(self.sdk_root, "28")
        self.run.assert_not_called()

    def test_metadata_revision_mismatch_fails(self):
        ndk, _ = self.install_ndk("28.2.123")
        (ndk / "source.properties").write_text("Pkg.Revision = 28.1.123\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "does not match"):
            resolve_android_ndk.resolve_ndk(self.sdk_root, "28")
        self.run.assert_not_called()

    def test_missing_revision_fails(self):
        ndk, _ = self.install_ndk("28.2.123")
        (ndk / "source.properties").write_text("Pkg.Desc = Android NDK\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "does not match"):
            resolve_android_ndk.resolve_ndk(self.sdk_root, "28")

    def test_missing_toolchain_fails(self):
        ndk, _ = self.install_ndk("28.2.123")
        (ndk / "build" / "cmake" / "android.toolchain.cmake").unlink()
        with self.assertRaisesRegex(ValueError, "Missing NDK toolchain"):
            resolve_android_ndk.resolve_ndk(self.sdk_root, "28")
        self.run.assert_not_called()

    def test_missing_compiler_fails(self):
        _, compiler = self.install_ndk("28.2.123")
        compiler.unlink()
        with self.assertRaisesRegex(ValueError, "NDK compiler"):
            resolve_android_ndk.resolve_ndk(self.sdk_root, "28")
        self.run.assert_not_called()

    def test_unusable_compiler_fails(self):
        self.install_ndk("28.2.123")
        self.run.side_effect = subprocess.CalledProcessError(1, ["clang", "--version"])
        with self.assertRaises(subprocess.CalledProcessError):
            resolve_android_ndk.resolve_ndk(self.sdk_root, "28")

    def test_exact_installed_revision_is_reused(self):
        expected, compiler = self.install_ndk("28.0.123")
        self.install_ndk("28.2.456")
        self.assertEqual(resolve_android_ndk.resolve_ndk(self.sdk_root, "28.0.123"), expected)
        self.run.assert_called_once_with([str(compiler), "--version"], check=True, stdout=sys.stderr)

    def test_missing_exact_revision_is_installed_and_validated(self):
        def install(command, **_kwargs):
            if "--install" in command:
                self.assertEqual(command[1:], ["--install", "ndk;28.0.123"])
                self.install_ndk("28.0.123")

        self.run.side_effect = install
        expected = self.sdk_root / "ndk" / "28.0.123"
        self.assertEqual(resolve_android_ndk.resolve_ndk(self.sdk_root, "28.0.123"), expected)
        self.assertEqual(self.run.call_count, 2)

    def test_exact_installation_failure_propagates(self):
        self.run.side_effect = subprocess.CalledProcessError(1, ["sdkmanager"])
        with self.assertRaises(subprocess.CalledProcessError):
            resolve_android_ndk.resolve_ndk(self.sdk_root, "28.0.123")
        self.run.assert_called_once()

    def test_invalid_version_is_rejected(self):
        for version in ("latest", "28.2", "../28", "28.2.123-beta1"):
            with self.subTest(version=version), self.assertRaisesRegex(ValueError, "Expected an NDK major"):
                resolve_android_ndk.resolve_ndk(self.sdk_root, version)
        self.run.assert_not_called()

    def test_cli_writes_only_resolved_path_to_stdout(self):
        ndk, _ = self.install_ndk("28.2.123")
        output = io.StringIO()
        with (
            mock.patch.object(sys, "argv", ["resolve_android_ndk.py", "--android-sdk-root", str(self.sdk_root)]),
            contextlib.redirect_stdout(output),
        ):
            resolve_android_ndk.main()
        self.assertEqual(output.getvalue(), f"{ndk}\n")


if __name__ == "__main__":
    unittest.main()
