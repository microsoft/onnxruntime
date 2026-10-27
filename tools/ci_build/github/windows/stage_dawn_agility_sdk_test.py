# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import hashlib
import shutil
import subprocess
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

import stage_dawn_agility_sdk as staging


@unittest.skipUnless(shutil.which("cmake"), "Requires CMake")
class DawnAgilitySdkCMakeTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.mirror = self.root / "mirror with spaces"
        self.url = "https://www.nuget.org/api/v2/package/Microsoft.Direct3D.D3D12/1.721.3-preview"
        self.archive = self.mirror / "www.nuget.org/api/v2/package/Microsoft.Direct3D.D3D12/1.721.3-preview.zip"
        self.archive.parent.mkdir(parents=True)
        with zipfile.ZipFile(self.archive, "w") as package:
            package.writestr("Microsoft.Direct3D.D3D12.nuspec", "<package />")
            package.writestr("build/native/include/d3d12.h", "local SDK fixture")
        self.digest = hashlib.sha1(self.archive.read_bytes()).hexdigest()

    def configure_sdk(self, digest=None):
        cmake_dir = Path(__file__).resolve().parents[4] / "cmake"
        source = (cmake_dir / "external" / "onnxruntime_external_deps.cmake").read_text(encoding="utf-8")
        loader = source[
            source.index("file(STRINGS deps.txt") : source.index('message(STATUS "Loading Dependencies ...")')
        ]
        start = source.index(
            "    if (DAWN_USE_AGILITY_SDK)", source.index("    onnxruntime_fetchcontent_makeavailable(dawn)")
        )
        sdk = source[start : source.index("\n  endif()\n  endif()", start)]
        (self.root / "deps.txt").write_text(f"dawn_agility_sdk;{self.url};{digest or self.digest}\n", encoding="utf-8")
        script = self.root / "CMakeLists.txt"
        script.write_text(
            "cmake_minimum_required(VERSION 3.28)\n"
            "project(dawn_agility_sdk_local NONE)\n"
            "include(FetchContent)\n"
            f'include("{(cmake_dir / "external" / "helper_functions.cmake").as_posix()}")\n'
            "set(DAWN_USE_AGILITY_SDK ON)\n"
            'set(DAWN_AGILITY_SDK_DIR "${CMAKE_CURRENT_BINARY_DIR}/dawn-src/third_party/agility-sdk")\n'
            f'set(onnxruntime_CMAKE_DEPS_MIRROR_DIR "{self.mirror.as_posix()}")\n'
            + loader
            + f'\nif(NOT DEP_URL_dawn_agility_sdk STREQUAL "{self.archive.as_posix()}")\n'
            '  message(FATAL_ERROR "SDK did not resolve to the local ZIP")\n'
            "endif()\n" + sdk,
            encoding="utf-8",
        )
        return subprocess.run(
            ["cmake", "-S", str(self.root), "-B", str(self.root / "build")],
            cwd=self.root,
            capture_output=True,
            text=True,
            check=False,
        )

    def test_extracts_zip_mirror_of_extensionless_url(self):
        result = self.configure_sdk()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        header = (
            self.root
            / "build"
            / "dawn-src"
            / "third_party"
            / "agility-sdk"
            / "src"
            / "build"
            / "native"
            / "include"
            / "d3d12.h"
        )
        self.assertEqual(header.read_text(encoding="utf-8"), "local SDK fixture")

    def test_prefers_exact_mirror_path_when_both_exist(self):
        self.url += ".zip"
        self.archive.with_name(self.archive.name + ".zip").write_bytes(b"not the expected archive")
        result = self.configure_sdk()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_rejects_local_archive_hash_mismatch(self):
        result = self.configure_sdk(digest="0" * 40)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("SHA1", result.stdout + result.stderr)
        self.assertIn("does not match expected value", result.stdout + result.stderr)
        self.assertNotIn("Do not know how to extract", result.stderr)


class StageDawnAgilitySdkTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source dir's"
        (self.source / "cmake").mkdir(parents=True)
        self.config = self.source / "NuGet.config"
        self.config.write_text("<configuration />", encoding="utf-8")
        self.mirror = self.root / "mirror with spaces"
        self.archive = self.root / "sdk.nupkg"
        self.archive.write_bytes(b"test package bytes, preserved without repacking")
        self.digest = hashlib.sha1(self.archive.read_bytes()).hexdigest()
        self.mode = ""
        self.package_dir = None
        which = mock.patch.object(staging.shutil, "which", return_value=str(self.root / "nuget.exe"))
        self.which = which.start()
        self.addCleanup(which.stop)
        run = mock.patch.object(staging.subprocess, "run", side_effect=self.fake_nuget)
        self.run = run.start()
        self.addCleanup(run.stop)
        self.write_pin()

    def write_pin(self, package_id="Microsoft.Direct3D.D3D12", version="1.721.3-preview", digest=None):
        self.url = f"https://www.nuget.org/api/v2/package/{package_id}/{version}"
        self.pin = f"dawn_agility_sdk;{self.url};{digest or self.digest}\n"
        self.deps = self.source / "cmake" / "deps.txt"
        self.deps.write_text("#Name;Url;SHA1\n" + self.pin, encoding="utf-8")
        self.destination = self.mirror / f"www.nuget.org/api/v2/package/{package_id}/{version}.zip"

    def fake_nuget(self, args, *, check):
        self.assertTrue(check)
        self.package_dir = Path(args[args.index("-OutputDirectory") + 1])
        if self.mode == "failure":
            raise subprocess.CalledProcessError(23, args)
        output = self.package_dir / args[2]
        output.mkdir()
        if self.mode != "missing":
            (output / "sdk.nupkg").write_bytes(self.archive.read_bytes())
        if self.mode == "multiple":
            (output / "other.nupkg").write_bytes(self.archive.read_bytes())

    def run_staging(self):
        try:
            return staging.stage_dawn_agility_sdk(self.source, self.mirror)
        finally:
            if self.package_dir is not None:
                self.assertFalse(self.package_dir.exists())

    def test_stages_exact_archive_without_changing_deps(self):
        original = self.deps.read_bytes()
        self.assertEqual(self.run_staging(), self.destination)
        self.assertEqual(self.destination.read_bytes(), self.archive.read_bytes())
        self.assertEqual(self.deps.read_bytes(), original)
        self.which.assert_called_once_with("nuget.exe")
        args = self.run.call_args.args[0]
        self.assertEqual(args[:3], [self.which.return_value, "install", "Microsoft.Direct3D.D3D12"])
        self.assertEqual(args[args.index("-Version") + 1], "1.721.3-preview")
        self.assertEqual(args[args.index("-ConfigFile") + 1], str(self.config))
        self.assertEqual(args[args.index("-PackageSaveMode") + 1], "nupkg")
        self.assertEqual(args[args.index("-DependencyVersion") + 1], "Ignore")
        for flag in ("-NonInteractive", "-ExcludeVersion"):
            self.assertIn(flag, args)
        self.assertNotIn("-Source", args)

    def test_preserves_comments_other_entries_line_endings_and_bom(self):
        prefix = f"# Same URL in a comment: {self.url}\r\nother;https://example.test/other.zip;{'1' * 40}\r\n"
        suffix = "# trailing comment\r\n"
        self.deps.write_bytes((prefix + self.pin.rstrip("\n") + "\r\n" + suffix).encode("utf-8-sig"))
        original = self.deps.read_bytes()
        self.run_staging()
        self.assertEqual(self.deps.read_bytes(), original)

    def test_uses_updated_pin_and_overwrites_stale_archive(self):
        self.write_pin(package_id="Microsoft.Direct3D.D3D12.Test", version="2.0.0-preview")
        self.destination.parent.mkdir(parents=True)
        self.destination.write_bytes(b"stale")
        self.run_staging()
        self.assertEqual(self.destination.read_bytes(), self.archive.read_bytes())
        args = self.run.call_args.args[0]
        self.assertEqual(args[2], "Microsoft.Direct3D.D3D12.Test")
        self.assertEqual(args[args.index("-Version") + 1], "2.0.0-preview")

    def test_accepts_uppercase_sha1(self):
        self.write_pin(digest=self.digest.upper())
        self.assertEqual(self.run_staging(), self.destination)

    def test_rejects_hash_mismatch(self):
        self.write_pin(digest="0" * 40)
        original = self.deps.read_bytes()
        with self.assertRaisesRegex(ValueError, "Agility SDK SHA1 mismatch"):
            self.run_staging()
        self.assertFalse(self.destination.exists())
        self.assertEqual(self.deps.read_bytes(), original)

    def test_rejects_failed_nuget_command(self):
        self.mode = "failure"
        with self.assertRaises(subprocess.CalledProcessError) as error:
            self.run_staging()
        self.assertEqual(error.exception.returncode, 23)
        self.assertFalse(self.destination.exists())

    def test_rejects_missing_or_ambiguous_archive(self):
        for mode in ("missing", "multiple"):
            with self.subTest(mode=mode):
                self.mode = mode
                with self.assertRaisesRegex(ValueError, "Expected exactly one retained .nupkg"):
                    self.run_staging()
                self.assertFalse(self.destination.exists())

    def test_rejects_invalid_pins_before_acquiring_package(self):
        for entry in (
            "",
            self.pin + self.pin,
            self.pin.replace(self.digest, "invalid"),
            self.pin.replace("https:", "http:"),
        ):
            with self.subTest(entry=entry):
                self.deps.write_text(entry, encoding="utf-8")
                with self.assertRaises(ValueError):
                    self.run_staging()
                self.run.assert_not_called()
                self.assertFalse(self.destination.exists())

    def test_rejects_missing_config(self):
        self.config.unlink()
        with self.assertRaisesRegex(FileNotFoundError, "NuGet configuration not found"):
            self.run_staging()
        self.run.assert_not_called()

    def test_reports_missing_nuget_without_bootstrapping(self):
        self.which.return_value = None
        with self.assertRaisesRegex(FileNotFoundError, "nuget.exe must be on PATH"):
            self.run_staging()
        self.run.assert_not_called()
        self.assertFalse(self.destination.exists())

    def test_cli_passes_directory_arguments(self):
        with (
            mock.patch(
                "sys.argv",
                ["stage_dawn_agility_sdk.py", "--source-dir", str(self.source), "--mirror-dir", str(self.mirror)],
            ),
            mock.patch.object(staging, "stage_dawn_agility_sdk") as stage,
        ):
            staging.main()
        stage.assert_called_once_with(self.source, self.mirror)


if __name__ == "__main__":
    unittest.main()
