import argparse
import contextlib
import io
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import unittest.mock
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pack_nuget


class PackNugetTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.staging = self.root / "staging"

    def stage_platform(self, platform, missing=None, *, include_agility_sdk=True, require_agility_sdk=False):
        source = Path(tempfile.mkdtemp(dir=self.root))
        files = list(pack_nuget.PLATFORMS[platform][1])
        if platform.startswith("win_") and include_agility_sdk:
            files.extend(pack_nuget.WINDOWS_AGILITY_SDK_BINARIES)
        for filename in files:
            if filename != missing:
                path = source / filename
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(filename.encode())
        args = argparse.Namespace(
            artifacts_dir=None,
            require_agility_sdk=require_agility_sdk,
            **{f"binary_dir_{name}": source if name == platform else None for name in pack_nuget.PLATFORMS},
        )
        with contextlib.redirect_stdout(io.StringIO()):
            pack_nuget.stage_binaries(self.staging, args, [platform])

    def write_staged_package(self, package):
        with zipfile.ZipFile(package, "w") as archive:
            for path in self.staging.rglob("*"):
                if path.is_file():
                    archive.write(path, path.relative_to(self.staging).as_posix())
            archive.writestr("buildTransitive/Microsoft.ML.OnnxRuntime.EP.WebGpu.targets", b"test")

    def test_stage_binaries_preserves_layout(self):
        for platform, (rid, files) in pack_nuget.PLATFORMS.items():
            with self.subTest(platform=platform):
                self.stage_platform(platform)
                expected_files = list(files)
                if rid.startswith("win-"):
                    expected_files.extend(pack_nuget.WINDOWS_AGILITY_SDK_BINARIES)
                for filename in expected_files:
                    destination = self.staging / "runtimes" / rid / "native" / filename
                    self.assertEqual(destination.read_bytes(), filename.encode())

    def test_stage_binaries_rejects_missing_agility_dll(self):
        for platform in ("win_x64", "win_arm64"):
            for missing in pack_nuget.WINDOWS_AGILITY_SDK_BINARIES:
                for required in (False, True):
                    with (
                        self.subTest(platform=platform, missing=missing, required=required),
                        self.assertRaises(pack_nuget.PackError),
                    ):
                        self.stage_platform(platform, missing, require_agility_sdk=required)

    def test_parse_agility_requirement(self):
        for required in (False, True):
            with self.subTest(required=required):
                argv = ["pack_nuget.py", "--version", "0.1.0"]
                if required:
                    argv.append("--require-agility-sdk")
                with unittest.mock.patch.object(sys, "argv", argv):
                    self.assertEqual(pack_nuget.parse_args().require_agility_sdk, required)

    def test_stage_binaries_agility_requirement(self):
        for platform in pack_nuget.PLATFORMS:
            for include_sdk in (False, True):
                for required in (False, True):
                    with self.subTest(platform=platform, include_sdk=include_sdk, required=required):
                        self.staging = self.root / f"{platform}-{include_sdk}-{required}"
                        if platform.startswith("win_") and required and not include_sdk:
                            with self.assertRaises(pack_nuget.PackError):
                                self.stage_platform(platform, include_agility_sdk=False, require_agility_sdk=True)
                        else:
                            self.stage_platform(platform, include_agility_sdk=include_sdk, require_agility_sdk=required)
                            rid = pack_nuget.PLATFORMS[platform][0]
                            sdk_dir = self.staging / "runtimes" / rid / "native" / "D3D12"
                            self.assertEqual(sdk_dir.is_dir(), platform.startswith("win_") and include_sdk)

    def test_verify_package_allows_no_sdk_but_rejects_partial_sdk(self):
        for platform in ("win_x64", "win_arm64"):
            self.stage_platform(platform, include_agility_sdk=False)
        entries = {path.relative_to(self.staging).as_posix() for path in self.staging.rglob("*") if path.is_file()}
        entries.add("buildTransitive/Microsoft.ML.OnnxRuntime.EP.WebGpu.targets")
        package = self.root / "test.nupkg"
        for rid in ("win-x64", "win-arm64"):
            for sdk_file in (None, *pack_nuget.WINDOWS_AGILITY_SDK_BINARIES):
                with self.subTest(rid=rid, sdk_file=sdk_file):
                    with zipfile.ZipFile(package, "w") as archive:
                        for entry in entries:
                            archive.writestr(entry, b"test")
                        if sdk_file:
                            archive.writestr(f"runtimes/{rid}/native/{sdk_file}", b"test")
                    if sdk_file:
                        with self.assertRaises(pack_nuget.PackError):
                            pack_nuget.verify_package(package, self.staging)
                    else:
                        pack_nuget.verify_package(package, self.staging)
                    with self.assertRaises(pack_nuget.PackError):
                        pack_nuget.verify_package(package, self.staging, require_agility_sdk=True)

    def test_pack_only_enforces_agility_requirement(self):
        self.stage_platform("win_x64", include_agility_sdk=False)

        def produce_package(command, *, check):
            self.assertTrue(check)
            output = Path(command[command.index("--output") + 1])
            self.write_staged_package(output / "test.nupkg")

        for required in (False, True):
            with self.subTest(required=required):
                args = argparse.Namespace(
                    version="0.1.0",
                    configuration="Release",
                    nuget_config=None,
                    pack_only=True,
                    require_agility_sdk=required,
                )
                with (
                    unittest.mock.patch.object(pack_nuget.subprocess, "run", side_effect=produce_package) as run,
                    contextlib.redirect_stdout(io.StringIO()),
                ):
                    if required:
                        with self.assertRaises(pack_nuget.PackError):
                            pack_nuget.do_pack(self.staging / "test.csproj", self.root, args)
                    else:
                        pack_nuget.do_pack(self.staging / "test.csproj", self.root, args)
                    self.assertIn("--no-build", run.call_args.args[0])

    def test_pack_preserves_history_and_replaces_only_current_packages(self):
        self.stage_platform("win_x64")
        output = self.root / "output"
        output.mkdir()
        package_id = "Microsoft.ML.OnnxRuntime.EP.WebGpu"
        old_package = output / f"{package_id}.0.1.0.nupkg"
        old_symbols = output / f"{package_id}.0.1.0.snupkg"
        current_package = output / f"{package_id}.0.2.0.nupkg"
        current_symbols = output / f"{package_id}.0.2.0.snupkg"
        for package in (old_package, old_symbols, current_package, current_symbols):
            package.write_bytes(b"previous build")

        def produce_package(command, *, check):
            self.assertTrue(check)
            temporary_output = Path(command[command.index("--output") + 1])
            self.assertNotEqual(temporary_output, output)
            self.write_staged_package(temporary_output / current_package.name)
            (temporary_output / current_symbols.name).write_bytes(b"current symbols")

        args = argparse.Namespace(
            version="0.2.0",
            configuration="Release",
            nuget_config=None,
            pack_only=True,
            require_agility_sdk=True,
        )
        log = io.StringIO()
        with (
            unittest.mock.patch.object(pack_nuget.subprocess, "run", side_effect=produce_package),
            contextlib.redirect_stdout(log),
        ):
            pack_nuget.do_pack(self.staging / "test.csproj", output, args)

        pack_nuget.verify_package(current_package, self.staging, require_agility_sdk=True)
        self.assertEqual(current_symbols.read_bytes(), b"current symbols")
        self.assertEqual(old_package.read_bytes(), b"previous build")
        self.assertEqual(old_symbols.read_bytes(), b"previous build")
        self.assertNotIn(old_package.name, log.getvalue())
        self.assertNotIn(old_symbols.name, log.getvalue())
        self.assertIn(f"Produced: {current_package.name}", log.getvalue())
        self.assertIn(f"Produced: {current_symbols.name}", log.getvalue())

    def test_pack_does_not_accept_stale_package_when_nothing_is_produced(self):
        self.stage_platform("win_x64")
        package = self.root / "Microsoft.ML.OnnxRuntime.EP.WebGpu.0.2.0.nupkg"
        self.write_staged_package(package)
        original = package.read_bytes()
        args = argparse.Namespace(
            version="0.2.0",
            configuration="Release",
            nuget_config=None,
            pack_only=True,
            require_agility_sdk=True,
        )
        with (
            unittest.mock.patch.object(pack_nuget.subprocess, "run"),
            contextlib.redirect_stdout(io.StringIO()),
            self.assertRaisesRegex(pack_nuget.PackError, "no .nupkg files found"),
        ):
            pack_nuget.do_pack(self.staging / "test.csproj", self.root, args)
        self.assertEqual(package.read_bytes(), original)

    def test_verify_package_checks_sdk_layout_and_targets(self):
        for platform in ("win_x64", "win_arm64"):
            self.stage_platform(platform)
        entries = {path.relative_to(self.staging).as_posix() for path in self.staging.rglob("*") if path.is_file()}
        entries.add("buildTransitive/Microsoft.ML.OnnxRuntime.EP.WebGpu.targets")
        package = self.root / "test.nupkg"
        for missing in (None, *sorted(entries)):
            with self.subTest(missing=missing):
                with zipfile.ZipFile(package, "w") as archive:
                    for entry in entries:
                        if entry != missing:
                            archive.writestr(entry, b"test")
                    if missing:
                        archive.writestr(Path(missing).name, b"flattened")
                if missing:
                    with self.assertRaises(pack_nuget.PackError):
                        pack_nuget.verify_package(package, self.staging)
                else:
                    pack_nuget.verify_package(package, self.staging)
                    pack_nuget.verify_package(package, self.staging, require_agility_sdk=True)

    @unittest.skipUnless(shutil.which("dotnet"), "Requires the .NET SDK")
    def test_sdk_layout_after_build_and_publish_with_rid_fallback(self):
        package_id = "Microsoft.ML.OnnxRuntime.EP.WebGpu"
        feed = self.root / "feed"
        feed.mkdir()
        environment = dict(os.environ, NUGET_PACKAGES=str(self.root / "packages"))

        def run_dotnet(*arguments):
            result = subprocess.run(
                ["dotnet", *map(str, arguments)],
                check=False,
                cwd=self.root,
                env=environment,
                capture_output=True,
                text=True,
                timeout=120,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            return result.stdout.strip()

        sdk_major = int(run_dotnet("--version").split(".")[0])
        if sdk_major < 8:
            self.skipTest("Requires .NET SDK 8 or later")

        manifest = ET.Element("package")
        metadata = ET.SubElement(manifest, "metadata")
        for key, value in {
            "id": package_id,
            "version": "0.0.1",
            "authors": "test",
            "description": "Native asset layout fixture",
        }.items():
            ET.SubElement(metadata, key).text = value
        with zipfile.ZipFile(feed / f"{package_id}.0.0.1.nupkg", "w") as archive:
            archive.writestr(f"{package_id}.nuspec", ET.tostring(manifest))
            targets = pack_nuget.PROJECT_DIR / f"{package_id}.targets"
            archive.write(targets, f"buildTransitive/{targets.name}")
            for rid in ("win-x64", "win-arm64"):
                for filename in pack_nuget.WINDOWS_BINARIES + pack_nuget.WINDOWS_AGILITY_SDK_BINARIES:
                    archive.writestr(f"runtimes/{rid}/native/{filename}", f"{rid}/{filename}".encode())

        for rid in (None, "win-x64", "win-arm64", "win10-x64", "win10-arm64"):
            with self.subTest(rid=rid):
                project_dir = self.root / (rid or "portable")
                project_dir.mkdir()
                project = ET.Element("Project", Sdk="Microsoft.NET.Sdk")
                properties = ET.SubElement(project, "PropertyGroup")
                settings = {
                    "TargetFramework": f"net{sdk_major}.0",
                    "CopyLocalLockFileAssemblies": "true",
                    "UseRidGraph": "true",
                    "SelfContained": "false",
                }
                if rid:
                    settings["RuntimeIdentifier"] = rid
                for key, value in settings.items():
                    ET.SubElement(properties, key).text = value
                items = ET.SubElement(project, "ItemGroup")
                ET.SubElement(items, "PackageReference", Include=package_id, Version="0.0.1")
                project_file = project_dir / "consumer.csproj"
                ET.ElementTree(project).write(project_file, encoding="utf-8", xml_declaration=True)

                run_dotnet("restore", project_file, "--source", feed, "-p:NuGetAudit=false")
                for action in ("build", "publish"):
                    output = self.root / f"{rid or 'portable'}-{action}"
                    run_dotnet(action, project_file, "--no-restore", "--output", output, "--verbosity", "quiet")
                    expected_rids = (rid.replace("win10-", "win-"),) if rid else ("win-x64", "win-arm64")
                    for expected_rid in expected_rids:
                        native_dir = output if rid else output / "runtimes" / expected_rid / "native"
                        for filename in pack_nuget.WINDOWS_BINARIES + pack_nuget.WINDOWS_AGILITY_SDK_BINARIES:
                            binary = native_dir / filename
                            self.assertTrue(binary.is_file(), f"{action}: missing {binary}")
                            self.assertEqual(binary.read_bytes(), f"{expected_rid}/{filename}".encode())


if __name__ == "__main__":
    unittest.main()
