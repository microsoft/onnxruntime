# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import contextlib
import io
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from unittest import mock

import generate_nuspec_for_native_nuget as nuspec


class NativeNugetTests(unittest.TestCase):
    def test_retired_qnn_execution_provider_is_rejected(self):
        argv = ["generate_nuspec_for_native_nuget.py"]
        for option in (
            "package_name",
            "package_version",
            "target_architecture",
            "build_config",
            "ort_build_path",
            "native_build_path",
            "packages_path",
            "sources_path",
            "commit_id",
        ):
            argv.extend([f"--{option}", "unused"])
        argv.extend(["--execution_provider", "qnn"])
        stderr = io.StringIO()
        with (
            mock.patch.object(sys, "argv", argv),
            contextlib.redirect_stderr(stderr),
            self.assertRaises(SystemExit) as error,
        ):
            nuspec.parse_arguments()
        self.assertEqual(error.exception.code, 2)
        self.assertIn("invalid choice: 'qnn'", stderr.getvalue())

    def test_retired_qnn_package_is_rejected_before_generating_files(self):
        for args in (
            argparse.Namespace(package_name="Microsoft.ML.OnnxRuntime.QNN"),
            argparse.Namespace(package_name="Microsoft.ML.OnnxRuntime", execution_provider="qnn"),
        ):
            with (
                self.subTest(args=args),
                mock.patch.object(nuspec, "generate_files") as generate_files,
                self.assertRaisesRegex(ValueError, "In-tree QNN packaging has been retired"),
            ):
                nuspec.generate_nuspec(args)
            generate_files.assert_not_called()

    def test_supported_windows_packages_preserve_runtime_and_props(self):
        for package_name, execution_provider, provider_dll in (
            ("Microsoft.ML.OnnxRuntime", "None", None),
            ("Microsoft.ML.OnnxRuntime.Gpu.Windows", "cuda", "onnxruntime_providers_cuda.dll"),
            ("Microsoft.ML.OnnxRuntime.DirectML", "None", None),
            ("Microsoft.ML.OnnxRuntime.Snpe", "snpe", None),
        ):
            with self.subTest(package_name=package_name), tempfile.TemporaryDirectory() as native_build_path:
                args = argparse.Namespace(
                    package_name=package_name,
                    package_version="1.0.0",
                    target_architecture="x64",
                    build_config="Release",
                    ort_build_path="build",
                    native_build_path=native_build_path,
                    packages_path="packages",
                    sources_path="source",
                    commit_id="test-commit",
                    is_release_build="true",
                    execution_provider=execution_provider,
                    sdk_info="",
                )
                with (
                    mock.patch.object(nuspec, "is_windows", return_value=True),
                    mock.patch.object(nuspec, "is_linux", return_value=False),
                    mock.patch.object(nuspec, "copy_file") as copy_file,
                    mock.patch.object(nuspec.os.path, "exists", return_value=False),
                ):
                    xml = "\n".join(nuspec.generate_nuspec(args))
                package = ET.fromstring(xml)
                self.assertEqual(package.findtext("metadata/id"), package_name)
                files = package.findall("files/file")
                self.assertTrue(any(file.attrib["src"].endswith("onnxruntime.dll") for file in files))
                if provider_dll:
                    self.assertTrue(any(file.attrib["src"].endswith(provider_dll) for file in files))
                self.assertTrue(any(call.args[0].endswith("props.xml") for call in copy_file.call_args_list))
                self.assertNotIn("Qnn", xml)
                self.assertNotIn("qnn", xml)


if __name__ == "__main__":
    unittest.main()
