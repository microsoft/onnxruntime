# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import generate_nuspec_for_native_nuget as generator


class MobileFrameworkTest(unittest.TestCase):
    def test_mobile_dependencies_target_dotnet_10(self):
        lines = []
        generator.generate_dependencies(lines, "Microsoft.ML.OnnxRuntime", "1.0.0")
        dependencies = ET.fromstring("".join(lines))
        mobile_groups = {
            group.attrib["targetFramework"]: group
            for group in dependencies
            if group.attrib["targetFramework"].startswith("net")
        }
        self.assertEqual(
            set(mobile_groups),
            {"net10.0-android36.0", "net10.0-ios18.7", "net10.0-maccatalyst18.7"},
        )
        for group in mobile_groups.values():
            self.assertEqual(
                group.find("dependency").attrib,
                {"id": "Microsoft.ML.OnnxRuntime.Managed", "version": "1.0.0"},
            )

    def test_mobile_targets_match_dependency_frameworks(self):
        cases = (
            ("Microsoft.ML.OnnxRuntime", ("net10.0-android36.0", "net10.0-ios18.7", "net10.0-maccatalyst18.7")),
            ("Microsoft.ML.OnnxRuntime.Training", ("net10.0-android36.0",)),
        )
        for package, frameworks in cases:
            with (
                self.subTest(package=package),
                tempfile.TemporaryDirectory() as native_directory,
                patch.object(generator, "is_windows", return_value=True),
                patch.object(generator, "is_linux", return_value=False),
                patch.object(generator, "copy_file") as copy_file,
            ):
                args = SimpleNamespace(
                    package_name=package,
                    sources_path=str(Path(__file__).resolve().parents[2]),
                    native_build_path=native_directory,
                    target_architecture="x64",
                    execution_provider="None",
                    is_release_build="true",
                )
                lines = []
                generator.generate_files(lines, args)
                files = ET.fromstring("".join(lines))
                mobile_targets = {file.attrib["target"] for file in files if "\\net10.0-" in file.attrib["target"]}
                self.assertEqual(
                    mobile_targets,
                    {f"{folder}\\{framework}" for folder in ("build", "buildTransitive") for framework in frameworks},
                )
                for call in copy_file.call_args_list:
                    self.assertTrue(Path(call.args[0]).is_file(), call.args[0])


if __name__ == "__main__":
    unittest.main()
