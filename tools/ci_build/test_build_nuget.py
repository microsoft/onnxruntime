# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import unittest
from unittest.mock import patch

import build


class BuildNugetTest(unittest.TestCase):
    def test_uses_sdk_msbuild_for_desktop_and_mobile(self):
        for windows, mobile in ((True, True), (True, False), (False, False)):
            with (
                self.subTest(windows=windows, mobile=mobile),
                patch.object(build, "is_windows", return_value=windows),
                patch.object(build, "is_linux", return_value=not windows),
                patch.object(build, "run_subprocess") as run_subprocess,
            ):
                build.build_nuget_package(
                    cmake_path="cmake",
                    source_dir="source",
                    build_dir="build",
                    configs=["Release"],
                    use_cuda=False,
                    use_openvino=False,
                    use_tensorrt=False,
                    use_dnnl=False,
                    use_winml=False,
                    use_qnn=False,
                    use_dml=False,
                    use_migraphx=False,
                    enable_training_apis=False,
                    msbuild_extra_options=[] if mobile else ["IncludeMobileTargets=false"],
                )

                commands = [call.args[0] for call in run_subprocess.call_args_list]
                self.assertEqual(len(commands), 3)
                solution = "OnnxRuntime.CSharp.sln" if mobile else "OnnxRuntime.DesktopOnly.CSharp.sln"
                self.assertEqual(commands[0][:3], ["dotnet", "restore", solution])
                self.assertEqual(commands[1][:3], ["dotnet", "msbuild", solution])
                self.assertEqual(commands[2][:3], ["dotnet", "msbuild", "OnnxRuntime.CSharp.proj"])
                for command in commands:
                    self.assertIn("/p:Configuration=Release", command)
                    self.assertEqual("/p:IncludeMobileTargets=false" in command, not mobile)


if __name__ == "__main__":
    unittest.main()
