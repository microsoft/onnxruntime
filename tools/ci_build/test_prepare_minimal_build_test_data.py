# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import prepare_minimal_build_test_data as prepare


class PrepareMinimalBuildTestDataTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory(prefix="minimal build test ")
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name).resolve()
        self.repo = self.root / "source"
        self.build_dir = self.root / "build"
        self.test_data_root = self.repo / "onnxruntime" / "test" / "testdata"
        self.e2e_models = self.test_data_root / "ort_minimal_e2e_test_data"
        self.e2e_models.mkdir(parents=True)
        self.standalone_ops = "test.customop;1;CustomOpOne\n"
        (self.e2e_models / "required_ops.standalone_invoker.config").write_text(self.standalone_ops, encoding="utf-8")
        custom_models = self.test_data_root / "custom_op_library"
        custom_models.mkdir()
        (custom_models / "test.onnx").write_bytes(b"model")
        self.wheel_dir = self.build_dir / "Debug" / "dist"
        self.wheel_dir.mkdir(parents=True)
        self.wheel = self.wheel_dir / "onnxruntime-1.0-cp312-cp312-linux_x86_64.whl"
        self.wheel.touch()
        self.custom_library = self.build_dir / "Debug" / "libcustom_op_library.so"
        self.custom_library.touch()
        self.custom_output = None
        self.conversion_cwd = None

        self.addCleanup(patch.stopall)
        patch.object(prepare, "REPO_DIR", self.repo).start()
        self.run = patch.object(prepare, "run", side_effect=self.fake_run).start()

    def fake_run(self, *args, cwd):
        if len(args) > 1 and Path(args[1]).name == "create_reduced_build_config.py":
            Path(args[-1]).write_text("# Generated\nai.onnx;1;Add\n", encoding="utf-8")
        if "--custom_op_library" in args:
            self.custom_output = Path(args[-1])
            self.conversion_cwd = Path(cwd)
            self.assertEqual((self.custom_output / "test.onnx").read_bytes(), b"model")

    def test_preparation_preserves_commands_and_configs_without_training_ops(self):
        prepare.prepare_test_data(self.build_dir)
        calls = self.run.call_args_list
        commands = [call.args for call in calls]
        self.assertEqual(
            commands[:6],
            [
                (
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    *([] if sys.prefix != sys.base_prefix else ["--user"]),
                    "-r",
                    str(self.repo / "tools" / "ci_build" / "github" / "linux" / "python" / "requirements.txt"),
                ),
                (sys.executable, str(self.repo / "tools" / "ci_build" / "op_registration_validator.py")),
                ("ccache", "--version"),
                ("ccache", "--zero-stats"),
                (
                    sys.executable,
                    str(self.repo / "tools" / "ci_build" / "build.py"),
                    "--build_dir",
                    str(self.build_dir),
                    "--cmake_generator",
                    "Ninja",
                    "--config",
                    "Debug",
                    "--skip_submodule_sync",
                    "--parallel",
                    "--use_cache",
                    "--use_vcpkg",
                    "--use_vcpkg_ms_internal_asset_cache",
                    "--use_binskim_compliant_compile_flags",
                    "--build_wheel",
                    "--skip_tests",
                    "--use_nnapi",
                    "--use_coreml",
                ),
                ("ccache", "--show-stats", "-vv"),
            ],
        )
        self.assertEqual(commands[6][-3:], ("--force-reinstall", "--no-deps", str(self.wheel)))
        self.assertNotIn("--find-links", commands[6])
        self.assertEqual(
            commands[7],
            (
                sys.executable,
                str(self.repo / "tools" / "python" / "convert_onnx_models_to_ort.py"),
                str(self.e2e_models),
            ),
        )
        self.assertEqual(
            commands[8],
            (sys.executable, "-m", "onnxruntime.tools.convert_onnx_models_to_ort", str(self.e2e_models)),
        )
        self.assertNotIn("--enable_type_reduction", commands[9])
        self.assertIn("--enable_type_reduction", commands[10])
        self.assertEqual(commands[9][-2], str(self.test_data_root))
        self.assertEqual(commands[10][-2], str(self.test_data_root))
        self.assertEqual(commands[11][-3:], ("--custom_op_library", str(self.custom_library), str(self.custom_output)))
        self.assertEqual(len(commands), 12)

        for call in calls[7:]:
            cwd = Path(call.kwargs["cwd"])
            self.assertTrue(cwd.is_absolute())
            self.assertFalse(cwd.is_relative_to(self.repo))
            self.assertEqual(cwd, self.conversion_cwd)
        test_data_dir = self.build_dir / "minimal_build_test_data"
        self.assertEqual(
            {path.name for path in test_data_dir.iterdir()},
            {"required_ops.ort_models.config", "required_ops_and_types.ort_models.config"},
        )
        for config in test_data_dir.iterdir():
            self.assertEqual(config.read_text(encoding="utf-8"), "# Generated\nai.onnx;1;Add\n\n" + self.standalone_ops)
        self.assertFalse(self.custom_output.exists())
        self.assertFalse(self.conversion_cwd.exists())

    def test_missing_custom_library_warns_and_skips_only_custom_conversion(self):
        self.custom_library.unlink()
        with self.assertLogs(prepare.log, level="WARNING") as logs:
            prepare.prepare_test_data(self.build_dir)
        self.assertIn("skipping custom-op conversion", logs.output[0])
        self.assertEqual(self.run.call_count, 11)

    def test_conversion_failure_propagates_and_cleans_up(self):
        def fail_conversion(*args, cwd):
            self.fake_run(*args, cwd=cwd)
            if "--custom_op_library" in args:
                raise subprocess.CalledProcessError(1, args)

        self.run.side_effect = fail_conversion
        with self.assertRaises(subprocess.CalledProcessError):
            prepare.prepare_test_data(self.build_dir)
        self.assertFalse(self.custom_output.exists())
        self.assertFalse(self.conversion_cwd.exists())

    def test_missing_custom_models_fails_and_cleans_up(self):
        (self.test_data_root / "custom_op_library" / "test.onnx").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "No custom-op ONNX models"):
            prepare.prepare_test_data(self.build_dir)
        self.assertFalse(list((self.build_dir / "minimal_build_test_data").glob("custom_ops_model_*")))

    def test_wheel_selection_rejects_missing_or_ambiguous_wheels(self):
        for count in (0, 2):
            with self.subTest(count=count):
                self.wheel.unlink(missing_ok=True)
                extra = self.wheel_dir / "onnxruntime-2.0-cp312-cp312-linux_x86_64.whl"
                if count == 2:
                    self.wheel.touch()
                    extra.touch()
                with self.assertRaisesRegex(RuntimeError, f"found {count}"):
                    prepare.prepare_test_data(self.build_dir)
                self.assertFalse(any("--force-reinstall" in call.args for call in self.run.call_args_list))
                self.run.reset_mock()
                extra.unlink(missing_ok=True)

    def test_build_failure_stops_preparation(self):
        def fail_build(*args, cwd):
            if len(args) > 1 and Path(args[1]).name == "build.py":
                raise subprocess.CalledProcessError(1, args)
            self.fake_run(*args, cwd=cwd)

        self.run.side_effect = fail_build
        with self.assertRaises(subprocess.CalledProcessError):
            prepare.prepare_test_data(self.build_dir)
        self.assertEqual(self.run.call_count, 5)

    def test_virtual_environment_omits_user_install(self):
        with patch.object(sys, "prefix", "test-venv"), patch.object(sys, "base_prefix", "test-base"):
            prepare.prepare_test_data(self.build_dir)
        for call in self.run.call_args_list:
            self.assertNotIn("--user", call.args)


if __name__ == "__main__":
    unittest.main()
