import importlib.util
import subprocess
import sys
import tempfile
import types
import unittest
import unittest.mock
from pathlib import Path

MODULE_PATH = Path(__file__).with_name("compile_triton.py")


def load_compile_triton_module():
    fake_triton = types.SimpleNamespace()
    with unittest.mock.patch.dict(sys.modules, {"triton": fake_triton}):
        spec = importlib.util.spec_from_file_location("compile_triton_under_test", MODULE_PATH)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    return module


class CompileTritonCommandExecutionTest(unittest.TestCase):
    def setUp(self):
        self.module = load_compile_triton_module()

    def test_convert_lib_to_obj_uses_subprocess_run(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            lib_file = "kernel.cubin"
            obj_file = "kernel.o"
            Path(tmp_dir, lib_file).write_text("binary-data", encoding="utf-8")
            Path(tmp_dir, obj_file).write_text("", encoding="utf-8")

            with (
                unittest.mock.patch.object(self.module.os, "system", return_value=0) as mock_system,
                unittest.mock.patch(
                    "subprocess.run", return_value=subprocess.CompletedProcess(args=[], returncode=0)
                ) as mock_run,
            ):
                result = self.module.convert_lib_to_obj(lib_file, tmp_dir)

            self.assertEqual(result, obj_file)
            mock_system.assert_not_called()
            mock_run.assert_called_once()
            command = mock_run.call_args.args[0]
            kwargs = mock_run.call_args.kwargs
            self.assertEqual(
                command,
                ["objcopy", "-I", "binary", "-O", "elf64-x86-64", "-B", "i386:x86-64", lib_file, obj_file],
            )
            self.assertEqual(kwargs["cwd"], tmp_dir)
            self.assertTrue(kwargs["check"])

    def test_archive_obj_files_uses_subprocess_run(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_obj_file = "triton_kernel.a"
            obj_files = ["kernel_0.o", "kernel_1.o"]
            for name in [*obj_files, out_obj_file]:
                Path(tmp_dir, name).write_text("", encoding="utf-8")

            with (
                unittest.mock.patch.object(self.module.os, "system", return_value=0) as mock_system,
                unittest.mock.patch(
                    "subprocess.run", return_value=subprocess.CompletedProcess(args=[], returncode=0)
                ) as mock_run,
            ):
                self.module.archive_obj_files(obj_files, tmp_dir, out_obj_file)

            mock_system.assert_not_called()
            mock_run.assert_called_once()
            command = mock_run.call_args.args[0]
            kwargs = mock_run.call_args.kwargs
            self.assertEqual(command, ["ar", "rcs", out_obj_file, *obj_files])
            self.assertEqual(kwargs["cwd"], tmp_dir)
            self.assertTrue(kwargs["check"])

    def test_convert_lib_to_obj_surfaces_subprocess_errors(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            lib_file = "kernel.cubin"
            Path(tmp_dir, lib_file).write_text("binary-data", encoding="utf-8")

            with (
                unittest.mock.patch(
                    "subprocess.run",
                    side_effect=subprocess.CalledProcessError(1, ["objcopy"]),
                ) as mock_run,
                self.assertRaisesRegex(RuntimeError, "objcopy.*kernel.o"),
            ):
                self.module.convert_lib_to_obj(lib_file, tmp_dir)

            mock_run.assert_called_once()

    def test_archive_obj_files_surfaces_os_errors(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_obj_file = "triton_kernel.a"
            obj_files = ["kernel_0.o", "kernel_1.o"]

            with (
                unittest.mock.patch(
                    "subprocess.run",
                    side_effect=FileNotFoundError("ar not found"),
                ) as mock_run,
                self.assertRaisesRegex(RuntimeError, "ar.*ar not found"),
            ):
                self.module.archive_obj_files(obj_files, tmp_dir, out_obj_file)

            mock_run.assert_called_once()


if __name__ == "__main__":
    unittest.main()
