#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools.ci_build.github.windows import dml_test_artifacts as artifacts


class DmlTestArtifactsTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.build = self.root / "producer with spaces" / artifacts.CONFIG
        self.runtime = self.build / artifacts.CONFIG
        self.runtime.mkdir(parents=True)
        for name in (
            "onnxruntime_test_all.exe",
            "onnxruntime_provider_test.exe",
            "onnxruntime.dll",
            "DirectML.dll",
            "custom_op_library.dll",
            "onnxruntime_test_python.py",
            "onnxruntime_test_python_iobinding.py",
            "onnxruntime_test_python_dmlgraph.py",
            "testdata/model.onnx",
            "onnxruntime/capi/onnxruntime_pybind11_state.pyd",
            "onnxruntime/capi/DirectML.dll",
            "dist/onnxruntime_directml.whl",
            "D3D12/D3D12Core.dll",
            ".hidden-fixture",
            "build.obj",
            "build.lib",
        ):
            path = self.runtime / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"fixture")
        self.onnx = self.build / "vcpkg_installed" / "x64-windows" / "include"
        self.data = self.onnx / "onnx" / "backend" / "test" / "data"
        self.data.mkdir(parents=True)
        (self.data / "model.onnx").write_bytes(b"onnx fixture")
        (self.build / "CMakeCache.txt").write_text(f"onnx_SOURCE_DIR:PATH={self.onnx.as_posix()}\n", encoding="utf-8")
        self.ctest = self.build / "CTestTestfile.cmake"
        self.ctest.write_text(
            f"# Source directory: {artifacts.REPO_ROOT.as_posix()}/cmake\n"
            f"# Build directory: {self.build.as_posix()}\n"
            f'add_test(relocation "{Path(sys.executable).as_posix()}" "-c" '
            "\"from pathlib import Path; assert Path('testdata/model.onnx').is_file()\")\n"
            f'set_tests_properties(relocation PROPERTIES WORKING_DIRECTORY "{self.runtime.as_posix()}")\n'
            'subdirs("nested")\n',
            encoding="utf-8",
        )
        nested = self.build / "nested" / "CTestTestfile.cmake"
        nested.parent.mkdir()
        nested.write_text(
            f"# Nested source: {artifacts.REPO_ROOT.as_posix()}\n"
            f"# Escaped build: {str(self.build).replace(chr(92), chr(92) * 2)}\n"
            f'add_test(onnx_fixture "{Path(sys.executable).as_posix()}" "-c" '
            '"from pathlib import Path; import sys; assert Path(sys.argv[1]).is_file()" '
            f'"{self.data.as_posix()}/model.onnx")\n',
            encoding="utf-8",
        )
        self.output = self.root / "artifact"

    def test_round_trip_retains_runtime_fixtures_and_relocates_ctest(self):
        artifacts.stage(self.build, self.output)
        self.assertEqual((self.output / self.data.relative_to(self.build) / "model.onnx").read_bytes(), b"onnx fixture")
        for name in ("custom_op_library.dll", "D3D12/D3D12Core.dll", ".hidden-fixture"):
            self.assertTrue((self.output / artifacts.CONFIG / name).is_file())
        for name in ("build.obj", "build.lib"):
            self.assertFalse((self.output / artifacts.CONFIG / name).exists())
        self.assertFalse((self.output / "CMakeCache.txt").exists())
        moved = self.root / "gpu runner with spaces" / artifacts.CONFIG
        moved.parent.mkdir()
        shutil.move(self.output, moved)
        shutil.rmtree(self.build)
        with mock.patch.object(artifacts, "REPO_ROOT", self.root / "gpu checkout"):
            artifacts.prepare(moved)
        for ctest in moved.rglob("CTestTestfile.cmake"):
            text = ctest.read_text(encoding="utf-8")
            self.assertNotIn(self.build.as_posix(), text)
            self.assertNotIn(str(self.build).replace("\\", "\\\\"), text)
            self.assertNotIn(artifacts.REPO_ROOT.as_posix(), text.replace(Path(sys.executable).as_posix(), ""))
        result = subprocess.run(
            ["ctest", "--test-dir", str(moved), "-C", artifacts.CONFIG, "--output-on-failure"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("100% tests passed", result.stdout)
        self.assertIn("2/2", result.stdout)

    def test_relocates_python_interpreter(self):
        artifacts.stage(self.build, self.output)
        paths = json.loads((self.output / artifacts.MANIFEST).read_text(encoding="utf-8"))
        paths["python"] = "C:/producer/python.exe"
        (self.output / artifacts.MANIFEST).write_text(json.dumps(paths), encoding="utf-8")
        self.ctest = self.output / "CTestTestfile.cmake"
        self.ctest.write_text('add_test(python "C:/producer/python.exe")', encoding="utf-8")
        artifacts.prepare(self.output)
        self.assertIn(Path(sys.executable).as_posix(), self.ctest.read_text(encoding="utf-8"))

    def test_missing_required_artifacts_fail_staging_and_prepare(self):
        for name in ("DirectML.dll", "onnxruntime_provider_test.exe", "onnxruntime_test_python_dmlgraph.py"):
            with self.subTest(name=name):
                path = self.runtime / name
                contents = path.read_bytes()
                path.unlink()
                with self.assertRaises(FileNotFoundError):
                    artifacts.stage(self.build, self.output)
                with self.assertRaises(FileNotFoundError):
                    artifacts.prepare(self.build)
                path.write_bytes(contents)

    def test_missing_or_ambiguous_wheel_fails(self):
        wheel = self.runtime / "dist" / "onnxruntime_directml.whl"
        wheel.unlink()
        with self.assertRaises(ValueError):
            artifacts.stage(self.build, self.output)
        wheel.write_bytes(b"wheel")
        wheel.with_name("another.whl").write_bytes(b"wheel")
        with self.assertRaises(ValueError):
            artifacts.stage(self.build, self.output)


if __name__ == "__main__":
    unittest.main()
