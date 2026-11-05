# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import json
import tempfile
import unittest
import zipfile
from email.parser import Parser
from pathlib import Path

import python_package_tree
import python_wheel_packager


class PythonPackagerTest(unittest.TestCase):
    def test_package_tree_and_gpu_wheel_use_source_version(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            package_init = source / "__init__.py"
            package_init.write_text('__version__ = "0.0.0"\n', encoding="utf-8")
            tools_init = source / "tools_init.py"
            tools_init.write_text("# tools\n", encoding="utf-8")
            tools_append = source / "tools_append.py"
            tools_append.write_text("have_torch = False\n", encoding="utf-8")
            loader = source / "_pybind_state.py.in"
            loader.write_text(
                "@ONNXRUNTIME_SETDLOPENFLAGS_GLOBAL@\n"
                "from .onnxruntime_pybind11_state import *\n"
                "@ONNXRUNTIME_SETDLOPENFLAGS_LOCAL@\n",
                encoding="utf-8",
            )
            preload_source = source / "_ld_preload.py"
            preload_source.write_text("# preload dependencies\n", encoding="utf-8")
            version = root / "VERSION_NUMBER"
            version.write_text("1.2.3\n", encoding="utf-8")
            native = source / "onnxruntime_pybind11_state.so"
            native.write_bytes(b"native")
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "version_file": str(version),
                        "package_name": "onnxruntime-gpu",
                        "cuda_enabled": True,
                        "cuda_version": "13.0",
                        "linux": True,
                        "tools_init_append": str(tools_append),
                        "files": [
                            {"source": str(package_init), "destination": "onnxruntime/__init__.py"},
                            {"source": str(tools_init), "destination": "onnxruntime/tools/__init__.py"},
                            {"source": str(loader), "destination": "onnxruntime/capi/_pybind_state.py"},
                            {"source": str(preload_source), "destination": "onnxruntime/capi/_ld_preload.py"},
                            {
                                "source": str(native),
                                "destination": "onnxruntime/capi/onnxruntime_pybind11_state.so",
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )
            package_tree = root / "package"
            package_tree.mkdir()
            python_package_tree.assemble(manifest, package_tree)
            self.assertIn('__version__ = "1.2.3"', (package_tree / "onnxruntime/__init__.py").read_text())
            self.assertIn("have_torch = False", (package_tree / "onnxruntime/tools/__init__.py").read_text())
            self.assertEqual(
                "package_name = 'onnxruntime-gpu'\n__version__ = '1.2.3'\ncuda_version = '13.0'\n",
                (package_tree / "onnxruntime/capi/build_and_package_info.py").read_text(),
            )
            preload = (package_tree / "onnxruntime/capi/_ld_preload.py").read_text()
            self.assertIn('["libcudart.so.13"]', preload)
            self.assertIn('os.environ["ORT_CUDA_UNAVAILABLE"] = "1"', preload)

            requirements = root / "requirements.txt"
            requirements.write_text("numpy >= 1.21.6\npackaging\n", encoding="utf-8")
            description = root / "README.rst"
            description.write_text("ONNX Runtime\n", encoding="utf-8")
            license_file = root / "LICENSE"
            license_file.write_text("MIT\n", encoding="utf-8")
            output = root / "dist"
            output.mkdir()
            args = argparse.Namespace(
                package_tree=package_tree,
                version_file=version,
                requirements_file=requirements,
                description_file=description,
                license_file=[license_file],
                output=output,
                distribution="onnxruntime-gpu",
                python_tag="cp314",
                abi="cp314",
                platform="linux_x86_64",
                python_requires=">=3.11",
                cuda_version="13.0",
            )
            wheel_path = python_wheel_packager.build_wheel(args)
            self.assertEqual("onnxruntime_gpu-1.2.3-cp314-cp314-linux_x86_64.whl", wheel_path.name)
            with zipfile.ZipFile(wheel_path) as wheel:
                metadata = Parser().parsestr(wheel.read("onnxruntime_gpu-1.2.3.dist-info/METADATA").decode("utf-8"))
                self.assertEqual("onnxruntime-gpu", metadata["Name"])
                self.assertEqual("1.2.3", metadata["Version"])
                self.assertIn("nvidia-cuda-runtime~=13.0; extra == 'cuda'", metadata.get_all("Requires-Dist"))
                self.assertIn("nvidia-cudnn-cu13~=9.0; extra == 'cudnn'", metadata.get_all("Requires-Dist"))

    def test_package_tree_rejects_nonempty_precreated_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "package"
            output.mkdir()
            (output / "stale.txt").write_text("stale", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "must be empty"):
                python_package_tree._prepare_output(output)

    def test_linux_cuda_preload_rejects_unsupported_major(self):
        with tempfile.TemporaryDirectory() as directory:
            preload = Path(directory) / "_ld_preload.py"
            preload.write_text("", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "support CUDA major versions 12 and 13"):
                python_package_tree._append_linux_cuda_preload(preload, "11.8")

    def test_wheel_rejects_nonempty_precreated_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "dist"
            output.mkdir()
            (output / "stale.whl").write_text("stale", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "must be empty"):
                python_wheel_packager._prepare_output(output)

    def test_wheel_rejects_file_as_output_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "dist"
            output.write_text("not a directory", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not a directory"):
                python_wheel_packager._prepare_output(output)


if __name__ == "__main__":
    unittest.main()
