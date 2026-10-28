# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import hashlib
import importlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import mock

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))
build = importlib.import_module("build")


def kernel(provider, types=("tensor(float)",), domain=""):
    return SimpleNamespace(
        op_name="Add",
        domain=domain,
        provider=provider,
        version_range=(14, 2147483647),
        type_constraints={"T": list(types)},
    )


class KernelDocumentationTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.output = self.root / "OperatorKernels.md"
        self.output.write_text("existing documentation\n", encoding="utf-8")
        self.metadata = self.root / "metadata.json"
        self.library = self.root / "provider.dll"
        self.library.write_bytes(b"test plugin")
        self.plugin = f"WebGpuExecutionProvider:{self.library}"

        # Test the generator without requiring an installed ORT runtime or a GPU.
        self.ort = ModuleType("onnxruntime")
        self.ort.__version__ = "test-runtime"
        self.ort.register_execution_provider_library = mock.Mock()
        self.ort.get_ep_devices = mock.Mock(
            return_value=[SimpleNamespace(ep_name="WebGpuExecutionProvider", ep_metadata={"version": "test-plugin"})]
        )
        self.rtpy = ModuleType("onnxruntime.capi.onnxruntime_pybind11_state")
        self.rtpy.get_all_operator_schema = mock.Mock(
            return_value=[
                SimpleNamespace(
                    name="Add",
                    domain="",
                    inputs=[SimpleNamespace(name=name, typeStr="T") for name in ("A", "B")],
                    outputs=[SimpleNamespace(name="C", typeStr="T")],
                )
            ]
        )
        self.rtpy.get_all_opkernel_def = mock.Mock(return_value=[kernel("CPUExecutionProvider")])
        self.rtpy.get_registered_ep_kernel_defs = mock.Mock(
            return_value=[kernel("WebGpuExecutionProvider", ("tensor(float)", "tensor(float16)"))]
        )
        self.ort.capi = ModuleType("onnxruntime.capi")
        self.ort.capi.onnxruntime_pybind11_state = self.rtpy
        modules = {
            "onnxruntime": self.ort,
            "onnxruntime.capi": self.ort.capi,
            "onnxruntime.capi.onnxruntime_pybind11_state": self.rtpy,
        }
        spec = importlib.util.spec_from_file_location(
            "gen_opkernel_doc_under_test", REPO_DIR / "tools" / "python" / "gen_opkernel_doc.py"
        )
        self.generator = importlib.util.module_from_spec(spec)
        with mock.patch.dict(sys.modules, modules):
            spec.loader.exec_module(self.generator)

    def generate(self, plugins=None, providers=("CPU", "WebGPU")):
        self.generator.main(self.output, providers, plugins, self.metadata)

    def assert_existing_document_preserved(self):
        self.assertEqual(self.output.read_text(encoding="utf-8"), "existing documentation\n")
        self.assertFalse(self.metadata.exists())

    def test_builtin_and_plugin_tables_and_provenance(self):
        self.generate([self.plugin])
        document = self.output.read_text(encoding="utf-8")
        self.assertIn("## Operators implemented by CPUExecutionProvider", document)
        self.assertIn("## Operators implemented by WebGpuExecutionProvider", document)
        self.assertIn("*in* A:**T**<br> *in* B:**T**<br> *out* C:**T**", document)
        self.assertIn("|14+|**T** = tensor(float), tensor(float16)|", document)
        self.ort.register_execution_provider_library.assert_called_once_with(
            "WebGpuExecutionProvider", str(self.library)
        )
        metadata = json.loads(self.metadata.read_text(encoding="utf-8"))
        self.assertEqual(metadata["onnxruntime_version"], "test-runtime")
        self.assertEqual(metadata["session_options"], "default")
        self.assertEqual(metadata["plugins"][0]["version"], "test-plugin")
        self.assertEqual(metadata["plugins"][0]["sha256"], hashlib.sha256(b"test plugin").hexdigest())
        self.assertEqual(metadata["providers"]["WebGpuExecutionProvider"], {"operators": 1, "registrations": 1})
        self.assertEqual(metadata["document_sha256"], hashlib.sha256(self.output.read_bytes()).hexdigest())

    def test_builtin_only_generation_does_not_query_plugins(self):
        self.generate(providers=("cpu",))
        self.ort.register_execution_provider_library.assert_not_called()
        self.rtpy.get_registered_ep_kernel_defs.assert_not_called()
        self.ort.get_ep_devices.assert_not_called()
        metadata = json.loads(self.metadata.read_text(encoding="utf-8"))
        self.assertEqual(list(metadata["providers"]), ["CPUExecutionProvider"])
        self.assertEqual(metadata["plugins"], [])

    def test_filter_and_registration_order_are_deterministic(self):
        registrations = [
            kernel("WebGpuExecutionProvider"),
            kernel("WebGpuExecutionProvider", ("tensor(float16)",)),
        ]
        self.rtpy.get_registered_ep_kernel_defs.return_value = registrations
        self.generate([self.plugin], ("webgpu",))
        first = self.output.read_bytes()
        self.rtpy.get_registered_ep_kernel_defs.return_value = registrations[::-1]
        self.generate([self.plugin], ("WebGpuExecutionProvider",))
        self.assertEqual(self.output.read_bytes(), first)
        self.assertNotIn(b"CPUExecutionProvider", first)
        self.assertEqual(first.count(b"|14+|"), 1)
        metadata = json.loads(self.metadata.read_text(encoding="utf-8"))
        self.assertEqual(metadata["providers"]["WebGpuExecutionProvider"]["registrations"], 2)

    def test_windows_library_path_keeps_drive_colon(self):
        path = r"C:\build output\provider.dll"
        with mock.patch.object(Path, "is_file", return_value=True):
            self.generator.load_plugin_ep_kernel_defs([f"WebGpuExecutionProvider:{path}"])
        self.ort.register_execution_provider_library.assert_called_once_with("WebGpuExecutionProvider", path)

    def test_invalid_plugin_specs_preserve_existing_document(self):
        for spec in ("WebGPU", ":provider.dll", "WebGpuExecutionProvider:"):
            with self.subTest(spec=spec), self.assertRaisesRegex(ValueError, "expected NAME:PATH"):
                self.generate([spec])
            self.assert_existing_document_preserved()

    def test_missing_library_preserves_existing_document(self):
        self.library.unlink()
        with self.assertRaisesRegex(FileNotFoundError, "Plugin EP library does not exist"):
            self.generate([self.plugin])
        self.assert_existing_document_preserved()
        self.ort.register_execution_provider_library.assert_not_called()

    def test_registration_failure_is_not_ignored(self):
        for message in ("cannot load library", "EP library is already registered under WebGpuExecutionProvider"):
            self.ort.register_execution_provider_library.side_effect = RuntimeError(message)
            with self.subTest(message=message), self.assertRaisesRegex(RuntimeError, message):
                self.generate([self.plugin])
            self.assert_existing_document_preserved()
        self.rtpy.get_registered_ep_kernel_defs.assert_not_called()

    def test_query_failure_preserves_existing_document(self):
        self.rtpy.get_registered_ep_kernel_defs.side_effect = RuntimeError("No devices found for EP")
        with self.assertRaisesRegex(RuntimeError, "No devices found"):
            self.generate([self.plugin])
        self.assert_existing_document_preserved()

    def test_empty_plugin_registry_is_an_error(self):
        self.rtpy.get_registered_ep_kernel_defs.return_value = []
        with self.assertRaisesRegex(RuntimeError, "returned no registered kernels"):
            self.generate([self.plugin])
        self.assert_existing_document_preserved()

    def test_wrong_provider_registry_is_an_error(self):
        self.rtpy.get_registered_ep_kernel_defs.return_value = [kernel("CPUExecutionProvider")]
        with self.assertRaisesRegex(RuntimeError, "different execution provider"):
            self.generate([self.plugin])
        self.assert_existing_document_preserved()

    def test_metadata_cannot_overwrite_document(self):
        with self.assertRaisesRegex(ValueError, "output paths must be different"):
            self.generator.main(self.output, ["CPU"], metadata_path=self.output)
        self.assert_existing_document_preserved()


class BuildDocumentationTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.source = self.root / "source"
        self.build_dir = self.root / "build"
        self.config = "RelWithDebInfo"
        self.github_output = self.root / "github-output"
        self.source.joinpath("docs").mkdir(parents=True)
        self.source.joinpath("tools", "python").mkdir(parents=True)
        for name in ("gen_contrib_doc.py", "gen_opkernel_doc.py"):
            self.source.joinpath("tools", "python", name).write_text("# test generator\n", encoding="utf-8")
        for name in ("OperatorKernels.md", "ContribOperators.md"):
            self.source.joinpath("docs", name).write_text("existing documentation\n", encoding="utf-8")
        self.cwd = self.build_dir / self.config / self.config
        self.cwd.mkdir(parents=True)
        self.cwd.joinpath("onnxruntime_providers_webgpu.dll").write_bytes(b"test plugin")
        self.calls = []
        self.diff = b""
        self.fail_query = False

    def subprocess(self, command, **kwargs):
        self.calls.append((command, kwargs))
        is_kernel_generator = command[1] == "gen_opkernel_doc.py"
        if is_kernel_generator and self.fail_query:
            raise subprocess.CalledProcessError(1, command)
        output = Path(command[command.index("--output_path") + 1])
        output.write_text(f"generated {output.name}\n", encoding="utf-8")
        if is_kernel_generator:
            metadata = Path(command[command.index("--metadata_path") + 1])
            metadata.write_text('{"onnxruntime_version": "test-runtime"}', encoding="utf-8")

    def git_output(self, command, **_kwargs):
        if command[1] == "rev-parse":
            return b"test-source-commit\n"
        self.assertEqual(command[:3], ["git", "diff", "--ignore-blank-lines"])
        return self.diff

    def generate(self, validate=True, use_webgpu="shared_lib", platform_name="windows"):
        with (
            mock.patch.object(build, "is_windows", return_value=platform_name == "windows"),
            mock.patch.object(build, "is_macOS", return_value=platform_name == "macos"),
            mock.patch.object(build, "run_subprocess", side_effect=self.subprocess),
            mock.patch.object(build.subprocess, "check_output", side_effect=self.git_output),
            mock.patch.dict(os.environ, {"GITHUB_OUTPUT": str(self.github_output)}),
        ):
            build.generate_documentation(str(self.source), str(self.build_dir), [self.config], validate, use_webgpu)

    def test_webgpu_uses_matching_built_library_and_publishes_metadata(self):
        self.generate()
        command, kwargs = self.calls[1]
        self.assertEqual(kwargs["cwd"], str(self.cwd))
        self.assertEqual(
            command[command.index("--providers") + 1 : command.index("--plugin-ep")],
            ["CPU", "CUDA", "DML", "WebGPU"],
        )
        self.assertEqual(command[-1], f"WebGpuExecutionProvider:{self.cwd / 'onnxruntime_providers_webgpu.dll'}")
        artifact_dir = self.cwd / "generated-docs"
        metadata = json.loads((artifact_dir / "generation-metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(metadata["source_commit"], "test-source-commit")
        self.assertEqual(metadata["build_configuration"], self.config)
        for name in ("OperatorKernels.md", "ContribOperators.md"):
            content = (artifact_dir / name).read_bytes()
            self.assertEqual(content, self.source.joinpath("docs", name).read_bytes())
            self.assertEqual(metadata["documents"][name], hashlib.sha256(content).hexdigest())
        self.assertEqual(self.github_output.read_text(encoding="utf-8"), "documentation_generated=true\n")

    def test_stale_docs_fail_after_artifact_is_ready(self):
        self.diff = b"documentation changed"
        with self.assertRaisesRegex(build.BuildError, "Generated documents have diffs"):
            self.generate()
        self.assertTrue(self.cwd.joinpath("generated-docs", "OperatorKernels.md").is_file())
        self.assertTrue(self.cwd.joinpath("generated-docs", "ContribOperators.md").is_file())
        self.assertTrue(self.cwd.joinpath("generated-docs", "generation-metadata.json").is_file())
        self.assertIn("documentation_generated=true", self.github_output.read_text(encoding="utf-8"))

    def test_generation_failure_does_not_signal_artifact_readiness(self):
        self.fail_query = True
        with self.assertRaises(subprocess.CalledProcessError):
            self.generate()
        self.assertFalse(self.github_output.exists())
        self.assertFalse(self.cwd.joinpath("generated-docs", "OperatorKernels.md").exists())

    def test_missing_built_plugin_fails_before_generating_docs(self):
        self.cwd.joinpath("onnxruntime_providers_webgpu.dll").unlink()
        with self.assertRaisesRegex(build.BuildError, "requires the built plugin library"):
            self.generate()
        self.assertEqual(self.calls, [])
        self.assertFalse(self.github_output.exists())

    def test_in_tree_webgpu_requires_plugin_build(self):
        for mode in ("static_lib", "static_plugin"):
            with self.subTest(mode=mode), self.assertRaisesRegex(build.BuildError, "--use_webgpu shared_lib"):
                self.generate(use_webgpu=mode)
        self.assertEqual(self.calls, [])

    def test_non_webgpu_generation_retains_existing_provider_selection(self):
        self.generate(validate=False, use_webgpu=None)
        command = self.calls[1][0]
        self.assertNotIn("--plugin-ep", command)
        self.assertEqual(command[command.index("--providers") + 1 :], ["CPU", "CUDA", "DML"])
        self.assertTrue(self.github_output.exists())

    def test_native_plugin_library_names(self):
        for platform_name, name in (
            ("linux", "libonnxruntime_providers_webgpu.so"),
            ("macos", "libonnxruntime_providers_webgpu.dylib"),
        ):
            with self.subTest(platform_name=platform_name):
                cwd = self.build_dir / self.config
                cwd.joinpath(name).write_bytes(b"test plugin")
                self.generate(validate=False, platform_name=platform_name)
                command, kwargs = self.calls[-1]
                self.assertEqual(kwargs["cwd"], str(cwd))
                self.assertEqual(command[-1], f"WebGpuExecutionProvider:{cwd / name}")


if __name__ == "__main__":
    unittest.main()
