import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np
import onnx
from create_cases_from_json import case_directory_name, generate
from onnx import numpy_helper

ROOT = Path(__file__).resolve().parents[1]
SETTINGS = None


class PilotTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.cases = self.root / "cases"
        generate(ROOT / "cases/pilot.json", self.cases)
        self.profile = {"schema_version": 1, "kind": "builtin", "ep_name": "CPUExecutionProvider"}
        self.scalar_id = "ai.onnx.Mul.opset14.float32_scalar"
        self.scalar = self.cases / case_directory_name(self.scalar_id)
        self.add = self.cases / case_directory_name("ai.onnx.Add.opset14.float32_broadcast")

    def replay(self, expected_code=0, cases=None):
        profile = self.root / "profile.json"
        report = self.root / "report.json"
        profile.write_text(json.dumps(self.profile), encoding="utf-8")
        command = [
            str(SETTINGS.runner),
            "--runtime",
            str(SETTINGS.runtime),
            "--profile",
            str(profile),
            "--cases",
            str(cases or self.cases),
            "--report",
            str(report),
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, expected_code, result.stdout + result.stderr)
        return json.loads(report.read_text(encoding="utf-8"))

    def manifest(self, **changes):
        path = self.scalar / "case.json"
        document = json.loads(path.read_text(encoding="utf-8"))
        document.update(changes)
        path.write_text(json.dumps(document), encoding="utf-8")

    def wrong_expected(self):
        path = self.scalar / "test_data_set_0/output_0.pb"
        expected = numpy_helper.to_array(onnx.load_tensor(path)).copy()
        tensor = numpy_helper.from_array(expected + np.float32(1), name="Y")
        path.write_bytes(tensor.SerializeToString())

    def plugin(self):
        if SETTINGS.plugin_library is None:
            self.skipTest("Pass --plugin-library to validate plugin execution")
        self.profile = {
            "schema_version": 1,
            "kind": "plugin",
            "ep_name": "ConformanceExampleEP",
            "registration_name": "ConformanceExampleEP",
            "library": str(SETTINGS.plugin_library),
        }

    def test_cpu_assignment_and_execution(self):
        report = self.replay()
        self.assertEqual(report["passed"], 3)
        for result in report["results"]:
            self.assertEqual(result["status"], "PASS")
            self.assertEqual(result["assignment"][0]["ep_name"], "CPUExecutionProvider")
            manifest = json.loads(
                (self.cases / case_directory_name(result["id"]) / "case.json").read_text(encoding="utf-8")
            )
            self.assertEqual(result["source"], manifest["source"])

    def test_native_execution_without_source_or_python(self):
        staging = self.root / "native"
        staging.mkdir()
        runner = Path(shutil.copy2(SETTINGS.runner, staging))
        runtime = Path(shutil.copy2(SETTINGS.runtime, staging))
        shutil.copytree(self.cases, staging / "cases")
        (staging / "profile.json").write_text(json.dumps(self.profile), encoding="utf-8")
        environment = os.environ.copy()
        environment.pop("PYTHONPATH", None)
        environment["PATH"] = str(Path(os.environ["SYSTEMROOT"]) / "System32") if os.name == "nt" else "/usr/bin:/bin"
        command = [
            str(runner),
            "--runtime",
            runtime.name,
            "--profile",
            "profile.json",
            "--cases",
            "cases",
            "--report",
            "report.json",
        ]
        result = subprocess.run(command, cwd=staging, env=environment, capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(json.loads((staging / "report.json").read_text())["passed"], 3)
        if SETTINGS.plugin_library:
            plugin = Path(shutil.copy2(SETTINGS.plugin_library, staging))
            profile = {
                "schema_version": 1,
                "kind": "plugin",
                "ep_name": "ConformanceExampleEP",
                "registration_name": "ConformanceExampleEP",
                "library": str(plugin),
            }
            (staging / "profile.json").write_text(json.dumps(profile), encoding="utf-8")
            command[command.index("--cases") + 1] = str(Path("cases") / self.scalar.name)
            result = subprocess.run(command, cwd=staging, env=environment, capture_output=True, text=True, check=False)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            report = json.loads((staging / "report.json").read_text())
            self.assertEqual(report["passed"], 1)
            self.assertEqual(report["results"][0]["assignment"][0]["ep_name"], "ConformanceExampleEP")

    def test_wrong_expected_fails(self):
        self.wrong_expected()
        report = self.replay(1, self.scalar)
        self.assertEqual(report["results"][0]["phase"], "execution_or_comparison")

    def test_wrong_shape_fails(self):
        path = self.scalar / "test_data_set_0/output_0.pb"
        tensor = onnx.load_tensor(path)
        tensor.dims.append(1)
        path.write_bytes(tensor.SerializeToString())
        self.assertEqual(self.replay(1, self.scalar)["results"][0]["phase"], "execution_or_comparison")

    def test_missing_required_node_rejected(self):
        self.manifest(required_nodes=["missing"])
        self.assertIn("Required node absent", self.replay(2)["error"])

    def test_empty_datasets_rejected(self):
        self.manifest(datasets=[])
        self.assertIn("no datasets", self.replay(2)["error"])

    def test_unknown_schema_rejected(self):
        self.manifest(schema_version=2)
        self.assertEqual(self.replay(2)["status"], "ERROR")

    def test_fractional_repeat_count_rejected(self):
        self.manifest(repeat_count=1.5)
        self.assertIn("Unsupported field", self.replay(2)["error"])

    def test_excessive_repeat_count_rejected(self):
        self.manifest(repeat_count=2**32 + 1)
        self.assertIn("Unsupported field", self.replay(2)["error"])

    def test_removed_provider_name_rejected(self):
        self.profile["provider_name"] = "CPU"
        self.assertIn("Unsupported field", self.replay(2)["error"])

    def test_removed_cpu_kind_rejected(self):
        self.profile["kind"] = "cpu"
        self.assertIn("Unknown target kind", self.replay(2)["error"])

    def test_nonobject_options_rejected(self):
        self.profile["options"] = []
        self.assertEqual(self.replay(2)["status"], "ERROR")

    def test_multiple_datasets_execute(self):
        path = self.scalar / "case.json"
        document = json.loads(path.read_text(encoding="utf-8"))
        self.manifest(datasets=document["datasets"] * 2)
        self.assertEqual(self.replay(cases=self.scalar)["passed"], 1)

    def test_unrepresentable_definitions_rejected(self):
        original = json.loads((ROOT / "cases/pilot.json").read_text(encoding="utf-8"))
        mutations = (
            ("case", "repeat_count", 2),
            ("case", "attributes", {}),
            ("input", "dtype", "int64"),
            ("input", "initializer", "false"),
            ("input", "values", ["2"]),
            ("output", "initializer", True),
        )
        for target, field, value in mutations:
            with self.subTest(target=target, field=field):
                document = json.loads(json.dumps(original))
                case = document["cases"][0]
                definition = case if target == "case" else case[f"{target}s"][0]
                definition[field] = value
                path = self.root / "invalid-definition.json"
                path.write_text(json.dumps(document), encoding="utf-8")
                with self.assertRaises(ValueError):
                    generate(path, self.root / "invalid-cases")
                self.assertFalse((self.root / "invalid-cases").exists())

    def test_irrelevant_profile_fields_rejected(self):
        self.profile["library"] = "ignored.dll"
        self.assertIn("Unsupported field", self.replay(2)["error"])

    def test_unknown_field_rejected(self):
        self.manifest(ignored_assertion=True)
        self.assertIn("Unsupported field", self.replay(2)["error"])

    def test_invalid_source_rejected(self):
        path = self.scalar / "case.json"
        original = json.loads(path.read_text(encoding="utf-8"))
        for source in (None, {}, [], "source.json"):
            with self.subTest(source=source):
                self.manifest(source=source)
                self.assertIn("source provenance", self.replay(2)["error"])

        del original["source"]
        path.write_text(json.dumps(original), encoding="utf-8")
        self.assertEqual(self.replay(2)["status"], "ERROR")

    def test_parent_path_rejected(self):
        self.manifest(model="../model.onnx")
        self.assertIn("escape", self.replay(2)["error"])

    def test_invalid_tensor_rejected(self):
        path = self.scalar / "test_data_set_0/output_0.pb"
        tensor = onnx.load_tensor(path)
        tensor.raw_data = b""
        path.write_bytes(tensor.SerializeToString())
        self.assertIn("tensor data size", self.replay(2)["error"])

    def test_contradictory_tensor_payload_rejected(self):
        path = self.scalar / "test_data_set_0/output_0.pb"
        tensor = onnx.load_tensor(path)
        tensor.int32_data.append(6)
        path.write_bytes(tensor.SerializeToString())
        self.assertIn("float32", self.replay(2)["error"])

    def test_invalid_tolerance_rejected(self):
        self.manifest(comparison={"Y": {"atol": -1, "rtol": 0, "nan_equal": True}})
        self.assertEqual(self.replay(2)["status"], "ERROR")

    def test_nonfloat_initializer_rejected(self):
        case = self.cases / case_directory_name("ai.onnx.Mul.opset14.float32_initializer")
        path = case / "model.onnx"
        model = onnx.load_model(path)
        model.graph.initializer[0].data_type = onnx.TensorProto.INT64
        path.write_bytes(model.SerializeToString())
        self.assertIn("float32", self.replay(2, case)["error"])

    def test_custom_domain_rejected(self):
        path = self.scalar / "model.onnx"
        model = onnx.load_model(path)
        model.graph.node[0].domain = "custom"
        path.write_bytes(model.SerializeToString())
        self.assertIn("standard-domain", self.replay(2, self.scalar)["error"])

    def test_duplicate_case_rejected(self):
        shutil.copytree(self.scalar, self.cases / "duplicate")
        self.assertIn("Duplicate case ID", self.replay(2)["error"])

    def test_empty_selection_rejected(self):
        empty = self.root / "empty"
        empty.mkdir()
        self.assertIn("No cases selected", self.replay(2, empty)["error"])

    def test_unknown_skip_rejected(self):
        self.profile["skips"] = {"unknown": "not a waiver"}
        self.assertIn("Skip must name", self.replay(2)["error"])

    def test_all_skipped_is_not_green(self):
        self.profile["skips"] = {self.scalar_id: "test all-skipped disposition"}
        report = self.replay(1, self.scalar)
        self.assertEqual(report["passed"], 0)
        self.assertEqual(report["results"][0]["status"], "SKIP")

    def test_missing_provider_rejected(self):
        self.profile.update(kind="builtin", ep_name="MissingExecutionProvider")
        self.assertIn("unavailable", self.replay(2)["error"])

    def test_generation_is_deterministic_and_immutable(self):
        other = self.root / "other"
        generate(ROOT / "cases/pilot.json", other)

        def hashes(root):
            return {
                path.relative_to(root): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in root.rglob("*")
                if path.is_file()
            }

        self.assertEqual(hashes(self.cases), hashes(other))
        with self.assertRaisesRegex(ValueError, "already exists"):
            generate(ROOT / "cases/pilot.json", other)

    def test_cpp_export_cpu_replay(self):
        if SETTINGS.exported_cases is None:
            self.skipTest("Pass --exported-cases to validate the C++ producer")
        self.assertGreater(self.replay(cases=SETTINGS.exported_cases)["passed"], 0)

    def test_plugin_mul_and_assignment(self):
        self.plugin()
        report = self.replay(cases=self.scalar)
        self.assertEqual(report["passed"], 1)
        self.assertEqual(report["results"][0]["assignment"][0]["ep_name"], "ConformanceExampleEP")

    def test_plugin_initializer(self):
        self.plugin()
        report = self.replay(cases=self.cases / case_directory_name("ai.onnx.Mul.opset14.float32_initializer"))
        self.assertEqual(report["passed"], 1)

    def test_plugin_wrong_expected_fails(self):
        self.plugin()
        self.wrong_expected()
        report = self.replay(1, self.scalar)
        self.assertEqual(report["results"][0]["phase"], "execution_or_comparison")

    def test_plugin_unsupported_add_cannot_fall_back(self):
        self.plugin()
        report = self.replay(1, self.add)
        self.assertEqual(report["results"][0]["phase"], "session")
        self.assertIn("fallback", report["results"][0]["reason"].lower())

    def test_plugin_unavailable_device_is_error(self):
        self.plugin()
        self.profile["device_index"] = 100000
        self.assertIn("device is unavailable", self.replay(2)["error"])

    def test_cpp_export_plugin_replay(self):
        self.plugin()
        if SETTINGS.exported_cases is None:
            self.skipTest("Pass --exported-cases to validate the C++ producer")
        self.assertGreater(self.replay(cases=SETTINGS.exported_cases)["passed"], 0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner", required=True, type=Path)
    parser.add_argument("--runtime", required=True, type=Path)
    parser.add_argument("--plugin-library", type=Path)
    parser.add_argument("--exported-cases", type=Path)
    SETTINGS, remaining = parser.parse_known_args()
    SETTINGS.runner = SETTINGS.runner.resolve()
    SETTINGS.runtime = SETTINGS.runtime.resolve()
    if SETTINGS.plugin_library:
        SETTINGS.plugin_library = SETTINGS.plugin_library.resolve()
    if SETTINGS.exported_cases:
        SETTINGS.exported_cases = SETTINGS.exported_cases.resolve()
    unittest.main(argv=[__file__, *remaining])
