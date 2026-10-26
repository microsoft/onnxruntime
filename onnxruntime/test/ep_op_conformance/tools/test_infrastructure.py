import json
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import onnx
from ci.package_artifacts import ROOT, check_generated, create, seal, verify, verify_all
from create_cases_from_json import (
    case_directory_name,
    generate,
    identity,
    materialize,
    validate_fields,
    verify_generated,
)


class DefinitionFieldsTest(unittest.TestCase):
    def test_required_and_optional_fields_accepted(self):
        validate_fields({"name": "X"}, ("name",), ("initializer",))
        validate_fields({"name": "X", "initializer": True}, ("name",), ("initializer",))

    def test_missing_fields_listed_in_sorted_order(self):
        with self.assertRaises(ValueError) as error:
            validate_fields({}, ("zebra", "alpha"))
        self.assertEqual(str(error.exception), "Invalid definition fields: missing: alpha, zebra")

    def test_unsupported_fields_listed_in_sorted_order(self):
        with self.assertRaises(ValueError) as error:
            validate_fields({"name": "X", "zebra": 0, "alpha": 0}, ("name",))
        self.assertEqual(str(error.exception), "Invalid definition fields: unsupported: alpha, zebra")

    def test_missing_and_unsupported_fields_both_listed(self):
        with self.assertRaises(ValueError) as error:
            validate_fields({"extra": 0}, ("name",), ("initializer",))
        self.assertEqual(str(error.exception), "Invalid definition fields: missing: name; unsupported: extra")

    def test_nonobject_reports_actual_type(self):
        for value in (None, [], "text", 1):
            with self.subTest(value=value), self.assertRaises(ValueError) as error:
                validate_fields(value, ("name",))
            self.assertEqual(str(error.exception), f"Expected a definition object, got {type(value).__name__}")


class CaseNamingTest(unittest.TestCase):
    def test_prefixed_portable_names(self):
        for case_id in (
            "ai.onnx.Mul.opset14.float32_scalar",
            "CON.example.Mul.opset14.Mixed_types2",
            "nul.Mul.opset14.basic",
            "COM1.Mul.opset14.basic",
            "LPT9.Mul.opset14.basic",
            "ai.onnx.Mul.opset14." + "a" * 128,
        ):
            with self.subTest(identity=case_id):
                self.assertEqual(case_directory_name(case_id), f"case-{case_id}")

    def test_invalid_components_rejected(self):
        for case_id in (
            None,
            "",
            ".",
            "..",
            "ai..Mul.opset14.basic",
            "ai.onnx.Mul.opset0.basic",
            "ai.onnx.Mul.opset14.float32.scalar",
            "ai.onnx.Mul.opset14.float32-scalar",
            "ai.onnx.Mul.opset14.",
            "ai.onnx.Mul.opset14.a/b",
            "ai.onnx.Mul.opset14.a\\b",
            "ai.onnx.Mul.opset14.a:b",
            "ai.onnx.Mul.opset14.a b",
            "ai.onnx.Mul.opset14.\u00e9",
        ):
            with self.subTest(identity=case_id), self.assertRaisesRegex(ValueError, "Invalid case ID"):
                case_directory_name(case_id)


class GenerationTest(unittest.TestCase):
    def test_direct_and_build_cache_cli(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            script = ROOT / "tools/create_cases_from_json.py"
            arguments = [sys.executable, "-B", str(script), "--definitions", str(ROOT / "cases/pilot.json")]
            for destination in (("--output", root / "direct"), ("--build-cache", root / "cache")):
                subprocess.run([*arguments, destination[0], str(destination[1])], check=True, capture_output=True)
            metadata = json.loads((root / "cache/current.json").read_text())
            cached = root / "cache" / metadata["directory"]
            expected = {
                path.relative_to(root / "direct"): path.read_bytes()
                for path in (root / "direct").rglob("*")
                if path.is_file()
            }
            self.assertEqual(
                expected, {path.relative_to(cached): path.read_bytes() for path in cached.rglob("*") if path.is_file()}
            )
            key = subprocess.run([*arguments, "--print-key"], check=True, capture_output=True, text=True).stdout.strip()
            self.assertEqual(key, metadata["directory"])
            subprocess.run(
                [*arguments, "--build-cache", str(root / "cache"), "--verify"], check=True, capture_output=True
            )
            for invalid in (("--verify",), ("--output", str(root / "new"), "--build-cache", str(root / "cache"))):
                result = subprocess.run([*arguments, *invalid], check=False, capture_output=True)
                self.assertEqual(result.returncode, 2)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.definitions = self.root / "pilot.json"
        self.definitions.write_bytes((ROOT / "cases/pilot.json").read_bytes())
        self.destination = self.root / "output" / "cases"

    def test_cases_written_before_next_case_is_built(self):
        document = json.loads(self.definitions.read_text())
        original_check = onnx.checker.check_model

        def check_model(model):
            self.assertFalse(self.destination.exists())
            if checker.call_count == 2:
                manifests = list(self.destination.parent.rglob("case.json"))
                self.assertEqual(len(manifests), 1)
                self.assertEqual(json.loads(manifests[0].read_text())["id"], document["cases"][0]["id"])
                self.assertTrue((manifests[0].parent / "model.onnx").is_file())
                self.assertTrue((manifests[0].parent / "test_data_set_0/output_0.pb").is_file())
            original_check(model)

        with patch("create_cases_from_json.onnx.checker.check_model", side_effect=check_model) as checker:
            generate(self.definitions, self.destination)
        self.assertEqual(checker.call_count, len(document["cases"]))
        self.assertEqual(len(list(self.destination.glob("*/case.json"))), len(document["cases"]))
        for path in self.destination.glob("*/case.json"):
            manifest = json.loads(path.read_text())
            self.assertEqual(manifest["source"], {"definition": self.definitions.name})
            self.assertEqual(path.parent.name, f"case-{manifest['id']}")
        self.assertEqual(list(self.destination.parent.iterdir()), [self.destination])

    def test_case_insensitive_collision_rejected(self):
        document = json.loads(self.definitions.read_text())
        document["cases"][1]["id"] = document["cases"][0]["id"].replace("Mul", "mul")
        self.definitions.write_text(json.dumps(document))
        with self.assertRaisesRegex(ValueError, "Duplicate case ID"):
            generate(self.definitions, self.destination)
        self.assertFalse(self.destination.exists())
        self.assertEqual(list(self.destination.parent.iterdir()), [])

    def test_late_validation_failure_cleans_staging(self):
        document = json.loads(self.definitions.read_text())
        document["cases"][-1]["comparison"]["atol"] = -1
        self.definitions.write_text(json.dumps(document))
        with self.assertRaisesRegex(ValueError, "Invalid comparison tolerance"):
            generate(self.definitions, self.destination)
        self.assertFalse(self.destination.exists())
        self.assertEqual(list(self.destination.parent.iterdir()), [])

    def test_write_failure_cleans_staging(self):
        original_write = Path.write_bytes

        def write_bytes(path, contents):
            if path.name == "case.json":
                self.assertTrue((path.parent / "model.onnx").is_file())
                raise OSError("simulated write failure")
            return original_write(path, contents)

        with (
            patch("create_cases_from_json.Path.write_bytes", new=write_bytes),
            self.assertRaisesRegex(OSError, "simulated write failure"),
        ):
            generate(self.definitions, self.destination)
        self.assertFalse(self.destination.exists())
        self.assertEqual(list(self.destination.parent.iterdir()), [])

    def test_publish_failure_cleans_staging(self):
        with (
            patch("create_cases_from_json.Path.rename", side_effect=OSError("simulated rename failure")),
            self.assertRaisesRegex(OSError, "simulated rename failure"),
        ):
            generate(self.definitions, self.destination)
        self.assertFalse(self.destination.exists())
        self.assertEqual(list(self.destination.parent.iterdir()), [])

    def test_nonempty_destination_created_before_rename_preserved(self):
        original_rename = Path.rename
        sentinel = self.destination / "keep.txt"

        def rename(path, target):
            self.destination.mkdir()
            sentinel.write_text("competing output")
            return original_rename(path, target)

        with patch("create_cases_from_json.Path.rename", new=rename), self.assertRaises(OSError):
            generate(self.definitions, self.destination)
        self.assertEqual(sentinel.read_text(), "competing output")
        self.assertEqual(list(self.destination.iterdir()), [sentinel])
        self.assertEqual(list(self.destination.parent.iterdir()), [self.destination])

    def test_existing_destination_preserved(self):
        self.destination.mkdir(parents=True)
        sentinel = self.destination / "keep.txt"
        sentinel.write_text("existing output")
        with (
            patch("create_cases_from_json.Path.read_text") as read_text,
            self.assertRaisesRegex(ValueError, "already exists"),
        ):
            generate(self.definitions, self.destination)
        read_text.assert_not_called()
        self.assertEqual(sentinel.read_text(), "existing output")
        self.assertEqual(list(self.destination.iterdir()), [sentinel])


class InfrastructureTest(unittest.TestCase):
    def setUp(self):
        revision = patch.dict("os.environ", {"GITHUB_SHA": "test-revision"})
        revision.start()
        self.addCleanup(revision.stop)
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.definitions = self.root / "pilot.json"
        self.definitions.write_bytes((ROOT / "cases/pilot.json").read_bytes())
        self.generated = self.root / "generated"
        materialize(self.definitions, self.generated)

    def package(self):
        runtime = self.root / "onnxruntime.dll"
        plugin = self.root / "example_plugin_ep.dll"
        runtime.write_bytes(b"runtime fixture")
        plugin.write_bytes(b"plugin fixture")
        runner = self.root / "runner.exe"
        runner.write_bytes(b"runner fixture")
        output = self.root / "package"
        create(self.generated, runtime, plugin, output, "test-revision", runner)
        return output

    def test_unchanged_generation_reuses_bytes(self):
        positive, metadata = check_generated(self.generated)
        times = {path: path.stat().st_mtime_ns for path in positive.rglob("*") if path.is_file()}
        materialize(self.definitions, self.generated)
        self.assertEqual(metadata, check_generated(self.generated)[1])
        self.assertTrue(all(path.stat().st_mtime_ns == stamp for path, stamp in times.items()))

    def test_generation_records_source_revision(self):
        self.assertEqual(check_generated(self.generated)[1]["source_revision"], "test-revision")

    def test_changed_input_keeps_old_cases_immutable(self):
        positive, original = check_generated(self.generated)
        document = json.loads(self.definitions.read_text())
        document["cases"][0]["comparison"]["atol"] = 0.25
        self.definitions.write_text(json.dumps(document))
        materialize(self.definitions, self.generated)
        self.assertNotEqual(positive, check_generated(self.generated)[0])
        self.assertTrue(positive.is_dir())
        self.assertEqual(original["files"], json.loads((self.generated / f"{positive.name}.json").read_text())["files"])

    def test_mutated_generated_bytes_rejected(self):
        positive, _ = check_generated(self.generated)
        next(positive.rglob("model.onnx")).write_bytes(b"corrupt")
        with self.assertRaisesRegex(ValueError, "immutable"):
            materialize(self.definitions, self.generated)

    def test_missing_generated_file_rejected(self):
        positive, _ = check_generated(self.generated)
        next(positive.rglob("model.onnx")).unlink()
        with self.assertRaisesRegex(ValueError, "immutable"):
            verify_generated(self.definitions, self.generated)

    def test_producer_versions_recorded_without_enforcement(self):
        packages = {"numpy": "test-numpy", "onnx": "test-onnx", "protobuf": "test-protobuf"}
        with patch("create_cases_from_json.importlib.metadata.version", side_effect=packages.__getitem__):
            self.assertEqual(identity(self.definitions)["producer_packages"], packages)

    def test_changed_producer_versions_keep_old_cases_immutable(self):
        positive, original = check_generated(self.generated)
        with patch("create_cases_from_json.importlib.metadata.version", return_value="changed-version"):
            with self.assertRaisesRegex(ValueError, "input identity mismatch"):
                verify_generated(self.definitions, self.generated)
            materialize(self.definitions, self.generated)
            verify_generated(self.definitions, self.generated)
        updated, metadata = check_generated(self.generated)
        self.assertNotEqual(positive, updated)
        self.assertEqual(metadata["producer_packages"], dict.fromkeys(("numpy", "onnx", "protobuf"), "changed-version"))
        self.assertTrue(positive.is_dir())
        self.assertEqual(original, json.loads((self.generated / f"{positive.name}.json").read_text()))

    def test_package_roundtrip_and_header_closure(self):
        output = self.package()
        self.assertEqual(verify_all(output / "source-kit", output / "cases", output / "fixture"), "test-revision")
        self.assertTrue((output / "source-kit/include/onnxruntime_ep_c_api.h").is_file())
        self.assertFalse(list((output / "source-kit").rglob("*.dll")))
        self.assertFalse(list((output / "source-kit").rglob("*.exe")))
        with zipfile.ZipFile(output / "cases/cases.zip") as canonical:
            identities = {entry["id"] for entry in json.loads(canonical.read("artifact.json"))["cases"]}
            self.assertFalse(any("wrong-output" in name for name in canonical.namelist()))
        with zipfile.ZipFile(output / "cases/negative-fixtures.zip") as negative:
            self.assertIn("cases/wrong-output/case.json", negative.namelist())
            profile = json.loads(negative.read("profiles/plugin.json"))
        self.assertTrue(set(profile["skips"]).issubset(identities))
        self.assertTrue(all(not identity.startswith("case-") for identity in profile["skips"]))
        self.assertEqual(
            check_generated(self.generated)[1]["files"],
            json.loads((output / "cases/generation.json").read_text())["files"],
        )

    def test_package_cannot_overwrite(self):
        output = self.package()
        with self.assertRaisesRegex(ValueError, "immutable"):
            create(
                self.generated,
                self.root / "onnxruntime.dll",
                self.root / "example_plugin_ep.dll",
                output,
                "test-revision",
                self.root / "runner.exe",
            )

    def test_package_tamper_rejected(self):
        output = self.package()
        (output / "fixture/onnxruntime.dll").write_bytes(b"corrupt")
        with self.assertRaisesRegex(ValueError, "hashes"):
            verify(output / "fixture", "fixture")

    def test_package_missing_dependency_rejected(self):
        output = self.package()
        (output / "fixture/example_plugin_ep.dll").unlink()
        with self.assertRaisesRegex(ValueError, "closure"):
            verify(output / "fixture", "fixture")

    def test_package_revision_mismatch_rejected(self):
        output = self.package()
        with self.assertRaisesRegex(ValueError, "revision"):
            verify_all(output / "source-kit", output / "cases", output / "fixture", "other-revision")

    def test_package_unsafe_path_rejected(self):
        output = self.package()
        path = output / "fixture/artifact.json"
        metadata = json.loads(path.read_text())
        metadata["files"]["../outside"] = "bad"
        path.write_text(json.dumps(metadata))
        with self.assertRaisesRegex(ValueError, "Unsafe"):
            verify(output / "fixture", "fixture")

    def test_package_compatibility_mismatch_rejected(self):
        output = self.package()
        fixture = output / "fixture"
        seal(fixture, "fixture", "test-revision", ort_version="bad", minimum_runtime_api=24, source_header_api=32)
        with self.assertRaisesRegex(ValueError, "compatibility"):
            verify_all(output / "source-kit", output / "cases", fixture)


if __name__ == "__main__":
    unittest.main()
