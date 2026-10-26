import json
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import onnx
import package_cases
from create_cases_from_json import generate
from google.protobuf.message import DecodeError
from onnx import numpy_helper

ROOT = Path(__file__).resolve().parents[1]


class PackageCasesTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.cases = self.root / "cases"
        generate(ROOT / "cases/pilot.json", self.cases)
        self.archive = self.root / "cases.zip"
        self.manifest = next(self.cases.glob("*/case.json"))
        self.document = json.loads(self.manifest.read_text())

    def create(self, output=None):
        package_cases.create(
            self.cases,
            output or self.archive,
            "preview-1",
            "test-revision",
            ROOT.parents[2] / "LICENSE",
            ROOT.parents[2] / "ThirdPartyNotices.txt",
        )

    def mutate(self, **changes):
        self.manifest.write_text(json.dumps({**self.document, **changes}))

    def test_roundtrip_preserves_bytes_and_records_requirements(self):
        self.create()
        metadata = package_cases.verify_cases_archive(self.archive, "test-revision")
        self.assertEqual(len(metadata["cases"]), 3)
        self.assertEqual(metadata["cases"][0]["opsets"], {"": 14})
        with zipfile.ZipFile(self.archive) as archive:
            for path in self.cases.rglob("*"):
                if path.is_file():
                    self.assertEqual(
                        archive.read("cases/" + path.relative_to(self.cases).as_posix()), path.read_bytes()
                    )

    def test_newer_model_versions_are_recorded_not_rejected_by_installed_schemas(self):
        path = self.manifest.parent / self.document["model"]
        model = onnx.load_model(path)
        model.ir_version = onnx.IR_VERSION + 1
        model.opset_import[0].version = onnx.defs.onnx_opset_version() + 1
        path.write_bytes(model.SerializeToString())
        self.create()
        metadata = package_cases.verify_cases_archive(self.archive)
        description = next(case for case in metadata["cases"] if case["id"] == self.document["id"])
        self.assertEqual(description["ir_version"], model.ir_version)
        self.assertEqual(description["opsets"], {"": model.opset_import[0].version})

    def test_archive_is_reproducible_and_cannot_overwrite(self):
        self.create()
        second = self.root / "second.zip"
        self.create(second)
        self.assertEqual(self.archive.read_bytes(), second.read_bytes())
        with self.assertRaises(FileExistsError):
            self.create()

    def test_invalid_case_field_rejected(self):
        self.mutate(ignored_assertion=True)
        with self.assertRaisesRegex(ValueError, "fields"):
            self.create()

    def test_traversal_and_missing_dependency_rejected(self):
        for model in ("../outside.onnx", "missing.onnx", "C:/outside.onnx", "nested\\model.onnx"):
            with self.subTest(model=model):
                self.mutate(model=model)
                with self.assertRaises(ValueError):
                    self.create()
        self.assertFalse(self.archive.exists())

    def test_duplicate_case_id_rejected(self):
        other = list(self.cases.glob("*/case.json"))[1]
        document = json.loads(other.read_text())
        document["id"] = self.document["id"].upper()
        other.write_text(json.dumps(document))
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.create()

    def test_unreferenced_file_rejected(self):
        (self.cases / "extra.txt").write_text("extra")
        with self.assertRaisesRegex(ValueError, "closure"):
            self.create()

    def test_malformed_tensor_rejected(self):
        (self.manifest.parent / self.document["datasets"][0]["outputs"][0]).write_bytes(b"\x80")
        with self.assertRaises(DecodeError):
            self.create()

    def test_mixed_tensor_storage_rejected(self):
        path = self.manifest.parent / self.document["datasets"][0]["outputs"][0]
        original = path.read_bytes()
        for field in ("float_data", "int32_data", "int64_data", "uint64_data", "double_data", "string_data"):
            tensor = onnx.TensorProto.FromString(original)
            getattr(tensor, field).append(b"wrong" if field == "string_data" else 1)
            path.write_bytes(tensor.SerializeToString())
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.create()

    def test_output_must_not_mutate_case_directory(self):
        with self.assertRaisesRegex(ValueError, "outside"):
            self.create(self.cases / "cases.zip")
        self.assertFalse((self.cases / "cases.zip").exists())

    def test_case_root_manifest_is_not_silently_ignored(self):
        (self.cases / "artifact.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "artifact.json"):
            self.create()

    def test_changed_input_during_packaging_rejected(self):
        original = package_cases.shutil.copytree

        def copytree(source, destination, *args, **kwargs):
            self.mutate(source={"definition": "changed.json"})
            return original(source, destination, *args, **kwargs)

        with (
            patch("package_cases.shutil.copytree", side_effect=copytree),
            self.assertRaisesRegex(ValueError, "changed"),
        ):
            self.create()
        self.assertFalse(self.archive.exists())

    def test_wrong_values_are_not_a_correctness_check(self):
        path = self.manifest.parent / self.document["datasets"][0]["outputs"][0]
        tensor = onnx.load_tensor(path)
        path.write_bytes(
            numpy_helper.from_array(numpy_helper.to_array(tensor) + np.float32(1), tensor.name).SerializeToString()
        )
        self.create()
        package_cases.verify_cases_archive(self.archive)

    def test_archive_tampering_and_added_files_rejected(self):
        self.create()
        with zipfile.ZipFile(self.archive, "a") as archive:
            archive.writestr("extra.txt", "unexpected")
        with self.assertRaisesRegex(ValueError, "closure"):
            package_cases.verify_cases_archive(self.archive)

    def test_archive_changed_content_rejected(self):
        self.create()
        changed = self.root / "changed.zip"
        with zipfile.ZipFile(self.archive) as source, zipfile.ZipFile(changed, "w") as target:
            for entry in source.infolist():
                target.writestr(entry, b"changed" if entry.filename == "LICENSE" else source.read(entry))
        with self.assertRaisesRegex(ValueError, "hashes"):
            package_cases.verify_cases_archive(changed)

    def test_case_insensitive_paths_rejected(self):
        for relative in ("CON.txt", "directory/file.", "directory/file ", "directory/file:stream", "directory/file?"):
            with self.subTest(relative=relative), self.assertRaises(ValueError):
                package_cases.safe_relative_path(relative)

    def test_generic_archive_handles_deliberately_invalid_fixtures(self):
        self.mutate(ignored_assertion=True)
        package_cases.write_archive(self.cases, self.archive, "ci-fixtures", "test-revision")
        package_cases.verify_archive(self.archive, "ci-fixtures")
        with self.assertRaisesRegex(ValueError, "kind"):
            package_cases.verify_cases_archive(self.archive)

    def test_unsafe_archive_entry_rejected_before_extraction(self):
        self.create()
        with zipfile.ZipFile(self.archive, "a") as archive:
            archive.writestr("../outside", "unsafe")
        with self.assertRaisesRegex(ValueError, "Unsafe"):
            package_cases.extract_archive(self.archive, self.root / "extract", "conformance-cases")
        self.assertFalse((self.root / "extract").exists())

    def test_cli_create_and_verify(self):
        script = Path(package_cases.__file__)
        commands = [
            [
                "create",
                "--cases",
                str(self.cases),
                "--output",
                str(self.archive),
                "--version",
                "preview-1",
                "--revision",
                "test-revision",
                "--license",
                str(ROOT.parents[2] / "LICENSE"),
                "--notices",
                str(ROOT.parents[2] / "ThirdPartyNotices.txt"),
            ],
            ["verify", "--archive", str(self.archive), "--revision", "test-revision"],
        ]
        for arguments in commands:
            result = subprocess.run(
                [sys.executable, "-B", str(script), *arguments], capture_output=True, text=True, check=False
            )
            self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
