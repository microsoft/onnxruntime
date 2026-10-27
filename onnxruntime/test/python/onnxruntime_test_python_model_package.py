# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
from helper import get_name

import onnxruntime as ort
from onnxruntime.capi import _pybind_state as C

# mul_1.onnx computes Y = X * W with this constant initializer.
MUL_1_WEIGHTS = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)


@unittest.skipUnless(hasattr(C, "ModelPackage"), "Model package authoring requires a non-minimal build")
class TestModelPackageAuthoring(unittest.TestCase):
    def test_author_package_with_shared_asset_and_variant(self):
        package_name = "my_model"
        variant_name = f"{package_name}.cpu"
        with tempfile.TemporaryDirectory() as temp:
            output_dir = Path(temp) / "package"
            og_staging = Path(temp) / "og_staging"
            output_dir.mkdir()
            og_staging.mkdir()
            (output_dir / "manifest.json").write_text(
                json.dumps({"schema_version": "1.0", "components": {}}), encoding="utf-8"
            )
            shutil.copyfile(get_name("mul_1.onnx"), og_staging / "mul_1.onnx")
            variant_dir = output_dir / package_name / variant_name
            variant_dir.mkdir(parents=True)
            shutil.copyfile(get_name("mul_1.onnx"), variant_dir / "model.onnx")

            package = ort.ModelPackage.open(
                str(output_dir), allow_external_paths=False, follow_symlinks=True, strict_unknown_fields=False
            )
            package.set_metadata(package_name, "1.0.0")
            shared_asset_uri = package.add_shared_asset(str(og_staging), copy_in=True)
            package.set_component_inline(package_name, json.dumps({"variants": {}}))
            package.set_variant(
                package_name,
                variant_name,
                json.dumps(
                    {
                        "variant_directory": f"{package_name}/{variant_name}",
                        "ep": "CPUExecutionProvider",
                        "device": "cpu",
                    }
                ),
            )
            ort_info = {
                "model_file": "model.onnx",
                "session_options": {
                    "session.model_external_initializers_file_folder_path": shared_asset_uri,
                },
            }
            package.set_variant_executor_info_inline(package_name, variant_name, "ort", json.dumps(ort_info))
            package.commit(mode=ort.ModelPackageWriteMode.PRESERVE)
            package.close()
            shutil.rmtree(og_staging)

            with ort.ModelPackage.open(str(output_dir)) as package:
                info = package.get_info()
                self.assertEqual(info["package_name"], package_name)
                self.assertEqual(info["package_version"], "1.0.0")
                self.assertEqual([a["uri"] for a in info["shared_assets"]], [shared_asset_uri])
                (variant,) = info["components"][0]["variants"]
                self.assertEqual(variant["ep"], "CPUExecutionProvider")
                self.assertEqual(variant["device"], "cpu")
                body = json.loads(package.get_variant_json(package_name, variant_name))
                self.assertEqual(body["executor_info"]["ort"], ort_info)
                asset_path = Path(package.resolve_asset_uri(shared_asset_uri))
                self.assertTrue((asset_path / "mul_1.onnx").is_file())

            context = ort.ModelPackageContext(str(output_dir))
            selected = context.select_component(package_name, ort.ModelPackageOptions(ort.SessionOptions()))
            self.assertEqual(selected.get_selected_variant_name(), variant_name)
            session = selected.create_session()
            x = np.arange(6, dtype=np.float32).reshape(3, 2)
            (output,) = session.run(["Y"], {"X": x}, None)
            np.testing.assert_array_equal(output, x * MUL_1_WEIGHTS)

    def test_invalid_inputs_report_native_errors(self):
        with ort.ModelPackage() as package:
            with self.assertRaises(RuntimeError):
                package.commit()
            with self.assertRaises(RuntimeError):
                package.set_component_inline("model", "not json")
            with self.assertRaises(RuntimeError):
                package.set_layout("invalid")
            with self.assertRaises(RuntimeError):
                ort.ModelPackage.open("missing-model-package-directory")


@unittest.skipUnless(hasattr(C, "ModelPackageContext"), "Model package API requires a non-minimal build")
class TestStableModelPackageApi(unittest.TestCase):
    def test_inspect_select_and_create_session(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "package"
            variant = root / "model" / "cpu"
            variant.mkdir(parents=True)
            shutil.copyfile(get_name("mul_1.onnx"), variant / "model.onnx")
            manifest = {
                "schema_version": "1.0",
                "components": {
                    "model": {
                        "variants": {
                            "cpu": {
                                "variant_directory": "model/cpu",
                                "ep": "CPUExecutionProvider",
                                "device": "cpu",
                                "executor_info": {"ort": {"model_file": "model.onnx"}},
                            }
                        }
                    }
                },
            }
            (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            package = ort.ModelPackageContext(str(root))
            self.assertEqual(package.get_schema_version(), 1)
            self.assertEqual(package.get_component_count(), 1)
            self.assertEqual(package.get_component_names(), ["model"])
            self.assertEqual(package.get_variant_count("model"), 1)
            self.assertEqual(package.get_variant_names("model"), ["cpu"])
            self.assertEqual(package.get_variant_ep_name("model", "cpu"), "CPUExecutionProvider")
            resolved = package.resolve_string_ref("model/cpu/model.onnx", must_exist=True)
            self.assertEqual(Path(resolved).resolve(), (variant / "model.onnx").resolve())
            with self.assertRaises(RuntimeError):
                package.get_variant_count("missing")

            options = ort.ModelPackageOptions(ort.SessionOptions())
            component = package.select_component("model", options)
            del package, options
            self.assertEqual(component.get_selected_variant_name(), "cpu")
            self.assertEqual(Path(component.get_selected_variant_folder_path()).resolve(), variant.resolve())
            sessions = [component.create_session(), component.create_session(ort.SessionOptions())]
            del component
            x = np.arange(6, dtype=np.float32).reshape(3, 2)
            for session in sessions:
                self.assertIsInstance(session, C.InferenceSession)
                (output,) = session.run(["Y"], {"X": x}, None)
                np.testing.assert_array_equal(output, x * MUL_1_WEIGHTS)


if __name__ == "__main__":
    unittest.main()
