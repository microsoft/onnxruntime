# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import tempfile
import unittest
from pathlib import Path

from _packaging_utils import gen_file_from_template, get_ort_version_substitutions


class OrtVersionPolicyTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.plugin_dir = Path(self.directory.name)
        self.write_policy()

    def write_policy(self, minimum="1.30.1\n", additional="1.28.3\n"):
        (self.plugin_dir / "MIN_ONNXRUNTIME_VERSION").write_text(minimum, encoding="utf-8")
        (self.plugin_dir / "ADDITIONAL_SUPPORTED_ONNXRUNTIME_VERSIONS").write_text(additional, encoding="utf-8")

    def test_exact_exception_and_minimum(self):
        self.assertEqual(
            get_ort_version_substitutions(self.plugin_dir),
            {
                "min_onnxruntime_version": "1.30.1",
                "supported_onnxruntime_versions": "`1.28.3`, or `1.30.1` or later",
            },
        )

    def test_empty_exception_list_preserves_minimum(self):
        self.write_policy(additional="")
        self.assertEqual(
            get_ort_version_substitutions(self.plugin_dir)["supported_onnxruntime_versions"],
            "`1.30.1` or later",
        )

    def test_multiple_exceptions(self):
        self.write_policy(additional="1.27.5\n1.28.3\n")
        self.assertEqual(
            get_ort_version_substitutions(self.plugin_dir)["supported_onnxruntime_versions"],
            "`1.27.5`, `1.28.3`, or `1.30.1` or later",
        )

    def test_whitespace_and_blank_lines(self):
        self.write_policy(minimum=" 1.30.1 \n", additional="\n 1.28.3 \n\t\n")
        self.assertEqual(
            get_ort_version_substitutions(self.plugin_dir)["supported_onnxruntime_versions"],
            "`1.28.3`, or `1.30.1` or later",
        )

    def test_invalid_minimum(self):
        for minimum in ("", "1.30", "1.30.1-dev", "1.30.1.0", "1.30.1\n1.31.0", "invalid", "\uff11.30.1"):
            with self.subTest(minimum=minimum):
                self.write_policy(minimum=minimum)
                with self.assertRaisesRegex(ValueError, "expected MAJOR.MINOR.PATCH"):
                    get_ort_version_substitutions(self.plugin_dir)

    def test_invalid_exception(self):
        for additional in ("1.28", "1.28.3-dev", "1.28.3.0", "1.28.3,1.29.0", "invalid", "1.\uff12\uff18.3"):
            with self.subTest(additional=additional):
                self.write_policy(additional=additional)
                with self.assertRaisesRegex(ValueError, "expected MAJOR.MINOR.PATCH"):
                    get_ort_version_substitutions(self.plugin_dir)

    def test_missing_policy_files(self):
        for name in ("MIN_ONNXRUNTIME_VERSION", "ADDITIONAL_SUPPORTED_ONNXRUNTIME_VERSIONS"):
            with self.subTest(name=name):
                self.write_policy()
                (self.plugin_dir / name).unlink()
                with self.assertRaises(FileNotFoundError):
                    get_ort_version_substitutions(self.plugin_dir)

    def test_repository_policy(self):
        substitutions = get_ort_version_substitutions(Path(__file__).resolve().parent)
        self.assertEqual(substitutions["min_onnxruntime_version"], "1.30.1")
        self.assertEqual(substitutions["supported_onnxruntime_versions"], "`1.28.3`, or `1.30.1` or later")

    def test_package_readme_templates(self):
        plugin_dir = Path(__file__).resolve().parent
        templates = (
            plugin_dir / "python" / "onnxruntime_ep_webgpu" / "README.md",
            plugin_dir / "csharp" / "Microsoft.ML.OnnxRuntime.EP.WebGpu" / "README.md",
        )
        for additional in ("", "1.28.3\n", "1.27.5\n1.28.3\n"):
            self.write_policy(additional=additional)
            substitutions = get_ort_version_substitutions(self.plugin_dir)
            for template in templates:
                with self.subTest(template=template, additional=additional):
                    output = self.plugin_dir / "README.md"
                    gen_file_from_template(template, output, substitutions)
                    text = output.read_text(encoding="utf-8")
                    self.assertIn(substitutions["supported_onnxruntime_versions"], text)
                    self.assertIn(substitutions["min_onnxruntime_version"], text)
                    self.assertNotIn("@min_onnxruntime_version@", text)
                    self.assertNotIn("@supported_onnxruntime_versions@", text)


if __name__ == "__main__":
    unittest.main()
