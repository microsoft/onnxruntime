# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import unittest

from check_shader_config import check_source


class ShaderConfigPolicyTest(unittest.TestCase):
    def test_declared_fields_and_constructor_locals_are_allowed(self):
        source = """
struct Config final {
  WEBGPU_CONFIG_MEMBERS(FIELDS);
  Config(bool value) : flag_{value} {
    int local = 0;
  }
};
"""
        self.assertEqual(check_source(source), [])

    def test_unencoded_members_are_rejected(self):
        for field in ("bool omitted;", "int omitted = 2;", "std::vector<int> omitted;", "bool omitted{};"):
            with self.subTest(field=field):
                source = f"struct Config final {{\nWEBGPU_CONFIG_MEMBERS(FIELDS);\n{field}\n}};"
                self.assertTrue(check_source(source))

    def test_legacy_and_handwritten_encoders_are_rejected(self):
        for source in (
            "class Old final : public Program<Old> {};",
            "program.CacheHint(value);",
            "void AppendTo(std::string&) const {}",
            "using ShaderConfigSchema = void;",
            "device.CreateShaderModule(&descriptor);",
            "device.CreateComputePipelineAsync(&descriptor);",
        ):
            with self.subTest(source=source):
                self.assertTrue(check_source(source))

    def test_comments_and_shader_literals_are_ignored(self):
        self.assertEqual(
            check_source('/* program.CacheHint(x); */ const char* s = R"(class Old : public Program<Old> {})";'), []
        )


if __name__ == "__main__":
    unittest.main()
