# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import unittest

from check_shader_config import check_source, schema_fields


class ShaderConfigPolicyTest(unittest.TestCase):
    def test_closed_declarations_and_host_factories_are_allowed(self):
        source = """
WEBGPU_DECLARE_SPECIALIZATION(Config, FIELDS);
static Config MakeConfig(bool value) { return Config{value}; }
WEBGPU_DECLARE_UNIFORMS(UNIFORMS);
"""
        self.assertEqual(check_source(source), [])

    def test_open_configuration_and_legacy_uniforms_are_rejected(self):
        for source in (
            "struct Config final { WEBGPU_CONFIG_MEMBERS(FIELDS); bool omitted; };",
            "WEBGPU_DECLARE_CONFIG(Config, FIELDS);",
            "WEBGPU_CONFIG_FIELD(bool, omitted);",
            "WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(uniform);",
        ):
            with self.subTest(source=source):
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

    def test_inventory_uses_declared_fields(self):
        source = "#define FIELDS(F) F(bool, fast) \\\n F(gsl::span<const int>, recipe)\n"
        self.assertEqual(
            schema_fields(source, "FIELDS"),
            [
                {"type": "bool", "name": "fast"},
                {"type": "gsl::span<const int>", "name": "recipe"},
            ],
        )


if __name__ == "__main__":
    unittest.main()
