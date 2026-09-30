# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Keep physical storage bindings private except for existing pointer kernels."""

import re
import unittest
from collections import Counter
from pathlib import Path


def _cpp_string_literals(text):
    # C++ storage_name/storage_offset identifiers are not shader binding names.
    tokens = re.compile(
        r"//[^\n]*|/\*.*?\*/|\'(?:\\.|[^\'\\])*\'"
        r'|R"(?P<delimiter>[^ ()\\\t\r\n]{0,16})\((?P<raw>.*?)\)(?P=delimiter)"'
        r'|"(?P<quoted>(?:\\.|[^"\\])*)"',
        re.DOTALL,
    )
    return "\n".join(
        match["raw"] if match["raw"] is not None else match["quoted"]
        for match in tokens.finditer(text)
        if match["raw"] is not None or match["quoted"] is not None
    )


# Existing atomic/subgroup-matrix paths retain direct storage access until their
# pointer-helper migration. Exact counts prevent additional uses being accepted.
_LEGACY_STORAGE_REFERENCES = {
    "onnxruntime/contrib_ops/webgpu/moe/gate.wgsl.template": {
        "storage_tokencount_for_expert": 1,
    },
    "onnxruntime/contrib_ops/webgpu/quantization/subgroup_matrix_matmul_nbits_16x16x16_128.wgsl.template": {
        "storage_bias": 16,
        "storage_input_a": 4,
        "storage_output": 16,
        "storage_weight_index_indirect": 3,
    },
    "onnxruntime/contrib_ops/webgpu/quantization/subgroup_matrix_matmul_nbits_8x16x16.wgsl.template": {
        "storage_bias": 1,
        "storage_input_a": 1,
        "storage_output": 8,
        "storage_tail_output": 4,
        "storage_weight_index_indirect": 3,
    },
    "onnxruntime/contrib_ops/webgpu/quantization/subgroup_matrix_matmul_nbits_8x8x8.wgsl.template": {
        "storage_bias": 8,
        "storage_weight_index_indirect": 3,
    },
    "onnxruntime/contrib_ops/webgpu/quantization/subgroup_matrix_matmul_nbits_prepack.wgsl.template": {
        "storage_input_a": 2,
        "storage_output_a": 2,
    },
    "onnxruntime/core/providers/webgpu/math/subgroup_matrix_gemm_8x16x16.wgsl.template": {
        "storage_input_a": 16,
        "storage_input_b": 8,
        "storage_input_c": 1,
    },
    "onnxruntime/core/providers/webgpu/math/subgroup_matrix_matmul.cc": {
        "storage_bias": 1,
    },
    "onnxruntime/core/providers/webgpu/math/subgroup_matrix_matmul_8x16x16.wgsl.template": {
        "storage_input_a": 8,
        "storage_input_b": 4,
    },
    "onnxruntime/core/providers/webgpu/tensor/scatter_elements.cc": {
        "storage_output": 1,
    },
    "onnxruntime/core/providers/webgpu/tensor/scatter_nd.cc": {
        "storage_output": 1,
    },
}

# These existing strings name an ONNX attribute and a local shader offset,
# not physical bindings. Keep their counts exact as well.
_NON_BINDING_STORAGE_REFERENCES = {
    "onnxruntime/core/providers/webgpu/nn/pool.cc": {"storage_order": 1},
    "onnxruntime/core/providers/webgpu/shader_variable.cc": {"storage_offset": 12},
}


class StorageAccessPolicyTest(unittest.TestCase):
    def test_cpp_scan_distinguishes_host_names_from_shader_strings(self):
        source = """
            auto storage_offset = 0;  // "storage_comment"
            /* R"(storage_comment)" */
            auto code = "storage_input[i]";
            auto raw = R"wgsl(storage_output[i] = storage_input[i];)wgsl";
            auto prefix = "storage_" + name;
        """
        self.assertEqual(
            _cpp_string_literals(source),
            "storage_input[i]\nstorage_output[i] = storage_input[i];\nstorage_",
        )

    def test_operators_do_not_bypass_storage_helpers(self):
        root = Path(__file__).resolve().parents[4]
        providers = (root / "onnxruntime/core/providers/webgpu", root / "onnxruntime/contrib_ops/webgpu")
        binding_owner = providers[0] / "shader_helper.cc"
        storage_name = re.compile(r"\bstorage_\w*")
        storage_declaration = re.compile(r"\bvar\s*<\s*storage\b")
        seen_legacy_files = set()
        checked = 0
        for provider in providers:
            self.assertTrue(provider.is_dir(), str(provider))
            for source in sorted(provider.rglob("*")):
                if not source.is_file() or not source.name.endswith(
                    (".cc", ".cpp", ".h", ".hpp", ".wgsl", ".wgsl.template")
                ):
                    continue
                if source == binding_owner or "node_modules" in source.parts:
                    continue
                checked += 1
                relative = source.relative_to(root).as_posix()
                if relative in _LEGACY_STORAGE_REFERENCES:
                    seen_legacy_files.add(relative)
                with self.subTest(source=relative):
                    text = source.read_text(encoding="utf-8")
                    if source.suffix in (".cc", ".cpp", ".h", ".hpp"):
                        text = _cpp_string_literals(text)
                    self.assertIsNone(
                        storage_declaration.search(text), "Only ShaderHelper may declare storage bindings"
                    )
                    self.assertEqual(
                        Counter(storage_name.findall(text)),
                        Counter(_LEGACY_STORAGE_REFERENCES.get(relative, {}))
                        + Counter(_NON_BINDING_STORAGE_REFERENCES.get(relative, {})),
                        "Physical binding names are private; use ShaderVariableHelper. "
                        "Existing pointer-kernel exceptions must not gain new direct accesses.",
                    )
        self.assertGreater(checked, 0)
        self.assertEqual(seen_legacy_files, set(_LEGACY_STORAGE_REFERENCES))


if __name__ == "__main__":
    unittest.main()
