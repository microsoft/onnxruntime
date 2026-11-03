import unittest

from build_adapter_inline_capacity17 import transform
from prepare_adapter_inline_capacity17_proofs import capacity17
from prepare_adapter_inline_proofs import ROOT, model_source, native_source
import test_adapter_inline_tensors as original_tests
from test_adapter_tensor_reuse import HOST_HEADER


class AdapterInlineCapacity17Tests(original_tests.AdapterInlineTensorTests):
    header_transform = staticmethod(transform)
    input_capacity = 17

    def test_actual_header_matches_new_capacity(self):
        source = transform(HOST_HEADER.read_text())
        self.assertIn("InlinedVector<Tensor, 17>", source)
        self.assertIn('input_capacity\\":17', source)
        self.assertIn("inputs > 17", source)
        self.assertNotIn("inputs > 8", source)

    def test_native_and_model_counters_require_capacity17(self):
        for name, transform_source in (
            ("run_adapter_tensor_reuse_native.py", native_source),
            ("run_adapter_tensor_reuse_model.py", model_source),
        ):
            source = capacity17(transform_source((ROOT / name).read_text()))
            self.assertIn('row["input_capacity"] == 17', source)
            self.assertNotIn('row["input_capacity"] == 8', source)
            self.assertIn("qwen38-int4-adapter-inline-tensors/build-v3", source)
            self.assertNotIn("adapter-inline-tensors-v2", source)
            if name.endswith("_model.py"):
                self.assertIn('assert vector_coverage["input_heap_fallbacks"] == 0', source)
                self.assertIn('assert vector_coverage["output_heap_fallbacks"] == 0', source)


if __name__ == "__main__":
    unittest.main()
