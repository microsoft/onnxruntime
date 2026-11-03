from pathlib import Path
import unittest

from build_dense_packed4_bindings import transform
from run_dense_packed4_bindings import verify_bindings


class DensePacked4BindingTests(unittest.TestCase):
    def test_three_live_gpu_banks_are_copied_and_used_in_both_modes(self):
        source = Path("/home/sayanshaw/repos/qwen38-int4-dense-packed4/build-v4/native_compare.cpp")
        modified = transform(source.read_text())
        self.assertIn("std::vector<std::vector<Ort::Value>> gpu_inputs(3)", modified)
        self.assertIn("GetTensorRawData() == gpu_inputs[other][0].GetTensorRawData()", modified)
        self.assertIn("env.CopyTensors(inputs, bank, nullptr)", modified)
        self.assertEqual(modified.count("names.data(), bank.data(), bank.size(),"), 2)
        self.assertEqual(modified.count("names.data(), inputs.data(), inputs.size(),"), 1)

    def test_real_rebinds_and_binding_hits_required_for_every_shape(self):
        def row(operator, requests):
            return dict(operator=operator, requests=requests, entries=1, captures=1,
                        hits=requests - 1, rebinds=2, binding_hits=requests - 3,
                        capacity=64, binding_entries=3, binding_capacity=64)
        rows = [row("MatMulNBitsExact", 128) for _ in range(8)] + [row("MatMulNBitsExact", 256)]
        rows += [row("DensePacked4", 128) for _ in range(8)]
        verify_bindings(rows)
        for field in ("rebinds", "binding_entries", "binding_hits"):
            value = rows[-1][field]
            rows[-1][field] = 0
            with self.assertRaises(AssertionError):
                verify_bindings(rows)
            rows[-1][field] = value


if __name__ == "__main__":
    unittest.main()
