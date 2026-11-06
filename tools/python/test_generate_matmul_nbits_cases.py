# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper

from tools.python.generate_matmul_nbits_cases import (
    Case,
    default_cases,
    generate_cases,
    load_cases,
    make_case,
    pack_codes,
)


class TestGenerateMatMulNBitsCases(unittest.TestCase):
    def test_pack_codes_low_bits_first(self):
        for bits, codes, expected in (
            (2, [0, 1, 2, 3, 1], [228, 1]),
            (4, [1, 2, 3], [33, 3]),
            (8, [0, 127, 255], [0, 127, 255]),
        ):
            with self.subTest(bits=bits):
                np.testing.assert_array_equal(pack_codes(np.array([codes], dtype=np.uint8), bits), [expected])
        with self.assertRaises(ValueError):
            pack_codes(np.array([16], dtype=np.uint8), 4)

    def test_model_shapes_slots_and_padding(self):
        for bits in (2, 4, 8):
            for dtype in ("float32", "float16"):
                for has_bias in (False, True):
                    for has_zeros in (False, True):
                        case = Case((2, 3, 65), 5, bits=bits, dtype=dtype, bias=has_bias, zero_points=has_zeros)
                        with self.subTest(case=case):
                            model, activation = make_case(case, 123)
                            onnx.checker.check_model(model)
                            self.assertEqual(activation.shape, (2, 3, 65))
                            self.assertEqual(activation.dtype, np.dtype(dtype))
                            node = model.graph.node[0]
                            self.assertEqual(len(model.graph.node), 1)
                            self.assertEqual(node.op_type, "MatMulNBits")
                            self.assertEqual(node.domain, "com.microsoft")
                            arrays = {tensor.name: numpy_helper.to_array(tensor) for tensor in model.graph.initializer}
                            self.assertEqual(arrays["B"].shape, (5, 3, 32 * bits // 8))
                            self.assertEqual(arrays["scales"].shape, (5, 3))
                            self.assertTrue(np.all(arrays["scales"] > 0))
                            if has_bias:
                                self.assertEqual(node.input[5], "bias")
                                self.assertEqual(node.input[4], "")
                                self.assertEqual(arrays["bias"].shape, (5,))
                            if has_zeros:
                                self.assertEqual(node.input[3], "zero_points")
                                self.assertEqual(arrays["zero_points"].shape, (5, (3 * bits + 7) // 8))
                            for column in range(5):
                                zero = (
                                    (int(arrays["zero_points"][column, 2 * bits // 8]) >> (2 * bits % 8))
                                    & ((1 << bits) - 1)
                                    if has_zeros
                                    else 1 << (bits - 1)
                                )
                                for position in range(1, 32):
                                    byte = int(arrays["B"][column, 2, position * bits // 8])
                                    code = (byte >> (position * bits % 8)) & ((1 << bits) - 1)
                                    self.assertEqual(code, zero)

    def test_logical_counts_and_reproducibility(self):
        cases = [Case((64,), 32), Case((2, 3, 64), 32, bias=True)]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = generate_cases(cases, root / "first", 123)
            second = generate_cases(list(reversed(cases)), root / "second", 123)
            by_id = {case["case_id"]: case for case in second["cases"]}
            for record in first["cases"]:
                self.assertEqual(record, by_id[record["case_id"]])
                self.assertEqual(
                    (root / "first" / record["model"]).read_bytes(),
                    (root / "second" / record["model"]).read_bytes(),
                )
                with (
                    np.load(root / "first" / record["inputs"], allow_pickle=False) as inputs1,
                    np.load(root / "second" / record["inputs"], allow_pickle=False) as inputs2,
                ):
                    np.testing.assert_array_equal(inputs1["A"], inputs2["A"])
            self.assertEqual(first["cases"][0]["logical_reference"]["core_dense_equivalent_ops"], "4096")
            self.assertEqual(first["cases"][0]["output_shape"], [32])
            self.assertEqual(first["cases"][1]["logical_reference"]["core_dense_equivalent_ops"], "24576")
            self.assertEqual(first["cases"][1]["logical_reference"]["bias_float_ops"], "192")
            self.assertEqual(json.loads((root / "first" / "manifest.json").read_text())["schema_version"], 1)
            with self.assertRaises(FileExistsError):
                generate_cases(cases, root / "first", 123)
            changed, changed_activation = make_case(cases[0], 999)
            original, original_activation = make_case(cases[0], 123)
            self.assertNotEqual(changed.SerializeToString(), original.SerializeToString())
            self.assertFalse(np.array_equal(changed_activation, original_activation))

    def test_invalid_settings_and_config(self):
        for settings in (
            {"a_shape": ()},
            {"a_shape": (True, 64)},
            {"a_shape": (0, 64)},
            {"n": -1},
            {"bits": 3},
            {"block_size": 24},
            {"dtype": "bfloat16"},
            {"bias": 1},
            {"accuracy_level": True},
            {"cohort": "test"},
        ):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                Case(**({"a_shape": (1, 64), "n": 32} | settings))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = root / "cases.json"
            config.write_text(json.dumps({"cases": [{"a_shape": [1, 64], "n": 32}]}))
            self.assertEqual(load_cases(config), [Case((1, 64), 32)])
            for invalid in ({}, {"cases": []}, {"cases": [{"a_shape": [64], "n": 32, "unknown": 1}]}):
                config.write_text(json.dumps(invalid))
                with self.assertRaises(ValueError):
                    load_cases(config)
            with self.assertRaises(ValueError):
                generate_cases([Case((64,), 32), Case((64,), 32, cohort="held_out")], root / "duplicates", 0)
            with self.assertRaises(ValueError):
                generate_cases(default_cases(), root / "negative_seed", -1)
            self.assertFalse((root / "duplicates").exists())
            self.assertFalse((root / "negative_seed").exists())

    def test_default_pilot_generates(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "pilot"
            manifest = generate_cases(default_cases(), root, 0)
            self.assertEqual(len(manifest["cases"]), 14)
            for record in manifest["cases"]:
                model = onnx.load(root / record["model"])
                onnx.checker.check_model(model)
                self.assertEqual(len(model.graph.input), 1)
                self.assertEqual(len(model.graph.node), 1)


if __name__ == "__main__":
    unittest.main()
