from pathlib import Path
import json
import tempfile
import unittest

import numpy as np
import onnx
from build_tensor_pairwise_probe import PAIRWISE, transform
from build_tensor_product_probe import transform as products
from prepare_tensor_dot4_probe import main as prepare


SOURCE = Path("/home/sayanshaw/repos/qwen38-int4-dense-block32/build-v1/matmul_nbits.cc")


class TensorPairwiseProbeTests(unittest.TestCase):
    def test_only_isolated_product_reconstruction_changes(self):
        original = products(SOURCE.read_text())
        actual = transform(SOURCE.read_text())
        restored = actual.replace('"TensorProductPairwiseProbeF16"', '"TensorProductProbeF16"')
        restored = restored.replace(PAIRWISE, "  return dot(vec4<f16>(1), p);")
        self.assertEqual(restored, original)
        self.assertIn("pack2x16float", actual)
        self.assertIn("subgroupMatrixStore<row_major>(&products", actual)
        self.assertIn("Tensor dot4 probe only supports the isolated FP16", actual)

    def test_default_fixture_generation_is_byte_identical(self):
        original = Path("/home/sayanshaw/repos/qwen38-int4-tensor-dot4-probe/fixtures-v1/manifest.json")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "fixtures"
            prepare(root=root)
            self.assertEqual(json.loads((root / "manifest.json").read_text()), json.loads(original.read_text()))

    def test_single_product_fixtures_keep_all_other_weights_zero(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "fixtures"
            prepare(root=root, terms=1)
            manifest = json.loads((root / "manifest.json").read_text())
            self.assertEqual([row["nonzero_k_positions"] for row in manifest], [[0], [4], [12]])
            for row in manifest:
                graph = onnx.load(root / row["fixture"]).graph
                weights = onnx.numpy_helper.to_array(next(t for t in graph.initializer if t.name == "weights")).copy()
                weights[:, 0, row["nonzero_k_positions"][0] // 2] = 0x88
                self.assertTrue(np.all(weights == 0x88))
            for terms in (True, 0, 2, 5):
                with self.subTest(terms=terms), self.assertRaises(ValueError):
                    prepare(root=root, terms=terms)


if __name__ == "__main__":
    unittest.main()
