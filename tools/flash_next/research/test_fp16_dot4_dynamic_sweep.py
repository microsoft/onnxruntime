from pathlib import Path
import struct
import tempfile
import unittest

import onnx
from onnx import numpy_helper

from prepare_fp16_dot4_dynamic_sweep import PARENT, dynamic_model, frame_bytes


ROOT = Path(__file__).parent


class DynamicSweepTests(unittest.TestCase):
    def test_binary_frame_layout(self):
        payload, bits = frame_bytes(PARENT / "seed0.onnx")
        self.assertEqual(len(payload), 416)
        self.assertEqual(bits.size, 64)
        self.assertEqual(len(struct.pack("<4sII", b"F16S", 1, 8)), 12)
        self.assertEqual(12 + 1000 * 416, 416012)
        model = onnx.load(PARENT / "seed0.onnx")
        values = {v.name: numpy_helper.to_array(v) for v in model.graph.initializer}
        self.assertEqual(payload[:128], values["a"].tobytes())
        self.assertEqual(payload[128:384], values["b"][:16].tobytes())
        self.assertEqual(payload[384:], values["scales"][:16].tobytes())

    def test_dynamic_graph_inputs(self):
        template = onnx.load(PARENT / "seed0.onnx")
        with tempfile.TemporaryDirectory(prefix="dynamic-sweep-model-") as directory:
            for columns in (1552, 160):
                path = Path(directory) / f"n{columns}.onnx"
                dynamic_model(template, columns, path)
                model = onnx.load(path)
                self.assertEqual(len(model.graph.initializer), 0)
                self.assertEqual([v.name for v in model.graph.input], ["a", "b", "scales"])
                self.assertEqual(model.graph.output[0].type.tensor_type.shape.dim[1].dim_value, columns)
                self.assertEqual([d.dim_value for d in model.graph.input[1].type.tensor_type.shape.dim], [columns, 1, 16])

    def test_single_session_and_reference_gate(self):
        source = (ROOT / "fp16_dot4_dynamic_sweep_compare.cpp").read_text()
        self.assertEqual(source.count("Ort::Session session"), 1)
        self.assertLess(source.index("Ort::Session session"), source.index("for (size_t index ="))
        self.assertIn("static_assert(sizeof(Frame) == 416", source)
        self.assertIn("for (int pattern : {47, 95})", source)
        self.assertIn("inputs.data(), inputs.size()", source)
        runner = (ROOT / "run_fp16_dot4_dynamic_sweep.py").read_text()
        self.assertIn('capture_digest != sha(original_capture)', runner)
        self.assertIn('(\"micro\", 2048), (\"sweep\", 256000)', runner)
        self.assertIn('\"--run-seconds\", \"900\"', runner)


if __name__ == "__main__":
    unittest.main()
