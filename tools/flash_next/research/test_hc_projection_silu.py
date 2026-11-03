from pathlib import Path
import unittest

import onnx
from onnx import helper

from build_hc_projection_silu import program_source
from prepare_hc_projection_silu import DOMAIN, fuse


class HCProjectionSiLUTests(unittest.TestCase):
    def graph(self):
        matmul = helper.make_node("MatMulNBits", ["x", "weights", "scales"], ["projected"],
            name="projection", domain=DOMAIN, K=10240, N=324, block_size=32, bits=4)
        split = helper.make_node("SplitScaledSiLUF16", ["projected", "sizes"], ["act", "tail"],
            name="split", domain=DOMAIN, alpha=0.25)
        return helper.make_model(helper.make_graph([matmul, split], "test", [], []))

    def test_exact_scope(self):
        model, count = fuse(self.graph())
        self.assertEqual(count, 1)
        node = model.graph.node[0]
        self.assertEqual(node.op_type, "HCProjectionSiLUF16")
        self.assertEqual(list(node.input), ["x", "sizes", "weights", "scales"])
        attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
        self.assertEqual(attrs["accuracy_level"], 4)
        self.assertEqual(attrs["alpha"], 0.25)

    def test_other_consumers_and_unsupported_shapes_preserved(self):
        for variant in ("consumer", "output", "shape", "bias"):
            model = self.graph()
            if variant == "consumer":
                model.graph.node.append(helper.make_node("Identity", ["projected"], ["extra"]))
            elif variant == "output":
                model.graph.output.append(helper.make_tensor_value_info("projected", onnx.TensorProto.FLOAT16, None))
            elif variant == "shape":
                next(a for a in model.graph.node[0].attribute if a.name == "N").i = 512
            else:
                model.graph.node[0].input.extend(["zero_points"])
            candidate, count = fuse(model)
            self.assertEqual(count, 0)
            self.assertEqual(candidate.SerializeToString(), model.SerializeToString())

    def test_projection_arithmetic_and_activation_rounding_preserved(self):
        path = Path("/home/sayanshaw/repos/onnxruntime-webgpu-h200-perf/build-int4-shared-post-v4/matmul_nbits.cc")
        original = path.read_text().split("class HCExactDenseProgram final", 1)[1].split(
            "\nclass MatMulNBitsSkinnyProgram", 1)[0]
        generated = program_source(path.read_text())
        start = '    code << "let lane = local_idx % 32u;\\n"'
        end = '         << "  for (var i = 0u; i < 32u; i++) { value += partials[local_idx + i]; }\\n"'
        self.assertEqual(original.split(start, 1)[1].split(end, 1)[0],
                         generated.split(start, 1)[1].split(end, 1)[0])
        for expression in ("activated_element_t(f32(value) * uniforms.alpha)",
                           "activated_element_t(stable_sigmoid(f32(z_t)))",
                           "activated_element_t(f32(z_t) * f32(sigmoid_t))"):
            self.assertIn(expression, generated)


if __name__ == "__main__":
    unittest.main()
