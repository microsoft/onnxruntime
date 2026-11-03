from pathlib import Path
import unittest

from onnx import helper

from prepare_hc_up_premix import DOMAIN, fuse


class HCUpPreMixTests(unittest.TestCase):
    def graph(self):
        return helper.make_model(helper.make_graph([
            helper.make_node("MatMulNBits", ["x", "weights", "scales"], ["projected"],
                name="projection", domain=DOMAIN, K=320, N=10240, block_size=32, bits=4),
            helper.make_node("HyperConnectionPreMixSigmoidF16", ["streams", "projected"], ["y"],
                name="mix", domain=DOMAIN, num_branches=4, reduction_scale=0.25),
        ], "test", [], []))

    def test_exact_scope_and_attribute_preservation(self):
        model, count = fuse(self.graph())
        self.assertEqual(count, 1)
        node = model.graph.node[0]
        self.assertEqual(node.op_type, "HCUpPreMixF16")
        self.assertEqual(list(node.input), ["streams", "x", "weights", "scales"])
        attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
        self.assertEqual(attrs["accuracy_level"], 4)
        self.assertEqual(attrs["reduction_scale"], 0.25)

    def test_shared_or_unsupported_projections_are_not_removed(self):
        for variant in ("consumer", "shape", "branches", "optional"):
            model = self.graph()
            if variant == "consumer":
                model.graph.node.append(helper.make_node("Identity", ["projected"], ["other"]))
            elif variant == "shape":
                next(a for a in model.graph.node[0].attribute if a.name == "K").i = 2560
            elif variant == "branches":
                next(a for a in model.graph.node[1].attribute if a.name == "num_branches").i = 0
            else:
                model.graph.node[0].input.append("zero_points")
            candidate, count = fuse(model)
            self.assertEqual(count, 0)
            self.assertEqual(candidate.SerializeToString(), model.SerializeToString())

    def test_rounding_and_branch_order_are_explicit(self):
        source = Path(__file__).with_name("hc_up_premix.cc").read_text()
        for text in ("for (var i = 0u; i < 32u; i++) { value += partials[local_idx + i]; }",
                     "gates[branch] = f16(1.0 / (1.0 + exp(-value)))",
                     "for (var c = 0u; c < uniforms.branches; c++)",
                     "y_element_t(sum * uniforms.reduction_scale)",
                     "auto projected = context.CreateGPUTensor",
                     "ApplyMatMulNBits(a, b, scales, nullptr, nullptr, 320, 10240"):
            self.assertIn(text, source)
        self.assertLess(source.index("helper.OutputShape() == streams->Shape()"),
                        source.index("return static_replay_.Run"))


if __name__ == "__main__":
    unittest.main()
