import unittest

import numpy as np
from onnx import TensorProto, helper, numpy_helper

from fuse_hc_post_norm import fuse
from prepare_hc_post_norm import fixture


class PostNormFusionTests(unittest.TestCase):
    def model(self, shared=False):
        return fixture(33, 1, shared, False, [])

    def test_preserves_both_outputs_and_initializers(self):
        model = self.model()
        fused, count = fuse(model)
        self.assertEqual(count, 1)
        self.assertEqual(list(fused.graph.node[0].output), ["y", "normalized"])
        self.assertEqual(list(fused.graph.node[0].input), ["streams", "branch", "gate", "norm"])
        self.assertEqual([t.SerializeToString() for t in model.graph.initializer],
                         [t.SerializeToString() for t in fused.graph.initializer])
        self.assertEqual(len(model.graph.node), 2)

    def test_shared_chain_retains_shared_operands(self):
        fused, _ = fuse(self.model(shared=True))
        self.assertEqual(fused.graph.node[0].op_type, "HCSharedPostNormF16")
        self.assertEqual(list(fused.graph.node[0].input)[-2:], ["shared", "shared_gate"])

    def test_rejects_different_divisor(self):
        model = self.model()
        model.graph.initializer[-2].CopyFrom(numpy_helper.from_array(np.array(3, np.float16), "divisor"))
        with self.assertRaisesRegex(ValueError, "No compatible"):
            fuse(model)

    def test_constant_nodes_are_resolved(self):
        model = self.model()
        divisor = model.graph.initializer[-2]
        node = helper.make_node("Constant", [], ["divisor"], value=divisor)
        del model.graph.initializer[-2]
        model.graph.node.insert(0, node)
        _, count = fuse(model)
        self.assertEqual(count, 1)

    def test_unknown_attributes_are_not_dropped(self):
        model = self.model()
        model.graph.node[1].attribute.append(helper.make_attribute("unknown", 1))
        with self.assertRaisesRegex(ValueError, "No compatible"):
            fuse(model)

    def test_non_fp16_norm_is_not_fused(self):
        model = self.model()
        norm = next(t for t in model.graph.initializer if t.name == "norm")
        norm.data_type = TensorProto.FLOAT
        with self.assertRaisesRegex(ValueError, "No compatible"):
            fuse(model)

    def test_second_application_is_rejected(self):
        model, _ = fuse(self.model())
        with self.assertRaisesRegex(ValueError, "Already fused"):
            fuse(model)


if __name__ == "__main__":
    unittest.main()
