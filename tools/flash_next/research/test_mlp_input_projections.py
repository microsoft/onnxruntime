import unittest

import onnx
from onnx import TensorProto, helper

from fuse_mlp_input_projections import OP, definition, fuse
from prepare_mlp_input_projections import fixture


class MlpInputProjectionTests(unittest.TestCase):
    def original(self):
        model = fixture(onnx.ModelProto(), 1, 0.015625)
        del model.graph.node[:]
        model.graph.node.extend(definition(21).node)
        del model.functions[:]
        for role, width in (("gate", 640), ("up", 640)):
            model.graph.value_info.append(helper.make_tensor_value_info(role, TensorProto.FLOAT16, [1, 1, width]))
        for node in model.graph.node:
            node.name = node.output[0]
        return model

    def test_exact_five_node_chain_and_initializer_preservation(self):
        model = self.original()
        candidate, count = fuse(model)
        self.assertEqual(count, 1)
        self.assertEqual([n.op_type for n in candidate.graph.node], [OP])
        self.assertEqual(list(candidate.graph.node[0].output), ["router", "activated", "shared"])
        self.assertEqual([t.SerializeToString() for t in model.graph.initializer],
                         [t.SerializeToString() for t in candidate.graph.initializer])
        onnx.checker.check_model(candidate)
        with self.assertRaisesRegex(ValueError, "Already fused"):
            fuse(candidate)

    def test_observable_intermediate_and_extra_consumer_are_not_removed(self):
        for output in (True, False):
            model = self.original()
            if output:
                model.graph.output.append(helper.make_tensor_value_info("gate", TensorProto.FLOAT16, [1, 1, 640]))
            else:
                model.graph.node.append(helper.make_node("Identity", ["gate"], ["other"]))
            self.assertEqual(fuse(model)[1], 0)

    def test_shared_activation_input_and_exact_attributes_are_required(self):
        for change in ("input", "bits", "scale_type"):
            model = self.original()
            if change == "input":
                model.graph.node[2].input[0] = "other"
            elif change == "bits":
                next(a for a in model.graph.node[0].attribute if a.name == "bits").i = 3
            else:
                model.graph.initializer[1].data_type = TensorProto.FLOAT
            self.assertEqual(fuse(model)[1], 0)

    def test_ambiguous_router_is_rejected(self):
        model = self.original()
        other = onnx.NodeProto()
        other.CopyFrom(model.graph.node[0])
        other.output[0] = "other_router"
        model.graph.node.insert(1, other)
        model.graph.value_info.append(helper.make_tensor_value_info("other_router", TensorProto.FLOAT16, [1, 1, 512]))
        self.assertEqual(fuse(model)[1], 0)


if __name__ == "__main__":
    unittest.main()
