import unittest
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from fuse_int4_mlp import fuse_int4_mlp


def pattern():
    tensors = [
        numpy_helper.from_array(np.zeros((8, 1, 16), np.uint8), "gate_b"),
        numpy_helper.from_array(np.full((8, 1, 16), 3, np.uint8), "up_b"),
        numpy_helper.from_array(np.ones((8, 1), np.float16), "gate_scales"),
        numpy_helper.from_array(np.ones((8, 1), np.float16), "up_scales"),
    ]
    attrs = dict(K=32, N=8, bits=4, block_size=32, accuracy_level=4)
    nodes = [
        helper.make_node("MatMulNBits", ["a", "gate_b", "gate_scales"], ["gate"],
                         domain="com.microsoft", **attrs),
        helper.make_node("MatMulNBits", ["a", "up_b", "up_scales"], ["up"],
                         domain="com.microsoft", **attrs),
        helper.make_node("Sigmoid", ["gate"], ["sigmoid"]),
        helper.make_node("Mul", ["gate", "sigmoid"], ["activation"]),
        helper.make_node("Mul", ["activation", "up"], ["y"]),
    ]
    return helper.make_model(
        helper.make_graph(nodes, "pattern",
                          [helper.make_tensor_value_info("a", TensorProto.FLOAT16, [1, 1, 32])],
                          [helper.make_tensor_value_info("y", TensorProto.FLOAT16, [1, 1, 8])],
                          tensors),
        opset_imports=[helper.make_opsetid("", 22), helper.make_opsetid("com.microsoft", 1)])


class FusionTests(unittest.TestCase):
    def test_fuses_without_norm_and_preserves_source_and_weights(self):
        source = pattern()
        before = source.SerializeToString()
        fused, report = fuse_int4_mlp(source)
        self.assertEqual(source.SerializeToString(), before)
        self.assertEqual(report["fusions"], 1)
        self.assertEqual(report["nodes_after"], 1)
        self.assertEqual(fused.graph.node[0].op_type, "MatMulNBitsMlp")
        self.assertEqual(list(fused.graph.node[0].input),
                         ["a", "", "", "gate_b", "gate_scales", "", "up_b", "up_scales", ""])
        self.assertEqual(list(fused.graph.node[0].output), ["y"])
        self.assertEqual([t.SerializeToString() for t in source.graph.initializer],
                         [t.SerializeToString() for t in fused.graph.initializer])
        onnx.checker.check_model(fused)

    def test_commuted_mul_inputs(self):
        source = pattern()
        for index in (3, 4):
            source.graph.node[index].input.reverse()
        self.assertEqual(fuse_int4_mlp(source)[1]["fusions"], 1)

    def test_preserves_shared_input_producer(self):
        source = pattern()
        source.graph.input[0].name = "original"
        source.graph.node.insert(0, helper.make_node("Identity", ["original"], ["a"]))
        fused, report = fuse_int4_mlp(source)
        self.assertEqual(report["fusions"], 1)
        self.assertEqual([n.op_type for n in fused.graph.node], ["Identity", "MatMulNBitsMlp"])
        self.assertEqual(fused.graph.node[1].input[0], "a")
        onnx.checker.check_model(fused)

    def test_extra_consumer_prevents_fusion(self):
        source = pattern()
        source.graph.node.append(helper.make_node("Identity", ["gate"], ["extra"]))
        self.assertEqual(fuse_int4_mlp(source)[1]["fusions"], 0)

    def test_intermediate_graph_output_prevents_fusion(self):
        source = pattern()
        source.graph.output.append(helper.make_tensor_value_info("activation", TensorProto.FLOAT16, [1, 1, 8]))
        self.assertEqual(fuse_int4_mlp(source)[1]["fusions"], 0)

    def test_attributes_must_match(self):
        source = pattern()
        for attr in source.graph.node[1].attribute:
            if attr.name == "accuracy_level":
                attr.i = 0
        self.assertEqual(fuse_int4_mlp(source)[1]["fusions"], 0)

    def test_optional_quantization_inputs_not_supported(self):
        source = pattern()
        source.graph.node[0].input.append("")
        self.assertEqual(fuse_int4_mlp(source)[1]["fusions"], 0)

    def test_weight_shape_must_match(self):
        source = pattern()
        source.graph.initializer[0].dims[1] = 2
        self.assertEqual(fuse_int4_mlp(source)[1]["fusions"], 0)

    def test_subgraphs_rejected_before_removing_implicit_uses(self):
        source = pattern()
        branch = helper.make_graph(
            [helper.make_node("Identity", ["gate"], ["branch_result"])], "branch", [],
            [helper.make_tensor_value_info("branch_result", TensorProto.FLOAT16, [1, 1, 8])])
        source.graph.initializer.append(numpy_helper.from_array(np.array(True), "condition"))
        source.graph.node.append(helper.make_node("If", ["condition"], ["extra"],
                                                 then_branch=branch, else_branch=branch))
        with self.assertRaisesRegex(ValueError, "implicit-input"):
            fuse_int4_mlp(source)

    def test_no_duplicate_fusion_of_one_pattern(self):
        fused, report = fuse_int4_mlp(pattern())
        self.assertEqual(report["removed_nodes"], 5)
        self.assertEqual(fuse_int4_mlp(fused)[1]["fusions"], 0)

    def run_cli(self, source, output):
        return subprocess.run(
            [sys.executable, str(Path(__file__).with_name("fuse_int4_mlp.py")),
             str(source), "--output", str(output)],
            capture_output=True, text=True)

    def test_cli_checks_before_publishing_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            source, output = (Path(temporary) / name for name in ("source.onnx", "output.onnx"))
            model = pattern()
            for index in (0, 1):
                model.graph.node[index].input[0] = "missing"
            onnx.save(model, source)
            result = self.run_cli(source, output)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("ValidationError", result.stderr)
            self.assertFalse(output.exists())
            self.assertEqual(sorted(p.name for p in Path(temporary).iterdir()), ["source.onnx"])

    def test_cli_does_not_overwrite_existing_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            source, output = (Path(temporary) / name for name in ("source.onnx", "output.onnx"))
            onnx.save(pattern(), source)
            output.write_bytes(b"preserve")
            result = self.run_cli(source, output)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("FileExistsError", result.stderr)
            self.assertEqual(output.read_bytes(), b"preserve")

    def test_cli_does_not_follow_dangling_output_symlink(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, output, target = (root / name for name in ("source.onnx", "output.onnx", "target"))
            onnx.save(pattern(), source)
            output.symlink_to(target)
            result = self.run_cli(source, output)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("FileExistsError", result.stderr)
            self.assertTrue(output.is_symlink())
            self.assertFalse(target.exists())

    def test_cli_publishes_checked_model(self):
        with tempfile.TemporaryDirectory() as temporary:
            source, output = (Path(temporary) / name for name in ("source.onnx", "output.onnx"))
            onnx.save(pattern(), source)
            result = self.run_cli(source, output)
            self.assertEqual(result.returncode, 0, result.stderr)
            onnx.checker.check_model(str(output))
            self.assertEqual(onnx.load(output).graph.node[0].op_type, "MatMulNBitsMlp")


if __name__ == "__main__":
    unittest.main()
