# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import sys
import unittest
from pathlib import Path

from onnx import TensorProto, helper

transformers_source = Path(__file__).resolve().parents[3] / "python" / "tools" / "transformers"
if transformers_source.is_dir():
    sys.path.append(str(transformers_source))
    from onnx_model import OnnxModel
else:
    from onnxruntime.transformers.onnx_model import OnnxModel


class TestOnnxModel(unittest.TestCase):
    @staticmethod
    def _create_model():
        then_node = helper.make_node("Identity", ["left_output"], ["then_output"], name="then_node")
        then_graph = helper.make_graph(
            [then_node],
            "then_graph",
            [],
            [helper.make_tensor_value_info("then_output", TensorProto.FLOAT, [1])],
        )
        else_node = helper.make_node("Identity", ["right"], ["else_output"], name="else_node")
        else_graph = helper.make_graph(
            [else_node],
            "else_graph",
            [],
            [helper.make_tensor_value_info("else_output", TensorProto.FLOAT, [1])],
        )

        producer = helper.make_node("Producer", ["input"], ["left", "", "right"], name="producer", domain="Test")
        left_child = helper.make_node("Add", ["left", "left"], ["left_output"], name="left_child")
        right_child = helper.make_node("Identity", ["right"], ["right_output"], name="right_child")
        missing_parent = helper.make_node("Identity", ["missing"], ["missing_output"], name="missing_parent")
        branch = helper.make_node(
            "If",
            ["condition"],
            ["output"],
            name="branch",
            then_branch=then_graph,
            else_branch=else_graph,
        )
        graph = helper.make_graph(
            [producer, left_child, right_child, missing_parent, branch],
            "main_graph",
            [
                helper.make_tensor_value_info("input", TensorProto.FLOAT, [1]),
                helper.make_tensor_value_info("condition", TensorProto.BOOL, []),
            ],
            [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1])],
        )
        return helper.make_model(graph), producer, left_child, right_child, missing_parent, then_node, else_node

    def test_name_maps_include_subgraphs_by_default(self):
        model, producer, _, _, _, then_node, else_node = self._create_model()
        onnx_model = OnnxModel(model)

        input_map = onnx_model.input_name_to_nodes()
        self.assertNotIn("", input_map)
        self.assertEqual([node.name for node in input_map["left"]], ["left_child", "left_child"])
        self.assertEqual(input_map["left_output"][0].name, then_node.name)
        self.assertEqual(input_map["right"][1].name, else_node.name)

        output_map = onnx_model.output_name_to_node()
        self.assertNotIn("", output_map)
        self.assertEqual(output_map["left"].name, producer.name)
        self.assertEqual(output_map["then_output"].name, then_node.name)
        self.assertEqual(output_map["else_output"].name, else_node.name)

        main_input_map = onnx_model.input_name_to_nodes(exclude_subgraphs=True)
        main_output_map = onnx_model.output_name_to_node(exclude_subgraphs=True)
        self.assertNotIn("left_output", main_input_map)
        self.assertNotIn("then_output", main_output_map)
        self.assertNotIn("else_output", main_output_map)

    def test_navigation_preserves_order_duplicates_and_output_selection(self):
        model, producer, left_child, right_child, missing_parent, _, _ = self._create_model()
        onnx_model = OnnxModel(model)
        input_map = onnx_model.input_name_to_nodes()
        output_map = onnx_model.output_name_to_node()

        self.assertEqual(
            [node.name for node in onnx_model.get_children(producer, input_map)],
            ["left_child", "left_child", "right_child", "else_node"],
        )
        self.assertEqual(
            [node.name for node in onnx_model.get_children(producer, input_map, output_index=0)],
            ["left_child", "left_child"],
        )
        self.assertEqual(onnx_model.get_children(producer, input_map, output_index=1), [])
        self.assertEqual(
            [node.name for node in onnx_model.get_children(producer, input_map, output_index=2)],
            ["right_child", "else_node"],
        )
        self.assertEqual(onnx_model.get_children(producer, input_map, output_index=3), [])

        self.assertEqual(
            [node.name for node in onnx_model.get_parents(left_child, output_map)], ["producer", "producer"]
        )
        self.assertEqual(onnx_model.get_parent(left_child, 0, output_map).name, producer.name)
        self.assertIsNone(onnx_model.get_parent(missing_parent, 0, output_map))
        self.assertIsNone(onnx_model.get_parent(left_child, 2, output_map))
        self.assertEqual(onnx_model.get_parent(right_child, 0, output_map).name, producer.name)


if __name__ == "__main__":
    unittest.main()
