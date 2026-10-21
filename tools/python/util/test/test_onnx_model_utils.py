# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pathlib
import unittest

import onnx
from onnx import TensorProto, helper, shape_inference

from ..mobile_helpers.usability_checker import check_shapes
from ..onnx_model_utils import (
    fix_output_shapes,
    fix_output_shapes_path,
    get_producer_consumer_maps,
    is_fixed_size_tensor,
    make_dim_param_fixed,
    make_input_shape_fixed,
)

script_dir = pathlib.Path(__file__).parent
ort_root = script_dir.parents[3]

# example usage from <ort root>/tools/python
# python -m unittest util/test/test_onnx_model_utils.py
# NOTE: at least on Windows you must use that as the working directory for all the imports to be happy


class TestGetProducerConsumerMaps(unittest.TestCase):
    @staticmethod
    def _create_model():
        # create a model with subgraphs and various types of shadowing
        body = helper.make_graph(
            [
                # shadow a1 in main graph.
                # LoopAdd_SubgraphOutput should be linked to this and a1 should not be an implicit input
                helper.make_node("Add", ["loop_state_in", "iter"], ["a1"], "LoopAdd_Shadows"),
                # main_graph_initializer should be handled (implicit input but no producer node)
                # graph input 'x' from main graph should also be handled
                helper.make_node("Add", ["main_graph_initializer", "x"], ["a2"], "LoopAdd_OuterScopeInitializer"),
                # implicit input should be handled - 'z' can be accessed from outside scope
                # Add2 in main graph should be implicit input of the Loop node
                helper.make_node("Add", ["z", "a1"], ["a3"], "LoopAdd_ImplicitInput"),
                # create subgraph output
                helper.make_node("Add", ["a2", "a3"], ["loop_state_out"], "LoopAdd_SubgraphOutput"),
            ],
            "Loop_body",
            [
                helper.make_tensor_value_info("iter", TensorProto.INT64, [1]),
                helper.make_tensor_value_info("cond", TensorProto.BOOL, [1]),
                helper.make_tensor_value_info("loop_state_in", TensorProto.FLOAT, [1]),
            ],
            [
                helper.make_tensor_value_info("cond", TensorProto.BOOL, [1]),
                helper.make_tensor_value_info("loop_state_out", TensorProto.FLOAT, [1]),
            ],
            [],
        )

        # Create the main graph
        graph_proto = helper.make_graph(
            [
                # create 'a1' which is shadowed in the subgraph. node should not be joined to Loop1
                helper.make_node("Add", ["x", "y"], ["a1"], "Add1"),
                # create 'z' which is an implicit input to subgraph. node should be joined to Loop1
                helper.make_node("Add", ["a1", "main_graph_initializer"], ["z"], "Add2"),
                # rename 'z' to use as explicit input to Loop
                helper.make_node("Identity", ["z"], ["state_var_in"], "RenameZ"),
                helper.make_node(
                    "Loop", ["max_trip_count", "keep_going", "state_var_in"], ["state_var_out"], "Loop1", body=body
                ),
                helper.make_node("Sub", ["a1", "state_var_out"], ["graph_output"], "sub_1"),
            ],
            "Main_graph",
            [
                helper.make_tensor_value_info("x", TensorProto.FLOAT, [1]),
                helper.make_tensor_value_info("y", TensorProto.FLOAT, [1]),
            ],
            [
                helper.make_tensor_value_info("graph_output", TensorProto.FLOAT, [1]),
            ],
            [
                helper.make_tensor("max_trip_count", TensorProto.INT64, [1], [2]),
                helper.make_tensor("main_graph_initializer", TensorProto.FLOAT, [1], [1.0]),
                helper.make_tensor("keep_going", TensorProto.BOOL, [1], [True]),
            ],
        )

        return helper.make_model(graph_proto)

    def test_model_with_subgraph(self):
        """
        Test a manually created model that has a subgraph and implicit inputs of all possible types.
        """

        model = self._create_model()
        node_to_producers, node_to_consumers = get_producer_consumer_maps(model.graph)

        main_graph_add_create_a1 = model.graph.node[0]
        main_graph_add_create_z = model.graph.node[1]
        main_graph_rename_z = model.graph.node[2]
        main_graph_loop = model.graph.node[3]
        main_graph_sub = model.graph.node[4]

        subgraph = main_graph_loop.attribute[0].g
        loop_add_shadow = subgraph.node[0]
        loop_add_outer_scope_init = subgraph.node[1]
        loop_add_implicit_input = subgraph.node[2]
        loop_add_subgraph_output = subgraph.node[3]

        def node_name(node):
            return f"{node.name}:{node.op_type}"

        def check_linked(producer, consumer):
            self.assertTrue(
                producer in node_to_producers[consumer],
                f"{node_name(producer)} not in producers for {node_name(consumer)}",
            )
            self.assertTrue(
                consumer in node_to_consumers[producer],
                f"{node_name(consumer)} not in consumers for {node_name(producer)}",
            )

        def check_not_linked(producer, consumer):
            self.assertFalse(
                producer in node_to_producers[consumer], f"{node_name(producer)} in producers for {node_name(consumer)}"
            )
            self.assertFalse(
                consumer in node_to_consumers[producer],
                f"{node_name(consumer)} not in consumers for {node_name(producer)}",
            )

        check_linked(main_graph_add_create_a1, main_graph_add_create_z)
        # a1 in main graph shouldn't be implicit input to loop as it is shadowed
        check_not_linked(main_graph_add_create_a1, main_graph_loop)
        # z is implicit input
        check_linked(main_graph_add_create_z, main_graph_loop)
        check_linked(main_graph_rename_z, main_graph_loop)
        check_linked(main_graph_loop, main_graph_sub)

        # check subgraph
        check_linked(loop_add_shadow, loop_add_implicit_input)
        check_linked(loop_add_outer_scope_init, loop_add_subgraph_output)
        check_linked(loop_add_implicit_input, loop_add_subgraph_output)


class TestDynamicDimReplacement(unittest.TestCase):
    def test_replace_symbolic_dim(self):
        """
        Update a model with a single symbolic input dimension. After replacement run shape inferencing to verify that
        all shapes in the model have fixed sizes.
        """
        model_path = (
            ort_root / "onnxruntime" / "test" / "testdata" / "CNTK" / "test_LSTM.tanh.bidirectional" / "model.onnx"
        )

        model = onnx.load_model(str(model_path))

        # validate the expected input after inferring shape info
        m2 = shape_inference.infer_shapes(model, True)
        dynamic_inputs, num_dynamic_values = check_shapes(m2.graph)
        self.assertEqual(len(dynamic_inputs), 1)
        self.assertEqual(dynamic_inputs[0].name, "Input3")
        self.assertGreater(num_dynamic_values, 0)

        # update original model
        make_dim_param_fixed(model.graph, "None", 4)

        # and validate the model no longer has dynamic values
        model = shape_inference.infer_shapes(model, True)
        dynamic_inputs, num_dynamic_values = check_shapes(model.graph)
        self.assertFalse(dynamic_inputs)
        self.assertEqual(num_dynamic_values, 0)

    def test_replace_input_shape(self):
        """
        Replace the entire shape for an input. This can be used when the model has inputs with unknown dimensions
        i.e. the dimension has no value and no symbolic name so it's harder to replace.
        """
        model_path = ort_root / "onnxruntime" / "test" / "testdata" / "gh_issue_9671.onnx"

        model = onnx.load_model(str(model_path))

        # validate the expected input after inferring shape info
        m2 = shape_inference.infer_shapes(model, True)
        dynamic_inputs, num_dynamic_values = check_shapes(m2.graph)
        self.assertEqual(len(dynamic_inputs), 3)
        self.assertEqual(dynamic_inputs[0].name, "X1")
        self.assertEqual(dynamic_inputs[1].name, "X2")
        self.assertEqual(dynamic_inputs[2].name, "X3")
        self.assertGreater(num_dynamic_values, 0)

        # update original model
        make_input_shape_fixed(model.graph, "X1", [2, 2, 4])
        make_input_shape_fixed(model.graph, "X2", [2, 4])
        make_input_shape_fixed(model.graph, "X3", [2, 2, 4])

        # and validate the model no longer has dynamic values
        model = shape_inference.infer_shapes(model, True)
        dynamic_inputs, num_dynamic_values = check_shapes(model.graph)
        self.assertFalse(dynamic_inputs)

    def test_replace_input_shape_with_dim_params(self):
        # replace the input shape where the existing shape also has dim_param entries.
        # in this case we should also iterate the rest of the model and replace other instances
        # of the dim_param with the new value.
        model_path = ort_root / "onnxruntime" / "test" / "testdata" / "fuse_mul_1.onnx"
        model = onnx.load_model(str(model_path))

        m2 = shape_inference.infer_shapes(model, True)
        dynamic_inputs, num_dynamic_values = check_shapes(m2.graph)
        self.assertEqual(len(dynamic_inputs), 1)
        self.assertEqual(dynamic_inputs[0].name, "X1")
        # input as well as other values in model have shape ['D'] so check > 1
        self.assertGreater(num_dynamic_values, 1)

        # replace X1's shape of ['D'] -> [4]
        make_input_shape_fixed(model.graph, "X1", [4])

        # validate the model no longer has dynamic values
        # we don't run shape_inference here as 'D' is the only dimension in the whole model, and we should have
        # replaced every instance of it if _make_input_shape_fixed worked as expected.
        model = shape_inference.infer_shapes(model, True)
        dynamic_inputs, num_dynamic_values = check_shapes(model.graph)
        self.assertFalse(dynamic_inputs)
        self.assertEqual(num_dynamic_values, 0)

    def test_fix_output_shape(self):
        """
        Replace an input shape in a model where that won't update the output shape automatically.
        Manually fix the output so the usage of the model is clearer.
        """
        model_path = ort_root / "onnxruntime" / "test" / "testdata" / "transform" / "fusion" / "bias_gelu_fusion.onnx"
        model = onnx.load_model(str(model_path))

        make_input_shape_fixed(model.graph, "A", [2, 2, 3072])

        # symbolic dim names in graph inputs don't match graph outputs so they won't have been updated yet
        self.assertFalse(is_fixed_size_tensor(model.graph.output[0]))
        fix_output_shapes(model)
        self.assertTrue(is_fixed_size_tensor(model.graph.output[0]))

    def test_fix_output_shape_path(self):
        """
        Replace an input shape in a model where that won't update the output shape automatically.
        Manually fix the output so the usage of the model is clearer.
        """
        model_path = ort_root / "onnxruntime" / "test" / "testdata" / "transform" / "fusion" / "bias_gelu_fusion.onnx"
        output_path = ort_root / "onnxruntime" / "test" / "testdata" / "transform" / \
            "fusion" / "bias_gelu_fusion_fixed_output.onnx"
        model = onnx.load_model(str(model_path))

        make_input_shape_fixed(model.graph, "A", [2, 2, 3072])

        # symbolic dim names in graph inputs don't match graph outputs so they won't have been updated yet
        self.assertFalse(is_fixed_size_tensor(model.graph.output[0]))
        fix_output_shapes_path(model, model_path, output_path)
        self.assertTrue(is_fixed_size_tensor(model.graph.output[0]))

        if output_path.exists():
            output_path.unlink()

    def test_invalid_replace_input_shape(self):
        model_path = ort_root / "onnxruntime" / "test" / "testdata" / "sklearn_bin_voting_classifier_soft.onnx"
        model = onnx.load_model(str(model_path))
        # test some invalid usages
        self.assertRaisesRegex(
            ValueError,
            "Rank mismatch. Existing:2 Replacement:3",
            make_input_shape_fixed,
            model.graph,
            "input",
            [1, 2, 3],
        )

        self.assertRaisesRegex(
            ValueError,
            "Can't replace existing fixed size of 2 with 3 for dimension 2",
            make_input_shape_fixed,
            model.graph,
            "input",
            [4, 3],
        )

        self.assertRaisesRegex(
            ValueError, "Input X1 was not found in graph inputs.", make_input_shape_fixed, model.graph, "X1", [2, 3]
        )
