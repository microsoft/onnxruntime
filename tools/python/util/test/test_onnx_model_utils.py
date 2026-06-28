# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pathlib
import unittest

import onnx
from onnx import TensorProto, helper, shape_inference

from ..mobile_helpers.usability_checker import check_shapes
from ..onnx_model_utils import (
    fix_output_shapes,
    get_producer_consumer_maps,
    is_fixed_size_tensor,
    make_dim_param_fixed,
    make_input_shape_fixed,
    update_tensor_metadata_for_permutation,
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


class TestUpdateTensorMetadataForPermutation(unittest.TestCase):
    def _make_dim_value(self, val: int):
        d = onnx.TensorShapeProto.Dimension()
        d.dim_value = val
        return d

    def _make_dim_param(self, name: str):
        d = onnx.TensorShapeProto.Dimension()
        d.dim_param = name
        return d

    def _make_value_info(self, name: str, dims):
        vi = helper.make_tensor_value_info(name, TensorProto.FLOAT, None)
        # overwrite with explicit dims to mix value/param
        shape = onnx.TensorShapeProto()
        for d in dims:
            shape.dim.append(d)
        vi.type.tensor_type.shape.CopyFrom(shape)
        return vi

    def test_updates_value_info_input_output(self):
        # Create a graph where the same tensor name B is input, output, and has a value_info
        dims = [self._make_dim_value(2), self._make_dim_value(3)]
        g_in = self._make_value_info("B", dims)
        g_out = self._make_value_info("B", dims)
        vinfo = self._make_value_info("B", dims)
        graph = helper.make_graph(nodes=[], name="g", inputs=[g_in], outputs=[g_out], initializer=[])
        graph.value_info.extend([vinfo])
        pre_counts = (len(graph.value_info), len(graph.input), len(graph.output))
        updated = update_tensor_metadata_for_permutation(graph, "B", [1, 0])
        self.assertEqual(updated, 3)

        for coll in (graph.value_info, graph.input, graph.output):
            vi = next(v for v in coll if v.name == "B")
            self.assertEqual(vi.type.tensor_type.shape.dim[0].dim_value, 3)
            self.assertEqual(vi.type.tensor_type.shape.dim[1].dim_value, 2)
        post_counts = (len(graph.value_info), len(graph.input), len(graph.output))
        self.assertEqual(pre_counts, post_counts)

    def test_preserves_dim_semantics(self):
        # First dim symbolic M, second is concrete 3 -> after perm becomes [3, M]
        dims = [self._make_dim_param("M"), self._make_dim_value(3)]
        vinfo = self._make_value_info("B", dims)
        graph = helper.make_graph([], "g", [], [])
        graph.value_info.extend([vinfo])

        updated = update_tensor_metadata_for_permutation(graph, "B", [1, 0])
        self.assertEqual(updated, 1)
        vi = next(v for v in graph.value_info if v.name == "B")
        self.assertTrue(vi.type.tensor_type.shape.dim[0].HasField("dim_value"))
        self.assertEqual(vi.type.tensor_type.shape.dim[0].dim_value, 3)
        self.assertTrue(vi.type.tensor_type.shape.dim[1].HasField("dim_param"))
        self.assertEqual(vi.type.tensor_type.shape.dim[1].dim_param, "M")

    def test_invalid_perm_strict(self):
        dims = [self._make_dim_value(2), self._make_dim_value(3)]
        vinfo = self._make_value_info("B", dims)
        graph = helper.make_graph([], "g", [], [])
        graph.value_info.extend([vinfo])

        with self.assertRaises(ValueError):
            update_tensor_metadata_for_permutation(graph, "B", [0, 2], strict_mode=True)
        with self.assertRaises(ValueError):
            update_tensor_metadata_for_permutation(graph, "B", [0, 0], strict_mode=True)

        # Non-strict: invalid indices out of range -> no update
        self.assertEqual(update_tensor_metadata_for_permutation(graph, "B", [0, 2], strict_mode=False), 0)

    def test_rank_mismatch_behavior(self):
        # Rank 1 value for B, perm requires rank 2
        dims = [self._make_dim_value(5)]
        vinfo = self._make_value_info("B", dims)
        graph = helper.make_graph([], "g", [], [])
        graph.value_info.extend([vinfo])

        # Non-strict: no-op
        self.assertEqual(update_tensor_metadata_for_permutation(graph, "B", [1, 0], strict_mode=False), 0)
        # Strict: error
        with self.assertRaises(ValueError):
            update_tensor_metadata_for_permutation(graph, "B", [1, 0], strict_mode=True)

    def test_scoped_updates_only(self):
        dims_b = [self._make_dim_value(2), self._make_dim_value(3)]
        dims_c = [self._make_dim_value(4), self._make_dim_value(5)]
        vb = self._make_value_info("B", dims_b)
        vc = self._make_value_info("C", dims_c)
        graph = helper.make_graph([], "g", [], [])
        graph.value_info.extend([vb, vc])

        updated = update_tensor_metadata_for_permutation(graph, "B", [1, 0])
        self.assertEqual(updated, 1)
        b = next(v for v in graph.value_info if v.name == "B")
        c = next(v for v in graph.value_info if v.name == "C")
        self.assertEqual([d.dim_value for d in b.type.tensor_type.shape.dim], [3, 2])
        self.assertEqual([d.dim_value for d in c.type.tensor_type.shape.dim], [4, 5])

    def test_subgraph_update(self):
        # Create a subgraph on its own and update that, verifying parent graph is unchanged.
        dims = [self._make_dim_value(2), self._make_dim_value(3)]
        sub_vi = self._make_value_info("B", dims)
        subgraph = helper.make_graph([], "sub", [], [])
        subgraph.value_info.extend([sub_vi])

        parent_vi = self._make_value_info("B", [self._make_dim_value(2), self._make_dim_value(3)])
        parent = helper.make_graph([], "parent", [], [])
        parent.value_info.extend([parent_vi])

        updated = update_tensor_metadata_for_permutation(subgraph, "B", [1, 0])
        self.assertEqual(updated, 1)
        # subgraph updated
        sv = next(v for v in subgraph.value_info if v.name == "B")
        self.assertEqual([d.dim_value for d in sv.type.tensor_type.shape.dim], [3, 2])
        # parent unchanged
        pv = next(v for v in parent.value_info if v.name == "B")
        self.assertEqual([d.dim_value for d in pv.type.tensor_type.shape.dim], [2, 3])

    def test_repeat_application_safe(self):
        # Applying permutation twice with [1,0] returns to original shape
        dims = [self._make_dim_value(2), self._make_dim_value(3)]
        vinfo = self._make_value_info("B", dims)
        graph = helper.make_graph([], "g", [], [])
        graph.value_info.extend([vinfo])

        update_tensor_metadata_for_permutation(graph, "B", [1, 0])
        vi = next(v for v in graph.value_info if v.name == "B")
        self.assertEqual([d.dim_value for d in vi.type.tensor_type.shape.dim], [3, 2])

        update_tensor_metadata_for_permutation(graph, "B", [1, 0])
        vi2 = next(v for v in graph.value_info if v.name == "B")
        self.assertEqual([d.dim_value for d in vi2.type.tensor_type.shape.dim], [2, 3])
