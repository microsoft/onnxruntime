# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import tempfile
import unittest
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

import onnxruntime as ort
from onnxruntime.quantization.onnx_model import ONNXModel


def float_info(name):
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, [2])


def scalar_info(name, dtype):
    return helper.make_tensor_value_info(name, dtype, [])


def make_model(nodes, inputs, outputs, initializers=()):
    graph = helper.make_graph(nodes, "captured_inputs", inputs, outputs, list(initializers))
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)], ir_version=10)


def captured_branch(name, capture, negate=False):
    return helper.make_graph(
        [helper.make_node("Neg" if negate else "Identity", [capture], [name + "_out"])],
        name,
        [],
        [float_info(name + "_out")],
    )


def run_cpu(model, feeds):
    onnx.checker.check_model(model)
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(model.SerializeToString(), sess_options=options, providers=["CPUExecutionProvider"])
    return session.run(None, feeds)[0]


def captured_if_model(nested=False):
    bias = numpy_helper.from_array(np.array([2.0, -3.0], dtype=np.float32), "bias")
    producer = helper.make_node("Add", ["x", "bias"], ["captured"], name="producer")
    if nested:
        inner = helper.make_node(
            "If",
            ["inner_condition"],
            ["inner_out"],
            then_branch=captured_branch("deep_then", "captured"),
            else_branch=captured_branch("deep_else", "captured", True),
        )
        then_branch = helper.make_graph([inner], "nested_then", [], [float_info("inner_out")])
    else:
        then_branch = captured_branch("then", "captured")
    choice = helper.make_node(
        "If",
        ["condition"],
        ["output"],
        name="choice",
        then_branch=then_branch,
        else_branch=captured_branch("else", "captured", True),
    )
    inputs = [float_info("x"), scalar_info("condition", TensorProto.BOOL)]
    if nested:
        inputs.append(scalar_info("inner_condition", TensorProto.BOOL))
    return make_model([producer, choice], inputs, [float_info("output")], [bias])


class TestCapturedSubgraphDependencies(unittest.TestCase):
    def test_sort_preserves_a_valid_if_capture(self):
        for condition in (False, True):
            with self.subTest(condition=condition):
                model = captured_if_model()
                feeds = {"x": np.array([5.0, 7.0], dtype=np.float32), "condition": np.array(condition)}
                expected = np.array([7.0, 4.0], dtype=np.float32) * (1 if condition else -1)
                np.testing.assert_array_equal(run_cpu(model, feeds), expected)
                wrapped = ONNXModel(model)
                wrapped.topological_sort()
                self.assertEqual([node.name for node in model.graph.node], ["producer", "choice"])
                np.testing.assert_array_equal(run_cpu(model, feeds), expected)

    def test_sort_and_save_nested_captures(self):
        for outer, inner in ((True, True), (True, False), (False, True)):
            with self.subTest(outer=outer, inner=inner), tempfile.TemporaryDirectory() as directory:
                model = captured_if_model(nested=True)
                original_subgraph = model.graph.node[1].SerializeToString()
                nodes = list(reversed(model.graph.node))
                model.graph.ClearField("node")
                model.graph.node.extend(nodes)
                destination = Path(directory) / "sorted.onnx"
                ONNXModel(model).save_model_to_file(destination)
                saved = onnx.load(destination)
                self.assertEqual(saved.graph.node[-1].SerializeToString(), original_subgraph)
                expected = np.array([7.0, 4.0], dtype=np.float32) * (1 if outer and inner else -1)
                np.testing.assert_array_equal(
                    run_cpu(
                        saved,
                        {
                            "x": np.array([5.0, 7.0], dtype=np.float32),
                            "condition": np.array(outer),
                            "inner_condition": np.array(inner),
                        },
                    ),
                    expected,
                )

    def test_loop_body_capture_waits_for_its_producer(self):
        body = helper.make_graph(
            [
                helper.make_node("Identity", ["body_condition"], ["next_condition"]),
                helper.make_node("Add", ["state", "captured"], ["next_state"]),
            ],
            "body",
            [
                scalar_info("iteration", TensorProto.INT64),
                scalar_info("body_condition", TensorProto.BOOL),
                float_info("state"),
            ],
            [scalar_info("next_condition", TensorProto.BOOL), float_info("next_state")],
        )
        producer = helper.make_node("Add", ["x", "bias"], ["captured"], name="producer")
        loop = helper.make_node("Loop", ["trip_count", "loop_condition", "x"], ["output"], name="loop", body=body)
        model = make_model(
            [loop, producer],
            [float_info("x")],
            [float_info("output")],
            [
                numpy_helper.from_array(np.array([2.0, -3.0], dtype=np.float32), "bias"),
                numpy_helper.from_array(np.array(2, dtype=np.int64), "trip_count"),
                numpy_helper.from_array(np.array(True), "loop_condition"),
            ],
        )
        ONNXModel(model).topological_sort()
        self.assertEqual([node.name for node in model.graph.node], ["producer", "loop"])
        np.testing.assert_array_equal(
            run_cpu(model, {"x": np.array([5.0, 7.0], dtype=np.float32)}),
            np.array([19.0, 15.0], dtype=np.float32),
        )

    def test_cleanup_keeps_captured_constant_and_initializer(self):
        for condition in (False, True):
            with self.subTest(condition=condition):
                branch = helper.make_graph(
                    [helper.make_node("Add", ["captured_constant", "captured_weight"], ["branch_output"])],
                    "branch",
                    [],
                    [float_info("branch_output")],
                )
                constant = helper.make_node(
                    "Constant",
                    [],
                    ["captured_constant"],
                    name="constant",
                    value=numpy_helper.from_array(np.array([3.0, -4.0], dtype=np.float32)),
                )
                unused = helper.make_node("Constant", [], ["unused_constant"], name="unused", value_float=99.0)
                choice = helper.make_node(
                    "If", ["condition"], ["output"], name="choice", then_branch=branch, else_branch=branch
                )
                model = make_model(
                    [constant, unused, choice],
                    [scalar_info("condition", TensorProto.BOOL)],
                    [float_info("output")],
                    [
                        numpy_helper.from_array(np.array([7.0, 2.0], dtype=np.float32), "captured_weight"),
                        numpy_helper.from_array(np.array([99.0, 99.0], dtype=np.float32), "unused_weight"),
                    ],
                )
                wrapped = ONNXModel(model)
                wrapped.remove_unused_constant()
                self.assertEqual([node.name for node in model.graph.node], ["constant", "choice"])
                self.assertEqual([value.name for value in model.graph.initializer], ["captured_weight"])
                wrapped.topological_sort()
                np.testing.assert_array_equal(
                    run_cpu(model, {"condition": np.array(condition)}), np.array([10.0, -2.0], dtype=np.float32)
                )

    def test_subgraph_local_names_do_not_keep_unused_outer_initializers(self):
        local = numpy_helper.from_array(np.array([3.0, 4.0], dtype=np.float32), "local")
        branch = helper.make_graph(
            [helper.make_node("Identity", ["local"], ["branch_output"])],
            "branch",
            [],
            [float_info("branch_output")],
            [local],
        )
        choice = helper.make_node("If", ["condition"], ["output"], then_branch=branch, else_branch=branch)
        model = make_model(
            [choice],
            [scalar_info("condition", TensorProto.BOOL)],
            [float_info("output")],
            [numpy_helper.from_array(np.array([99.0, 99.0], dtype=np.float32), "local")],
        )
        wrapped = ONNXModel(model)
        wrapped.remove_unused_constant()
        self.assertEqual(len(model.graph.initializer), 0)
        wrapped.topological_sort()
        np.testing.assert_array_equal(
            run_cpu(model, {"condition": np.array(True)}), np.array([3.0, 4.0], dtype=np.float32)
        )

    def test_graphs_attribute_captures_all_branch_dependencies(self):
        first = helper.make_node("Identity", ["x"], ["first"], name="first")
        second = helper.make_node("Neg", ["x"], ["second"], name="second")
        container = helper.make_node(
            "GraphContainer",
            [],
            ["output"],
            domain="test",
            name="container",
            branches=[captured_branch("first_branch", "first"), captured_branch("second_branch", "second")],
        )
        model = make_model([container, second, first], [float_info("x")], [float_info("output")])
        ONNXModel(model).topological_sort()
        self.assertEqual(model.graph.node[-1].name, "container")
        self.assertEqual({node.name for node in model.graph.node[:-1]}, {"first", "second"})

    def test_duplicate_explicit_inputs_and_output_constants_remain_valid(self):
        constant = helper.make_node(
            "Constant", [], ["output_constant"], value=numpy_helper.from_array(np.array([9.0, 10.0], dtype=np.float32))
        )
        add = helper.make_node("Add", ["x", "x"], ["output"])
        model = make_model(
            [add, constant],
            [float_info("x")],
            [float_info("output"), float_info("output_constant")],
        )
        wrapped = ONNXModel(model)
        wrapped.remove_unused_constant()
        wrapped.topological_sort()
        np.testing.assert_array_equal(
            run_cpu(model, {"x": np.array([2.0, 3.0], dtype=np.float32)}), np.array([4.0, 6.0], dtype=np.float32)
        )
        self.assertEqual(sum(node.op_type == "Constant" for node in model.graph.node), 1)

    def test_missing_capture_and_capture_cycle_are_rejected(self):
        for cyclic in (False, True):
            with self.subTest(cyclic=cyclic):
                model = captured_if_model()
                producer = model.graph.node[0]
                if cyclic:
                    producer.input[0] = "output"
                else:
                    producer.output[0] = "different"
                with self.assertRaisesRegex(AssertionError, "not a DAG"):
                    ONNXModel(model).topological_sort()


if __name__ == "__main__":
    unittest.main()
