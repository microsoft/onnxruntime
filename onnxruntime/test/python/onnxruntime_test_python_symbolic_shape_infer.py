# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os

import numpy

# -*- coding: UTF-8 -*-
import onnx
from onnx import AttributeProto, GraphProto, TensorProto, helper, numpy_helper  # noqa: F401

if os.path.exists(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "python",
        "tools",
        "symbolic_shape_infer.py",
    )
):
    # Allow running this test script without installing onnxruntime package.
    import sys

    sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "python", "tools"))
    from symbolic_shape_infer import SymbolicShapeInference
else:
    from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference

import unittest
from pathlib import Path


def unique_element(lst):
    assert len(lst) == 1
    return lst[0]


skipped_models = ["SSD-MobilenetV1", "SSD-int8", "Inception-1-int8"]


class TestSymbolicShapeInference(unittest.TestCase):
    def test_symbolic_shape_infer(self):
        cwd = os.getcwd()
        test_model_dir = os.path.join(cwd, "..", "models")
        for filename in Path(test_model_dir).rglob("*.onnx"):
            if filename.name.startswith("."):
                continue  # skip some bad model files

            # https://github.com/onnx/models/issues/562
            if any(model_name in str(filename) for model_name in skipped_models):
                print(f"Skip symbolic shape inference on : {filename!s}")
                continue

            print("Running symbolic shape inference on : " + str(filename))
            SymbolicShapeInference.infer_shapes(
                in_mp=onnx.load(str(filename)),
                auto_merge=True,
                int_max=100000,
                guess_output_rank=True,
            )

    def test_mismatched_types(self):
        graph = helper.make_graph(
            [
                helper.make_node(
                    "If",
                    ["x"],
                    ["out"],
                    name="if_node",
                    then_branch=helper.make_graph(
                        [
                            helper.make_node(
                                "Constant",
                                [],
                                ["one_float"],
                                value=helper.make_tensor("one_float_value", TensorProto.FLOAT, [], [1]),
                            )
                        ],
                        "then",
                        [],
                        [helper.make_tensor_value_info("one_float", TensorProto.FLOAT, [])],
                    ),
                    else_branch=helper.make_graph(
                        [
                            helper.make_node(
                                "Constant",
                                [],
                                ["one_double"],
                                value=helper.make_tensor("one_double", TensorProto.DOUBLE, [], [1]),
                            )
                        ],
                        "else",
                        [],
                        [helper.make_tensor_value_info("one_double", TensorProto.DOUBLE, [])],
                    ),
                )
            ],
            "graph",
            [helper.make_tensor_value_info("x", TensorProto.BOOL, [])],
            [helper.make_tensor_value_info("out", TensorProto.FLOAT, [])],
        )
        model = helper.make_model(graph, producer_name="test_mismatched_types")

        with self.assertRaisesRegex(ValueError, r"if_node.*FLOAT.*DOUBLE"):
            SymbolicShapeInference.infer_shapes(model, auto_merge=True)


class TestSymbolicShapeInferenceForOperators(unittest.TestCase):
    def _check_shapes(self, graph, inferred_graph, vis):  # type: (GraphProto, GraphProto, List[ValueInfoProto]) -> None
        names_in_vis = {x.name for x in vis}
        vis = list(x for x in graph.value_info if x.name not in names_in_vis) + vis
        inferred_vis = list(inferred_graph.value_info)
        vis = list(sorted(vis, key=lambda x: x.name))
        inferred_vis = list(sorted(inferred_vis, key=lambda x: x.name))
        if vis == inferred_vis:
            return
        # otherwise some custom logic to give a nicer diff
        vis_names = {x.name for x in vis}
        inferred_vis_names = {x.name for x in inferred_vis}
        assert vis_names == inferred_vis_names, (vis_names, inferred_vis_names)
        for vi, inferred_vi in zip(vis, inferred_vis):
            assert vi == inferred_vi, f"\n{vi}\n{inferred_vi}\n"
        raise AssertionError()

    def test_unsqueeze_opset_11(self):
        graph = helper.make_graph(
            [
                helper.make_node("Unsqueeze", ["input"], ["temp"], axes=[0]),
                helper.make_node("Identity", ["temp"], ["output"]),
            ],
            "Unsqueeze_Test",
            [
                helper.make_tensor_value_info("input", TensorProto.FLOAT, ["b", "s"]),
            ],
            [
                helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, "b", "s"]),
            ],
        )
        model = helper.make_model(graph, producer_name="Unsqueeze_Test_Model")
        model.opset_import[0].version = 11

        inferred = SymbolicShapeInference.infer_shapes(model, auto_merge=True)
        expected_shapes = [
            helper.make_tensor_value_info("temp", TensorProto.FLOAT, [1, "b", "s"]),
            helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, "b", "s"]),
        ]
        self._check_shapes(graph, inferred.graph, expected_shapes)

    def test_unsqueeze_opset_13(self):
        graph = helper.make_graph(
            [
                helper.make_node("Unsqueeze", ["input", "axes"], ["temp"]),
                helper.make_node("Identity", ["temp"], ["output"]),
            ],
            "Unsqueeze_Test",
            [
                helper.make_tensor_value_info("input", TensorProto.FLOAT, ["b", "s"]),
            ],
            [
                helper.make_tensor_value_info("output", TensorProto.FLOAT, ["b", "s", 1]),
            ],
            [
                helper.make_tensor("axes", TensorProto.INT64, [1], [-1]),
            ],
        )
        model = helper.make_model(graph, producer_name="Unsqueeze_Test_Model")
        model.opset_import[0].version = 13

        inferred = SymbolicShapeInference.infer_shapes(model, auto_merge=True)
        expected_shapes = [
            helper.make_tensor_value_info("temp", TensorProto.FLOAT, ["b", "s", 1]),
            helper.make_tensor_value_info("output", TensorProto.FLOAT, ["b", "s", 1]),
        ]
        self._check_shapes(graph, inferred.graph, expected_shapes)

    def test_gather_indices(self):
        graph = helper.make_graph(
            [
                helper.make_node(
                    "Constant",
                    [],
                    ["data"],
                    "constant",
                    value=helper.make_tensor("input", TensorProto.FLOAT, [5], [0.0, 1.0, 2.0, 3.0, 4.0]),
                ),
                helper.make_node("Gather", ["data", "indices"], ["output"], axis=0),
            ],
            "Gather_Test",
            [
                helper.make_tensor_value_info("indices", TensorProto.INT64, ["b"]),
            ],
            [
                helper.make_tensor_value_info("output", TensorProto.FLOAT, ["b"]),
            ],
        )
        model = helper.make_model(graph, producer_name="Gather_Test_Model")
        model.opset_import[0].version = 13

        inferred = SymbolicShapeInference.infer_shapes(model, auto_merge=True)
        expected_shapes = [
            helper.make_tensor_value_info("data", TensorProto.FLOAT, [5]),
            helper.make_tensor_value_info("output", TensorProto.FLOAT, ["b"]),
        ]
        self._check_shapes(graph, inferred.graph, expected_shapes)

    def test_embed_layer_norm(self):
        hidden_size = 32
        initializers = [
            helper.make_tensor(
                "word_embedding",
                TensorProto.FLOAT,
                [100, hidden_size],
                [1.0] * (100 * hidden_size),
            ),
            helper.make_tensor(
                "position_embedding",
                TensorProto.FLOAT,
                [20, hidden_size],
                [1.0] * (20 * hidden_size),
            ),
            helper.make_tensor(
                "segment_embedding",
                TensorProto.FLOAT,
                [2, hidden_size],
                [1.0] * (2 * hidden_size),
            ),
            helper.make_tensor("gamma", TensorProto.FLOAT, [hidden_size], [1.0] * hidden_size),
            helper.make_tensor("beta", TensorProto.FLOAT, [hidden_size], [1.0] * hidden_size),
        ]

        nodes = [
            helper.make_node(
                "EmbedLayerNormalization",
                inputs=[
                    "input_ids",
                    "segment_ids",
                    "word_embedding",
                    "position_embedding",
                    "segment_embedding",
                    "gamma",
                    "beta",
                ],
                outputs=["output", "mask_index"],
                domain="com.microsoft",
            ),
        ]

        inputs = [
            helper.make_tensor_value_info("input_ids", TensorProto.FLOAT, ["b", "s"]),
            helper.make_tensor_value_info("segment_ids", TensorProto.FLOAT, ["b", "s"]),
        ]

        outputs = [
            helper.make_tensor_value_info("output", TensorProto.FLOAT, None),
            helper.make_tensor_value_info("mask_index", TensorProto.INT32, None),
        ]

        graph = helper.make_graph(nodes, "Unsqueeze_Test", inputs, outputs, initializers)
        model = helper.make_model(graph)

        inferred = SymbolicShapeInference.infer_shapes(model, auto_merge=True)
        expected_shapes = [
            helper.make_tensor_value_info("output", TensorProto.FLOAT, ["b", "s", hidden_size]),
            helper.make_tensor_value_info("mask_index", TensorProto.INT32, ["b"]),
        ]
        self._check_shapes(graph, inferred.graph, expected_shapes)

    def test_softmax_cross_entropy_loss(self):
        hidden_size = 1024

        nodes = [
            helper.make_node("SoftmaxCrossEntropyLoss", inputs=["logits", "labels"], outputs=["loss"]),
        ]

        inputs = [
            helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["b", "s", hidden_size]),
            helper.make_tensor_value_info("labels", TensorProto.INT32, ["b", "s"]),
        ]

        outputs = [
            helper.make_tensor_value_info("loss", TensorProto.FLOAT, None),
        ]

        graph = helper.make_graph(nodes, "SoftmaxCrossEntropyLoss_Test", inputs, outputs, [])
        model = helper.make_model(graph)

        inferred = SymbolicShapeInference.infer_shapes(model, auto_merge=True)
        expected_shapes = [helper.make_tensor_value_info("loss", TensorProto.FLOAT, [])]
        self._check_shapes(graph, inferred.graph, expected_shapes)

    def _test_einsum_one_input_impl(self, input_0_shape, output_0_shape, eqn):
        nodes = [
            helper.make_node("Einsum", ["input_0"], ["output_0"], "einsum_0", equation=eqn),
        ]
        inputs = [
            helper.make_tensor_value_info("input_0", TensorProto.FLOAT, input_0_shape),
        ]
        outputs = [
            helper.make_tensor_value_info("output_0", TensorProto.FLOAT, None),
        ]
        graph = helper.make_graph(nodes, "Einsum_Test", inputs, outputs, [])
        model = helper.make_model(graph)

        inferred = SymbolicShapeInference.infer_shapes(model, auto_merge=True)
        expected_shapes = [helper.make_tensor_value_info("output_0", TensorProto.FLOAT, output_0_shape)]
        self._check_shapes(graph, inferred.graph, expected_shapes)

    def _test_einsum_two_inputs_impl(self, input_0_shape, input_1_shape, output_0_shape, eqn):
        nodes = [
            helper.make_node("Einsum", ["input_0", "input_1"], ["output_0"], "einsum_0", equation=eqn),
        ]
        inputs = [
            helper.make_tensor_value_info("input_0", TensorProto.FLOAT, input_0_shape),
            helper.make_tensor_value_info("input_1", TensorProto.FLOAT, input_1_shape),
        ]
        outputs = [
            helper.make_tensor_value_info("output_0", TensorProto.FLOAT, None),
        ]
        graph = helper.make_graph(nodes, "Einsum_Test", inputs, outputs, [])
        model = helper.make_model(graph)

        inferred = SymbolicShapeInference.infer_shapes(model, auto_merge=True)
        expected_shapes = [helper.make_tensor_value_info("output_0", TensorProto.FLOAT, output_0_shape)]
        self._check_shapes(graph, inferred.graph, expected_shapes)

    def test_einsum_matmul(self):
        self._test_einsum_two_inputs_impl([1, "b", 8], [2, 12, "n"], [1, "b", 12, "n"], "abc, cde -> abde")

    def test_einsum_batch_matmul(self):
        self._test_einsum_two_inputs_impl([5, 2, 3], [5, 3, 4], [5, 2, 4], "bij, bjk -> bik")

    def test_einsum_inner_prod(self):
        self._test_einsum_two_inputs_impl([5], [5], [], "i, i")

    def test_einsum_batch_diagonal(self):
        self._test_einsum_one_input_impl([3, 5, 5], [3, 5], "...ii ->...i")

    def test_einsum_sum(self):
        self._test_einsum_one_input_impl(["a", "b"], ["a"], "ij -> i")

    def test_einsum_transpose(self):
        self._test_einsum_one_input_impl(["a", "b"], ["b", "a"], "ij -> ji")

    def test_mul_precision(self):
        graph_input = onnx.helper.make_tensor_value_info("input", TensorProto.FLOAT, [1024])
        graph_output = onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, None)

        # initializers
        value = numpy.array([0.5], dtype=numpy.float32)
        constant = numpy_helper.from_array(value, name="constant")

        nodes = [
            # Get the shape of the input tensor: `input_tensor_shape = [1024]`.
            onnx.helper.make_node("Shape", ["input"], ["input_shape"]),
            # mul(1024, 0.5) => 512
            onnx.helper.make_node("Mul", ["input_shape", "constant"], ["output_shape"]),
            # Resize input
            onnx.helper.make_node(
                "Resize", inputs=["input", "", "", "output_shape"], outputs=["output"], mode="nearest"
            ),
        ]

        graph_def = onnx.helper.make_graph(nodes, "TestMulPrecision", [graph_input], [graph_output], [constant])
        model = SymbolicShapeInference.infer_shapes(onnx.helper.make_model(graph_def))
        output_dims = unique_element(model.graph.output).type.tensor_type.shape.dim
        self.assertEqual(len(output_dims), 1)
        self.assertEqual(output_dims[0].dim_value, 512)

    def test_div_precision(self):
        graph_input = onnx.helper.make_tensor_value_info("input", TensorProto.FLOAT, [768])
        graph_output = onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, None)

        # initializers
        value = numpy.array([1.5], dtype=numpy.float32)
        constant = numpy_helper.from_array(value, name="constant")

        nodes = [
            # Get the shape of the input tensor: `input_tensor_shape = [768]`.
            onnx.helper.make_node("Shape", ["input"], ["input_shape"]),
            # div(768, 1.5) => 512
            onnx.helper.make_node("Div", ["input_shape", "constant"], ["output_shape"]),
            # Resize input
            onnx.helper.make_node(
                "Resize", inputs=["input", "", "", "output_shape"], outputs=["output"], mode="nearest"
            ),
        ]

        graph_def = onnx.helper.make_graph(nodes, "TestDivPrecision", [graph_input], [graph_output], [constant])
        model = SymbolicShapeInference.infer_shapes(onnx.helper.make_model(graph_def))
        output_dims = unique_element(model.graph.output).type.tensor_type.shape.dim
        self.assertEqual(len(output_dims), 1)
        self.assertEqual(output_dims[0].dim_value, 512)

    def test_quantize_linear(self):
        """
        Test ONNX QuantizeLinear op.
        Check that the output shape is propagated from the first input and that the output data
        type comes from the zero-point input.
        """
        initializers = [
            helper.make_tensor(
                "scale",
                TensorProto.FLOAT,
                [],
                [1.0],
            ),
            helper.make_tensor(
                "zero_point",
                TensorProto.INT8,
                [],
                [16],
            ),
        ]

        nodes = [
            helper.make_node(
                "QuantizeLinear",
                inputs=[
                    "input_f32",
                    "scale",
                    "zero_point",
                ],
                outputs=["output_s8"],
            ),
        ]

        inputs = [
            helper.make_tensor_value_info("input_f32", TensorProto.FLOAT, ["b", 2, 3, 4]),
        ]

        outputs = [
            helper.make_tensor_value_info("output_s8", TensorProto.UNDEFINED, None),
        ]

        graph = helper.make_graph(nodes, "QuantizeLinear_Test", inputs, outputs, initializers)
        model = helper.make_model(graph)

        inferred = SymbolicShapeInference.infer_shapes(model, auto_merge=True)

        expected_shapes = [
            helper.make_tensor_value_info("output_s8", TensorProto.INT8, ["b", 2, 3, 4]),
        ]
        self._check_shapes(graph, inferred.graph, expected_shapes)

    def test_quantize_linear_ms_domain(self):
        """
        Test QuantizeLinear op ('com.microsoft' domain).
        Check that the output shape is propagated from the first input and that the output data
        type comes from the zero-point input.
        """
        initializers = [
            helper.make_tensor(
                "scale",
                TensorProto.FLOAT,
                [],
                [1.0],
            ),
            helper.make_tensor(
                "zero_point",
                TensorProto.UINT16,
                [],
                [16],
            ),
        ]

        nodes = [
            helper.make_node(
                "QuantizeLinear",
                inputs=[
                    "input_f32",
                    "scale",
                    "zero_point",
                ],
                outputs=["output_u16"],
                domain="com.microsoft",
            ),
        ]

        inputs = [
            helper.make_tensor_value_info("input_f32", TensorProto.FLOAT, ["b", 2, 3, 4]),
        ]

        outputs = [
            helper.make_tensor_value_info("output_u16", TensorProto.UNDEFINED, None),
        ]

        graph = helper.make_graph(nodes, "QuantizeLinear_MSDomain_Test", inputs, outputs, initializers)
        model = helper.make_model(graph)

        inferred = SymbolicShapeInference.infer_shapes(model, auto_merge=True)

        expected_shapes = [
            helper.make_tensor_value_info("output_u16", TensorProto.UINT16, ["b", 2, 3, 4]),
        ]
        self._check_shapes(graph, inferred.graph, expected_shapes)

    def test_quantize_linear_no_zp_input(self):
        """
        Test QuantizeLinear op ('com.microsoft' domain).
        Check that the output shape is propagated from the first input.
        The zero-point input is missing, so the output data type should default to uint8.
        """
        initializers = [
            helper.make_tensor(
                "scale",
                TensorProto.FLOAT,
                [],
                [1.0],
            ),
        ]

        nodes = [
            helper.make_node(
                "QuantizeLinear",
                inputs=[
                    "input_f32",
                    "scale",
                ],
                outputs=["output_u8"],
                domain="com.microsoft",
            ),
        ]

        inputs = [
            helper.make_tensor_value_info("input_f32", TensorProto.FLOAT, ["b", 2, 3, 4]),
        ]

        outputs = [
            helper.make_tensor_value_info("output_u8", TensorProto.UNDEFINED, None),
        ]

        graph = helper.make_graph(nodes, "QuantizeLinear_NoZP_Test", inputs, outputs, initializers)
        model = helper.make_model(graph)

        inferred = SymbolicShapeInference.infer_shapes(model, auto_merge=True)

        # Check that the output shape is propagated from the first input and that the
        # output data type comes from the zero-point input.
        expected_shapes = [
            helper.make_tensor_value_info("output_u8", TensorProto.UINT8, ["b", 2, 3, 4]),
        ]
        self._check_shapes(graph, inferred.graph, expected_shapes)

    def test_dequantize_linear_ms_domain(self):
        """
        Test DequantizeLinear operator ('com.microsoft' domain).
        Check that the output shape is propagated from the first input and that the output data
        type comes from the scale input.
        """
        initializers = [
            helper.make_tensor(
                "scale",
                TensorProto.FLOAT,
                [],
                [1.0],
            ),
            helper.make_tensor(
                "zero_point",
                TensorProto.UINT16,
                [],
                [16],
            ),
        ]

        nodes = [
            helper.make_node(
                "DequantizeLinear",
                inputs=[
                    "input_u16",
                    "scale",
                    "zero_point",
                ],
                outputs=["output_f32"],
                domain="com.microsoft",
            ),
        ]

        inputs = [
            helper.make_tensor_value_info("input_u16", TensorProto.UINT16, ["b", 2, 3, 4]),
        ]

        outputs = [
            helper.make_tensor_value_info("output_f32", TensorProto.UNDEFINED, None),
        ]

        graph = helper.make_graph(nodes, "DequantizeLinear_MSDomain_Test", inputs, outputs, initializers)
        model = helper.make_model(graph)

        inferred = SymbolicShapeInference.infer_shapes(model, auto_merge=True)

        expected_shapes = [
            helper.make_tensor_value_info("output_f32", TensorProto.FLOAT, ["b", 2, 3, 4]),
        ]
        self._check_shapes(graph, inferred.graph, expected_shapes)


class TestSymbolicShapeInferenceForSlice(unittest.TestCase):
    def check_slice_of_concat(self, input_dims, start, end, step, expected_output_dim):
        _dimstrmap = {dim: f"dim{i}" for i, dim in enumerate(input_dims)}

        def dimstrmap(dim):
            return _dimstrmap.get(dim, dim)

        def get_initializer(name):
            valuemap = {"zero": 0, "one": 1, "two": 2, "ten": 10, "intmax": 2**32}
            value = -valuemap[name[4:]] if name.startswith("neg_") else valuemap[name]
            return onnx.helper.make_tensor(name, TensorProto.INT64, [1], [value])

        initializers = [
            get_initializer(name)
            for name in [
                "zero",
                "one",
                "two",
                "ten",
                "intmax",
                "neg_intmax",
                "neg_one",
                "neg_ten",
            ]
        ]
        inputs = []
        nodes = []
        for i, dim in enumerate(input_dims):
            inputs.append(onnx.helper.make_tensor_value_info(f"t{i}", TensorProto.FLOAT, ["B", dim]))
            nodes.extend(
                [
                    onnx.helper.make_node("Shape", [f"t{i}"], [f"shape{i}"]),
                    onnx.helper.make_node("Slice", [f"shape{i}", "one", "two", "zero", "one"], [f"dim{i}"]),
                    onnx.helper.make_node("Neg", [f"dim{i}"], [f"neg_dim{i}"]),
                ]
            )

        def make_concat_dims(concat_name, dims):
            dims = [f"neg_{dimstrmap(dim[1:])}" if dim.startswith("-") else dimstrmap(dim) for dim in dims]
            return onnx.helper.make_node("Concat", dims, [concat_name], axis=0)

        nodes.extend(
            [
                onnx.helper.make_node("Concat", [inp.name for inp in inputs], ["concat"], axis=1),
                make_concat_dims("starts", ["zero", start]),
                make_concat_dims("ends", ["intmax", end]),
                make_concat_dims("axes", ["zero", "one"]),
                make_concat_dims("steps", ["one", step]),
                onnx.helper.make_node("Slice", ["concat", "starts", "ends", "axes", "steps"], ["output"]),
            ]
        )
        output = onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, ["d1", "d2"])
        graph_def = onnx.helper.make_graph(nodes, "graph", inputs, [output], initializer=initializers)
        model = SymbolicShapeInference.infer_shapes(onnx.helper.make_model(graph_def))
        output = unique_element(model.graph.output)
        shape = [d.dim_param if d.dim_param else d.dim_value for d in output.type.tensor_type.shape.dim]
        self.assertEqual(shape, ["B", expected_output_dim])

    def test_numeric_negative_indices_forward(self):
        self.check_slice_of_concat(["M"], "-ten", "-one", "one", 9)

    def test_numeric_negative_indices_backward(self):
        self.check_slice_of_concat(["M"], "-one", "-ten", "-one", 9)

    def test_symbolic_end_index(self):
        self.check_slice_of_concat(["M", "N"], "zero", "M", "one", "M")

    def test_symbolic_negative_start_index(self):
        self.check_slice_of_concat(["M", "N"], "-N", "intmax", "one", "N")

    def test_non_unit_step(self):
        self.check_slice_of_concat(["N", "N"], "zero", "intmax", "two", "N")

    def test_symbolic_step(self):
        self.check_slice_of_concat(["N", "N"], "zero", "intmax", "N", "floor(-1/N) + 3")

    def test_symbolic_negative_step(self):
        self.check_slice_of_concat(["N", "N"], "-one", "-intmax", "-N", "floor(-1/N) + 3")

    def test_flip(self):
        self.check_slice_of_concat(["N"], "-one", "-intmax", "-one", "N")

    def test_flip_of_concat(self):
        self.check_slice_of_concat(["N", "N", "N"], "-one", "-intmax", "-one", "3*N")

    def test_slice_of_min(self):
        graph_input = onnx.helper.make_tensor_value_info("input", TensorProto.FLOAT, ["N"])
        graph_output = onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, None)
        const_size = 42
        half_const_size = const_size // 2
        initializers = [
            onnx.helper.make_tensor("const_tensor", TensorProto.FLOAT, [const_size], [42.0] * const_size),
            onnx.helper.make_tensor("half_const_size", TensorProto.INT64, [], [half_const_size]),
            onnx.helper.make_tensor("zeros", TensorProto.INT64, [1], [0]),
            onnx.helper.make_tensor("ones", TensorProto.INT64, [1], [1]),
        ]

        nodes = [
            # Get the shape of the input tensor: `input_tensor_shape = [N]`.
            onnx.helper.make_node("Shape", ["input"], ["input_tensor_shape"]),
            # The starts of the const tensor slice: `starts = [21 - N + 1]`.
            onnx.helper.make_node("Sub", ["half_const_size", "input_tensor_shape"], ["starts_aux"]),
            onnx.helper.make_node("Add", ["ones", "starts_aux"], ["starts"]),
            # The ends of the const tensor slice: `ends = [21 + N]`.
            onnx.helper.make_node("Add", ["half_const_size", "input_tensor_shape"], ["ends"]),
            # Slice the const tensor: `slice_out = const_tensor[starts:ends]`.
            onnx.helper.make_node("Slice", ["const_tensor", "starts", "ends", "zeros", "ones"], ["slice_out"]),
            # Crop the const tensor slice using the shape of the input tensor: `slice_out_cropped = slice_out[0:input_tensor_shape]`.
            onnx.helper.make_node(
                "Slice", ["slice_out", "zeros", "input_tensor_shape", "zeros", "ones"], ["slice_out_cropped"]
            ),
            # Add the const tensor slice to the input tensor: `output = input + slice_out_cropped`.
            onnx.helper.make_node("Add", ["slice_out_cropped", "input"], ["output"]),
        ]

        graph_def = onnx.helper.make_graph(nodes, "SliceOfMin", [graph_input], [graph_output], initializer=initializers)
        model = SymbolicShapeInference.infer_shapes(onnx.helper.make_model(graph_def))
        output_dims = unique_element(model.graph.output).type.tensor_type.shape.dim
        self.assertEqual(len(output_dims), 1)
        self.assertEqual(output_dims[0].dim_param, "N")


if __name__ == "__main__":
    unittest.main()
