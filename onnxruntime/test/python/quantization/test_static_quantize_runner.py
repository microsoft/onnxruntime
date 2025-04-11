# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import contextlib
import json
import os
import tempfile
import unittest
from unittest import mock

import numpy as np
import onnx

from onnxruntime.quantization import static_quantize_runner


class StaticQuantizeRunnerTestBase:
    def setUp(self):
        self._tmp_model_dir = tempfile.TemporaryDirectory(prefix="ort.static_quant_runner_")
        self._tmp_dir_path = self._tmp_model_dir.name
        self.rand_seed = 0
        np.random.seed(self.rand_seed)

    def run_static_quantize_runner(self, script_args: list, override_dict=None) -> onnx.ModelProto:
        model = self.onnx_model()
        test_data_sets = self.get_test_data_set(model)
        for idx, test_data_set in enumerate(test_data_sets):
            test_data_set_dir = os.path.join(self._tmp_dir_path, f"test_data_set_{idx}")
            os.makedirs(test_data_set_dir, exist_ok=True)
            for data_idx, test_data in enumerate(test_data_set):
                data_path = os.path.join(test_data_set_dir, f"input_{data_idx}.pb")
                with open(data_path, "wb") as f:
                    tensor_proto = onnx.numpy_helper.from_array(test_data)
                    f.write(tensor_proto.SerializeToString())

        in_model_path = os.path.join(self._tmp_dir_path, "model.onnx")
        onnx.save_model(model, in_model_path)  # Save input test model to disk

        out_model_path = os.path.join(self._tmp_dir_path, "model_quant.onnx")

        # Call script's main() with custom command-line args.
        script = ["static_quantize_runner.py", "-i", in_model_path, "-o", out_model_path]
        if override_dict:
            quant_override_path = os.path.join(self._tmp_dir_path, "quant_override.json")
            with open(quant_override_path, "w") as f:
                json.dump(override_dict, f)
            script += ["--tensor_quant_overrides", quant_override_path]

        with mock.patch("sys.argv", script + script_args):
            static_quantize_runner.main()

        # check that output qdq model was generated
        self.assertTrue(os.path.exists(out_model_path))
        return onnx.load_model(out_model_path)

    def onnx_model(self) -> onnx.ModelProto:
        pass

    def get_test_data_set(self, model):
        inp_shapes = [[d.dim_value for d in inp.type.tensor_type.shape.dim] for inp in model.graph.input]
        inp_dtypes = [onnx.helper.tensor_dtype_to_np_dtype(inp.type.tensor_type.elem_type) for inp in model.graph.input]
        test_data_sets = [
            [
                np.random.rand(*inp_shape).astype(inp_dtype)
                for inp_shape, inp_dtype in zip(inp_shapes, inp_dtypes, strict=False)
            ]
        ]
        return test_data_sets

    def test_activation_weight_type(self):
        script_args_set = [
            (["--activation_type", "quint8", "--weight_type", "quint8"], True),
            (["--activation_type", "quint8", "--weight_type", "qint8"], True),
            (["--activation_type", "qint8", "--weight_type", "quint8"], False),
            (["--activation_type", "qint8", "--weight_type", "qint8"], True),
        ]
        for script_args, success in script_args_set:
            with (
                self.subTest(script_args=script_args),
                contextlib.nullcontext() if success else self.assertRaises(Exception),
            ):
                self.run_static_quantize_runner(script_args)

    def test_quant_format(self):
        script_args_set = [(["--quant_format", "qdq"], True), (["--quant_format", "qoperator"], True)]
        for script_args, success in script_args_set:
            with (
                self.subTest(script_args=script_args),
                contextlib.nullcontext() if success else self.assertRaises(Exception),
            ):
                self.run_static_quantize_runner(script_args)

    def test_calibration_method(self):
        script_args_set = [
            (["--calibration_method", "minmax"], True),
            (["--calibration_method", "entropy"], True),
            (["--calibration_method", "percentile"], True),
            (["--calibration_method", "distribution"], True),
        ]
        for script_args, success in script_args_set:
            with (
                self.subTest(script_args=script_args),
                contextlib.nullcontext() if success else self.assertRaises(Exception),
            ):
                self.run_static_quantize_runner(script_args)

    def test_calib_tensor_range_symmetric(self):
        # TODO: Check whether the calibrated tensor value is symmetric
        script_args_set = [
            (["--activation_type", "quint8", "--calib_tensor_range_symmetric"], True),
            (["--activation_type", "qint8", "--calib_tensor_range_symmetric"], True),
        ]
        for script_args, success in script_args_set:
            with (
                self.subTest(script_args=script_args, success=success),
                contextlib.nullcontext() if success else self.assertRaises(Exception),
            ):
                self.run_static_quantize_runner(script_args)

    def test_calib_moving_average(self):
        script_args_set = [
            (["--calib_moving_average"], True),
        ]
        for script_args, success in script_args_set:
            with (
                self.subTest(script_args=script_args, success=success),
                contextlib.nullcontext() if success else self.assertRaises(Exception),
            ):
                self.run_static_quantize_runner(script_args)

    def test_minimum_real_range(self):
        # TODO: Check whether the (rmin-rmax) follows the minimum range
        script_args_set = [
            (["--minimum_real_range", "0.001"], True),
        ]
        for script_args, success in script_args_set:
            with (
                self.subTest(script_args=script_args, success=success),
                contextlib.nullcontext() if success else self.assertRaises(Exception),
            ):
                self.run_static_quantize_runner(script_args)

    def test_use_qdq_contrib_ops(self):
        script_args_set = [
            (["--use_qdq_contrib_ops"], True),
        ]
        for script_args, success in script_args_set:
            with (
                self.subTest(script_args=script_args, success=success),
                contextlib.nullcontext() if success else self.assertRaises(Exception),
            ):
                quant_model = self.run_static_quantize_runner(script_args)
                for node in quant_model.graph.node:
                    if node.op_type in ["QuantizeLinear", "DeQuantizeLinear"]:
                        assert node.domain == "com.microsoft"

    def test_qdq_disable_weight_adjust_for_int32_bias(self):
        # TODO: Check whether the weight's scale is adjusted
        script_args_set = [
            (["--qdq_disable_weight_adjust_for_int32_bias"], True),
        ]
        for script_args, success in script_args_set:
            with (
                self.subTest(script_args=script_args, success=success),
                contextlib.nullcontext() if success else self.assertRaises(Exception),
            ):
                self.run_static_quantize_runner(script_args)


class TestAdd(StaticQuantizeRunnerTestBase, unittest.TestCase):
    def onnx_model(self, inp_shape=(4, 3, 32, 32)) -> onnx.ModelProto:
        graph_inputs = [
            onnx.helper.make_tensor_value_info("input_0", onnx.TensorProto.FLOAT, inp_shape),
            onnx.helper.make_tensor_value_info("input_1", onnx.TensorProto.FLOAT, inp_shape),
        ]
        graph_outputs = [onnx.helper.make_tensor_value_info("output_0", onnx.TensorProto.FLOAT, inp_shape)]
        initializers = []

        add_node = onnx.helper.make_node("Add", ["input_0", "input_1"], ["output_0"], name="Add0")
        graph = onnx.helper.make_graph(
            [add_node],
            "AddGraph",
            graph_inputs,
            graph_outputs,
            initializer=initializers,
        )
        opset_imports = [onnx.helper.make_opsetid("", 21)]
        model = onnx.helper.make_model(graph, opset_imports=opset_imports)
        model = onnx.shape_inference.infer_shapes(model)
        onnx.checker.check_model(model, True)
        return model

    def test_tensor_quant_overrides(self):
        override_dict = {"input_0": [{"scale": 0.005, "zero_point": 2}]}
        quant_model = self.run_static_quantize_runner(script_args=[], override_dict=override_dict)
        assert quant_model.graph.initializer[0].name == "input_0_zero_point"
        assert np.allclose(quant_model.graph.initializer[0].int32_data, 2)
        assert quant_model.graph.initializer[1].name == "input_0_scale"
        assert np.allclose(quant_model.graph.initializer[1].float_data, 0.005)


class TestMatMulConstB(StaticQuantizeRunnerTestBase, unittest.TestCase):
    def onnx_model(self, inp_shape=(4, 3, 5, 7), weight_shape=(4, 3, 7, 9), out_shape=(4, 3, 5, 9)) -> onnx.ModelProto:
        graph_inputs = [
            onnx.helper.make_tensor_value_info("input_0", onnx.TensorProto.FLOAT, inp_shape),
        ]
        graph_outputs = [onnx.helper.make_tensor_value_info("output_0", onnx.TensorProto.FLOAT, out_shape)]
        initializers = [
            onnx.helper.make_tensor(
                "weight_0", onnx.TensorProto.FLOAT, dims=weight_shape, vals=np.random.rand(*weight_shape)
            )
        ]
        matmul_input_names = ["input_0", "weight_0"]

        matmul_node = onnx.helper.make_node("MatMul", matmul_input_names, ["output_0"], name="MatMul0")
        graph = onnx.helper.make_graph(
            [matmul_node],
            "MatMulGraph",
            graph_inputs,
            graph_outputs,
            initializer=initializers,
        )
        opset_imports = [onnx.helper.make_opsetid("", 21)]
        model = onnx.helper.make_model(graph, opset_imports=opset_imports)
        model = onnx.shape_inference.infer_shapes(model)
        onnx.checker.check_model(model, True)
        return model

    def test_per_channel_quant(self):
        # TODO: Fix the str and int issue on the axis and enable the following subtests
        script_args_set = [
            (["--per_channel"], 3),
            # (["--per_channel", "--op_per_channel_axis", "MatMul", "0"], 4),
            # (["--per_channel", "--op_per_channel_axis", "MatMul", "1"], 3),
            # (["--per_channel", "--op_per_channel_axis", "MatMul", "2"], 7),
            # (["--per_channel", "--op_per_channel_axis", "MatMul", "3"], 9),
        ]
        for script_args, n_channels in script_args_set:
            with self.subTest(script_args=script_args):
                quant_model = self.run_static_quantize_runner(script_args)
                assert quant_model.graph.initializer[0].name == "input_0_zero_point"
                assert len(quant_model.graph.initializer[0].dims) == 0
                assert quant_model.graph.initializer[1].name == "input_0_scale"
                assert len(quant_model.graph.initializer[1].dims) == 0
                assert quant_model.graph.initializer[2].name == "weight_0_zero_point"
                assert quant_model.graph.initializer[2].dims[0] == n_channels
                assert quant_model.graph.initializer[3].name == "weight_0_scale"
                assert quant_model.graph.initializer[3].dims[0] == n_channels

    def test_add_qdq_pair_to_weight(self):
        script_args_set = [
            (["--add_qdq_pair_to_weight"], 7),
            ([], 6),
        ]
        for script_args, expect_num_nodes in script_args_set:
            with self.subTest(script_args=script_args, expect_num_nodes=expect_num_nodes):
                quant_model = self.run_static_quantize_runner(script_args)
                assert len(quant_model.graph.node) == expect_num_nodes

    def test_exlude_output_quantization(self):
        script_args_set = [
            (["--op_types_to_exclude_output_quantization", "MatMul"], 4),
            ([], 6),
        ]
        for script_args, expect_num_nodes in script_args_set:
            with self.subTest(script_args=script_args, expect_num_nodes=expect_num_nodes):
                quant_model = self.run_static_quantize_runner(script_args)
                assert len(quant_model.graph.node) == expect_num_nodes


class TestMatMul(StaticQuantizeRunnerTestBase, unittest.TestCase):
    def onnx_model(self, inp_shape1=(4, 3, 5, 7), inp_shape2=(4, 3, 7, 9), out_shape=(4, 3, 5, 9)) -> onnx.ModelProto:
        graph_inputs = [
            onnx.helper.make_tensor_value_info("input_0", onnx.TensorProto.FLOAT, inp_shape1),
            onnx.helper.make_tensor_value_info("input_1", onnx.TensorProto.FLOAT, inp_shape2),
        ]
        graph_outputs = [onnx.helper.make_tensor_value_info("output_0", onnx.TensorProto.FLOAT, out_shape)]
        matmul_input_names = ["input_0", "input_1"]

        matmul_node = onnx.helper.make_node("MatMul", matmul_input_names, ["output_0"], name="MatMul0")
        graph = onnx.helper.make_graph([matmul_node], "MatMulGraph", graph_inputs, graph_outputs)
        opset_imports = [onnx.helper.make_opsetid("", 21)]
        model = onnx.helper.make_model(graph, opset_imports=opset_imports)
        model = onnx.shape_inference.infer_shapes(model)
        onnx.checker.check_model(model, True)
        return model

    def test_matmul_const_b_only(self):
        # TODO: Support --matmul_const_b_only on qdq quant_format and refine the testcase
        script_args_set = [
            (["--quant_format", "qoperator", "--matmul_const_b_only"], 1),
            (["--quant_format", "qoperator"], 4),
        ]
        for script_args, expect_num_nodes in script_args_set:
            with self.subTest(script_args=script_args, expect_num_nodes=expect_num_nodes):
                quant_model = self.run_static_quantize_runner(script_args)
                assert len(quant_model.graph.node) == expect_num_nodes


class TestAddSideBySide(StaticQuantizeRunnerTestBase, unittest.TestCase):
    def onnx_model(self, inp_shape=(4, 3, 32, 32)) -> onnx.ModelProto:
        graph_inputs = [
            onnx.helper.make_tensor_value_info("input_0", onnx.TensorProto.FLOAT, inp_shape),
            onnx.helper.make_tensor_value_info("input_1", onnx.TensorProto.FLOAT, inp_shape),
            onnx.helper.make_tensor_value_info("input_2", onnx.TensorProto.FLOAT, inp_shape),
        ]
        graph_outputs = [
            onnx.helper.make_tensor_value_info("output_0", onnx.TensorProto.FLOAT, inp_shape),
            onnx.helper.make_tensor_value_info("output_1", onnx.TensorProto.FLOAT, inp_shape),
        ]
        add_node0 = onnx.helper.make_node("Add", ["input_0", "input_1"], ["output_0"], name="Add0")
        add_node1 = onnx.helper.make_node("Add", ["input_1", "input_2"], ["output_1"], name="Add1")
        graph = onnx.helper.make_graph(
            [add_node0, add_node1],
            "AddGraph",
            graph_inputs,
            graph_outputs,
            initializer=[],
        )
        opset_imports = [onnx.helper.make_opsetid("", 21)]
        model = onnx.helper.make_model(graph, opset_imports=opset_imports)
        model = onnx.shape_inference.infer_shapes(model)
        onnx.checker.check_model(model, True)
        return model

    def test_dedicated_qdq_pair(self):
        # TODO: Support --matmul_const_b_only on qdq quant_format and refine the testcase
        script_args_set = [([], 12), (["--dedicated_qdq_pair"], 14)]
        for script_args, expect_num_nodes in script_args_set:
            with self.subTest(script_args=script_args, expect_num_nodes=expect_num_nodes):
                quant_model = self.run_static_quantize_runner(script_args)
                assert len(quant_model.graph.node) == expect_num_nodes

    def test_nodes_to_quantize(self):
        script_args_set = [
            (["--nodes_to_quantize", "Add0"], 8),
            (["--nodes_to_quantize", "Add1"], 8),
            (["--nodes_to_quantize", "Add0", "Add1"], 12),
        ]
        for script_args, expect_num_nodes in script_args_set:
            with self.subTest(script_args=script_args, expect_num_nodes=expect_num_nodes):
                quant_model = self.run_static_quantize_runner(script_args)
                assert len(quant_model.graph.node) == expect_num_nodes

    def test_nodes_to_exclude(self):
        script_args_set = [
            (["--nodes_to_exclude", "Add0"], 8),
            (["--nodes_to_exclude", "Add1"], 8),
            (["--nodes_to_exclude", "Add0", "Add1"], 2),
        ]
        for script_args, expect_num_nodes in script_args_set:
            with self.subTest(script_args=script_args, expect_num_nodes=expect_num_nodes):
                quant_model = self.run_static_quantize_runner(script_args)
                assert len(quant_model.graph.node) == expect_num_nodes


class TestConv(StaticQuantizeRunnerTestBase, unittest.TestCase):
    def onnx_model(
        self, inp_shape=(4, 3, 32, 32), weight_shape=(8, 3, 5, 5), bias_shape=(8,), out_shape=(4, 8, 28, 28)
    ) -> onnx.ModelProto:
        graph_inputs = [
            onnx.helper.make_tensor_value_info("input_0", onnx.TensorProto.FLOAT, inp_shape),
        ]
        graph_outputs = [onnx.helper.make_tensor_value_info("output_0", onnx.TensorProto.FLOAT, out_shape)]
        initializers = [
            onnx.helper.make_tensor(
                "weight_0", onnx.TensorProto.FLOAT, dims=weight_shape, vals=np.random.rand(*weight_shape)
            ),
            onnx.helper.make_tensor(
                "bias_0", onnx.TensorProto.FLOAT, dims=bias_shape, vals=np.random.rand(*bias_shape)
            ),
        ]

        conv_node = onnx.helper.make_node(
            "Conv",
            ["input_0", "weight_0", "bias_0"],
            ["output_0"],
            name="Conv0",
            kernel_shape=[5, 5],
        )
        graph = onnx.helper.make_graph(
            [conv_node],
            "ConvGraph",
            graph_inputs,
            graph_outputs,
            initializer=initializers,
        )
        opset_imports = [onnx.helper.make_opsetid("", 21)]
        model = onnx.helper.make_model(graph, opset_imports=opset_imports)
        model = onnx.shape_inference.infer_shapes(model)
        onnx.checker.check_model(model, True)
        return model

    def test_disable_quantize_bias(self):
        script_args_set = [
            ([], 7),
            (["--disable_quantize_bias"], 6),
        ]
        for script_args, expect_num_nodes in script_args_set:
            with self.subTest(script_args=script_args, expect_num_nodes=expect_num_nodes):
                quant_model = self.run_static_quantize_runner(script_args)
                assert len(quant_model.graph.node) == expect_num_nodes


class TestConvRelu(StaticQuantizeRunnerTestBase, unittest.TestCase):
    def onnx_model(
        self, inp_shape=(4, 3, 32, 32), weight_shape=(8, 3, 5, 5), bias_shape=(8,), out_shape=(4, 8, 28, 28)
    ) -> onnx.ModelProto:
        graph_inputs = [
            onnx.helper.make_tensor_value_info("input_0", onnx.TensorProto.FLOAT, inp_shape),
        ]
        graph_outputs = [onnx.helper.make_tensor_value_info("output_0", onnx.TensorProto.FLOAT, out_shape)]
        initializers = [
            onnx.helper.make_tensor(
                "weight_0", onnx.TensorProto.FLOAT, dims=weight_shape, vals=np.random.rand(*weight_shape)
            ),
            onnx.helper.make_tensor(
                "bias_0", onnx.TensorProto.FLOAT, dims=bias_shape, vals=np.random.rand(*bias_shape)
            ),
        ]

        conv_node = onnx.helper.make_node(
            "Conv",
            ["input_0", "weight_0", "bias_0"],
            ["conv_output_0"],
            name="Conv0",
            kernel_shape=[5, 5],
        )

        relu_node = onnx.helper.make_node(
            "Relu",
            ["conv_output_0"],
            ["output_0"],
            name="Relu0",
        )
        graph = onnx.helper.make_graph(
            [conv_node, relu_node],
            "ConvReluGraph",
            graph_inputs,
            graph_outputs,
            initializer=initializers,
        )
        opset_imports = [onnx.helper.make_opsetid("", 21)]
        model = onnx.helper.make_model(graph, opset_imports=opset_imports)
        model = onnx.shape_inference.infer_shapes(model)
        onnx.checker.check_model(model, True)
        return model

    def test_qdq_keep_removable_activations(self):
        script_args_set = [([], 7), (["--qdq_keep_removable_activations"], 10)]
        for script_args, expect_num_nodes in script_args_set:
            with self.subTest(script_args=script_args, expect_num_nodes=expect_num_nodes):
                quant_model = self.run_static_quantize_runner(script_args)
                assert len(quant_model.graph.node) == expect_num_nodes


class TestIfGraph(StaticQuantizeRunnerTestBase, unittest.TestCase):
    def onnx_model(self) -> onnx.ModelProto:
        cond = onnx.helper.make_tensor_value_info("cond", onnx.TensorProto.BOOL, [1])
        input_0 = onnx.helper.make_tensor_value_info("input_0", onnx.TensorProto.FLOAT, [4, 3, 32, 32])
        input_1 = onnx.helper.make_tensor_value_info("input_1", onnx.TensorProto.FLOAT, [4, 3, 32, 32])
        output_0 = onnx.helper.make_tensor_value_info("output_0", onnx.TensorProto.FLOAT, [4, 3, 32, 32])

        then_graph = onnx.helper.make_graph(
            nodes=[onnx.helper.make_node("Add", ["input_0", "input_1"], ["output_0"], name="Add0")],
            name="AddGraph",
            inputs=[],
            outputs=[output_0],
            initializer=[],
        )
        else_graph = onnx.helper.make_graph(
            nodes=[onnx.helper.make_node("Mul", ["input_0", "input_1"], ["output_0"], name="Mul0")],
            name="MulGraph",
            inputs=[],
            outputs=[output_0],
            initializer=[],
        )
        if_node = onnx.helper.make_node(
            "If", inputs=["cond"], outputs=["output_0"], then_branch=then_graph, else_branch=else_graph
        )
        graph = onnx.helper.make_graph(
            nodes=[if_node], name="main_graph", inputs=[cond, input_0, input_1], outputs=[output_0]
        )
        opset_imports = [onnx.helper.make_opsetid("", 21)]
        model = onnx.helper.make_model(graph, opset_imports=opset_imports)
        model = onnx.shape_inference.infer_shapes(model)
        onnx.checker.check_model(model, True)
        return model

    def test_enable_subgraph(self):
        # TODO: Investigate the reason of failure on --enable_subgraph
        script_args_set = [(["--enable_subgraph", "--quant_format", "qoperator"], False)]
        for script_args, success in script_args_set:
            with (
                self.subTest(script_args=script_args),
                contextlib.nullcontext() if success else self.assertRaises(Exception),
            ):
                self.run_static_quantize_runner(script_args)

    def test_calibration_method(self):
        # TODO: Investigate the reason of failure on calibration methods
        script_args_set = [
            (["--calibration_method", "minmax"], True),
            (["--calibration_method", "entropy"], False),
            (["--calibration_method", "percentile"], False),
            (["--calibration_method", "distribution"], False),
        ]
        for script_args, success in script_args_set:
            with (
                self.subTest(script_args=script_args),
                contextlib.nullcontext() if success else self.assertRaises(Exception),
            ):
                self.run_static_quantize_runner(script_args)


class TestWhere(StaticQuantizeRunnerTestBase, unittest.TestCase):
    def onnx_model(self, inp_shape=(4, 3, 32, 32)) -> onnx.ModelProto:
        graph_inputs = [
            onnx.helper.make_tensor_value_info("cond", onnx.TensorProto.BOOL, inp_shape),
            onnx.helper.make_tensor_value_info("input_0", onnx.TensorProto.FLOAT, inp_shape),
            onnx.helper.make_tensor_value_info("input_1", onnx.TensorProto.FLOAT, inp_shape),
        ]
        graph_outputs = [onnx.helper.make_tensor_value_info("output_0", onnx.TensorProto.FLOAT, inp_shape)]

        add_node = onnx.helper.make_node("Add", ["input_0", "input_1"], ["Add_output_0"], name="Add0")
        mul_node = onnx.helper.make_node("Mul", ["input_0", "input_1"], ["Mul_output_0"], name="Mul0")
        where_node = onnx.helper.make_node(
            "Where", ["cond", "Add_output_0", "Mul_output_0"], ["output_0"], name="Where0"
        )
        graph = onnx.helper.make_graph(
            [add_node, mul_node, where_node],
            "WhereGraph",
            graph_inputs,
            graph_outputs,
            initializer=[],
        )
        opset_imports = [onnx.helper.make_opsetid("", 21)]
        model = onnx.helper.make_model(graph, opset_imports=opset_imports)
        model = onnx.shape_inference.infer_shapes(model)
        onnx.checker.check_model(model, True)
        return model

    def test_quant_format(self):
        # TODO: Investigate the failure on --quant_format qoperator
        script_args_set = [(["--quant_format", "qdq"], True), (["--quant_format", "qoperator"], False)]
        for script_args, success in script_args_set:
            with (
                self.subTest(script_args=script_args),
                contextlib.nullcontext() if success else self.assertRaises(Exception),
            ):
                self.run_static_quantize_runner(script_args)

    def test_force_quantize_no_input_check(self):
        script_args_set = [
            (["--nodes_to_exclude", "Add0", "Mul0"], 3),
            (["--nodes_to_exclude", "Add0", "Mul0", "--force_quantize_no_input_check"], 9),
        ]
        for script_args, expect_num_node in script_args_set:
            with self.subTest(script_args=script_args):
                quant_model = self.run_static_quantize_runner(script_args)
                assert len(quant_model.graph.node) == expect_num_node


if __name__ == "__main__":
    unittest.main()
