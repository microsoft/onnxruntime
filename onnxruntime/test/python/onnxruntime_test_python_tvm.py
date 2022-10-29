# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
"""
Module for unit testing of TVM EP
"""

import os
import sys
import tempfile
import unittest
from typing import Any, AnyStr, Dict, List, Tuple

import numpy
import tvm
from numpy.testing import assert_almost_equal
from onnx import ModelProto, TensorProto, mapping
from onnx.helper import make_graph, make_model, make_node, make_tensor_value_info

import onnxruntime

numpy.random.seed(32)


def is_windows():
    """
    Function to determine the Windows system
    """
    return sys.platform.startswith("win")


def get_model_with_dynamic_shapes() -> ModelProto:
    """
    Create model with Dynamic Shapes
    """
    x = make_tensor_value_info("X", TensorProto.FLOAT, [None, None])  # pylint: disable=invalid-name, no-member
    a = make_tensor_value_info("A", TensorProto.FLOAT, [None, None])  # pylint: disable=invalid-name, no-member
    b = make_tensor_value_info("B", TensorProto.FLOAT, [None, None])  # pylint: disable=invalid-name, no-member
    y = make_tensor_value_info("Y", TensorProto.FLOAT, [None, None])  # pylint: disable=invalid-name, no-member
    node1 = make_node("MatMul", ["X", "A"], ["XA"])
    node2 = make_node("Add", ["XA", "B"], ["Y"])
    graph = make_graph([node1, node2], "lr", [x, a, b], [y])
    onnx_model = make_model(graph)
    return onnx_model


def get_model_with_fixed_shapes() -> ModelProto:
    """
    Create model with Static Shapes
    """

    def change_input_shape(model: ModelProto, ind: int, shape: Tuple) -> None:
        """
        Function to change the input form
        """
        dims = model.graph.input[ind].type.tensor_type.shape.dim
        assert len(dims) == len(shape), "Input rank and new shape rank do not match."
        for i, new_dim in enumerate(shape):
            model.graph.input[ind].type.tensor_type.shape.dim[i].dim_value = new_dim

    dynamic_model = get_model_with_dynamic_shapes()
    change_input_shape(dynamic_model, 0, (1, 2))  # X
    change_input_shape(dynamic_model, 1, (2, 2))  # A
    change_input_shape(dynamic_model, 2, (1, 2))  # B
    return dynamic_model


def get_input_data_for_model_with_dynamic_shapes() -> Dict[AnyStr, numpy.ndarray]:
    """
    Create input data for model with dynamic shapes
    """
    a = numpy.random.randn(2, 2).astype(numpy.float32)  # pylint: disable=invalid-name
    b = numpy.random.randn(1, 2).astype(numpy.float32)  # pylint: disable=invalid-name
    x = numpy.random.randn(1, 2).astype(numpy.float32)  # pylint: disable=invalid-name
    data = {"A": a, "B": b, "X": x}
    return data


def get_input_data_for_model_with_fixed_shapes(onnx_model: ModelProto) -> Dict[AnyStr, numpy.ndarray]:
    """
    Create input data for model with static shapes
    """

    def get_onnx_input_names(model: ModelProto) -> List[AnyStr]:
        inputs = [node.name for node in model.graph.input]
        initializer = [node.name for node in model.graph.initializer]
        inputs = list(set(inputs) - set(initializer))
        return sorted(inputs)

    def get_onnx_input_types(model: ModelProto) -> List[numpy.dtype]:
        input_names = get_onnx_input_names(model)
        return [
            mapping.TENSOR_TYPE_TO_NP_TYPE[node.type.tensor_type.elem_type]
            for node in sorted(model.graph.input, key=lambda node: node.name)
            if node.name in input_names
        ]

    def get_onnx_input_shapes(model: ModelProto) -> List[List[int]]:
        input_names = get_onnx_input_names(model)
        return [
            [dv.dim_value for dv in node.type.tensor_type.shape.dim]
            for node in sorted(model.graph.input, key=lambda node: node.name)
            if node.name in input_names
        ]

    input_names = get_onnx_input_names(onnx_model)
    input_shapes = get_onnx_input_shapes(onnx_model)
    input_types = get_onnx_input_types(onnx_model)
    assert len(input_names) == len(input_types) == len(input_shapes)
    random_inputs = [numpy.random.uniform(size=shape).astype(dtype) for shape, dtype in zip(input_shapes, input_types)]
    return dict(zip(input_names, random_inputs))


def get_input_names_and_shapes(data: Dict[AnyStr, numpy.ndarray]) -> Tuple[List[AnyStr], List[AnyStr]]:
    """
    Create text representations for model input names and shapes
    """
    keys = list(data.keys())
    values = [data[key] for key in keys]
    return (
        list(data.keys()),
        [str(value.shape).replace(",", "").replace("(", "[").replace(")", "]") for value in values],
    )


def get_cpu_output(onnx_model: ModelProto, data: Dict[AnyStr, numpy.ndarray]) -> List[numpy.ndarray]:
    """
    Run inference with CPUExecutionProvider
    """
    # pylint: disable=no-member
    sess = onnxruntime.InferenceSession(
        onnx_model.SerializeToString(),
        providers=["CPUExecutionProvider"],
    )
    output = sess.run(None, data)
    return output


def get_tvm_output(
    onnx_model: ModelProto, data: Dict[AnyStr, numpy.ndarray], provider_options: Dict[AnyStr, Any]
) -> List[numpy.ndarray]:
    """
    Run inference with TVMExecutionProvider
    """
    session_options = onnxruntime.SessionOptions()  # pylint: disable=no-member
    session_options.log_severity_level = 0
    session_options.log_verbosity_level = 0
    # pylint: disable=no-member
    session_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL

    sess = onnxruntime.InferenceSession(
        onnx_model.SerializeToString(),
        session_options,
        providers=["TvmExecutionProvider"],
        provider_options=[provider_options],
    )

    output = sess.run(None, data)
    return output


# pylint: disable=no-member
def compile_virtual_machine(model: ModelProto, target_str: AnyStr) -> tvm.runtime.vm.Executable:
    """
    Compile ONNX model using VirtualMachine
    """
    ir_mod, _ = tvm.relay.frontend.from_onnx(
        model,
        opset=model.opset_import[0].version,
        freeze_params=True,
    )
    target = tvm.target.Target(target=target_str, host=target_str)
    return tvm.relay.backend.vm.compile(ir_mod, target)


def serialize_virtual_machine(vm_exec: tvm.runtime.vm.Executable) -> AnyStr:
    """
    Serialize VirtualMachine
    """
    temp_directory = tempfile.mkdtemp()
    path_consts = os.path.join(temp_directory, "consts")
    vm_exec.move_late_bound_consts(path_consts, byte_limit=256)
    lib_path = os.path.join(temp_directory, f"model.{'dll' if is_windows() else 'so'}")
    code_path = os.path.join(temp_directory, "model.ro")
    code, lib = vm_exec.save()
    lib.export_library(lib_path)
    with open(code_path, "wb") as code_file:
        code_file.write(code)
    return temp_directory


class TestTVM(unittest.TestCase):
    """
    Unit tests for TVM EP
    """

    @staticmethod
    def test_accuracy_for_model_with_dynamic_shapes():
        """
        Accuracy test for model with dynamic shapes
        """
        onnx_model = get_model_with_dynamic_shapes()
        data = get_input_data_for_model_with_dynamic_shapes()

        cpu_output = get_cpu_output(onnx_model, data)
        names, shapes = get_input_names_and_shapes(data)
        provider_options = dict(
            target="llvm",
            input_names=" ".join(names),
            input_shapes=" ".join(shapes),
        )
        tvm_output = get_tvm_output(onnx_model, data, provider_options)

        assert_almost_equal(cpu_output, tvm_output, decimal=5)

    @staticmethod
    def test_accuracy_for_tvm_so():
        """
        Accuracy test for TVMso Ep
        """
        onnx_model = get_model_with_fixed_shapes()
        data = get_input_data_for_model_with_fixed_shapes(onnx_model)

        cpu_output = get_cpu_output(onnx_model, data)

        compiled_vm_exec = compile_virtual_machine(onnx_model, target_str="llvm")
        so_folder = serialize_virtual_machine(compiled_vm_exec)
        provider_options = dict(
            target="llvm",
            so_folder=so_folder,
        )
        tvm_output = get_tvm_output(onnx_model, data, provider_options)

        assert_almost_equal(cpu_output, tvm_output, decimal=5)


if __name__ == "__main__":
    if "TvmExecutionProvider" not in onnxruntime.get_available_providers():
        raise AssertionError(f"Unable to find 'TvmExecutionProvider' in {onnxruntime.get_available_providers()}")
    unittest.main()
