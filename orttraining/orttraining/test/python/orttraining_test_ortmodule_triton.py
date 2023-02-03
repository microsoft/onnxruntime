# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import copy
import os

import _test_helpers
import onnx
import pytest
import torch
from onnx import TensorProto, helper
from torch._C import _from_dlpack
from torch.utils.dlpack import to_dlpack

from onnxruntime.training.ortmodule import DebugOptions, ORTModule
from onnxruntime.training.ortmodule.ort_triton import execute_triton_op

pytest.importorskip("triton")

DEVICE = "cuda"


@pytest.fixture(scope="session", autouse=True)
def run_before_test_session(request):
    def insert_use_triton_env():
        os.environ["ORTMODULE_USE_TRITON"] = "1"

    def remove_use_triton_env():
        del os.environ["ORTMODULE_USE_TRITON"]

    insert_use_triton_env()
    request.addfinalizer(remove_use_triton_env)


def _onnx_dtype_to_torch_dtype(onnx_dtype):
    if onnx_dtype == TensorProto.FLOAT:
        return torch.float32
    elif onnx_dtype == TensorProto.FLOAT16:
        return torch.float16
    else:
        raise RuntimeError("Unsupported ONNX dtype")


def _torch_add(input1, input2):
    return input1 + input2


def _torch_sub(input1, input2):
    return input1 - input2


def _torch_mul(input1, input2):
    return input1 * input2


def _torch_div(input1, input2):
    return input1 / input2


def _torch_softmax(input, **kwargs):
    axis = kwargs.get("axis", -1)
    return torch.softmax(input, axis)


class TorchFuncExecutor:
    _INFER_FUNC_MAP = {
        "Add": _torch_add,
        "Sub": _torch_sub,
        "Mul": _torch_mul,
        "Div": _torch_div,
        "Softmax": _torch_softmax,
    }

    @classmethod
    def run(cls, op_type, *torch_tensors, **kwargs):
        if op_type not in cls._INFER_FUNC_MAP:
            raise NotImplementedError(f"Unsupported op type: {op_type}")
        return cls._INFER_FUNC_MAP[op_type](*torch_tensors, **kwargs)


def _run_op_test(op_type, onnx_dtype, create_model_func, gen_inputs_func, **kwargs):
    rtol = kwargs.get("rtol", 1e-04)
    atol = kwargs.get("atol", 1e-05)
    pt_inputs = gen_inputs_func(_onnx_dtype_to_torch_dtype(onnx_dtype))
    ort_inputs = copy.deepcopy(pt_inputs)
    pt_outputs = TorchFuncExecutor.run(op_type, *pt_inputs, **kwargs)
    model_str = create_model_func(op_type, onnx_dtype, **kwargs).SerializeToString()
    ort_outputs = execute_triton_op("", hash(model_str), model_str, *[to_dlpack(tensor) for tensor in ort_inputs])
    if isinstance(pt_outputs, tuple):
        assert isinstance(ort_outputs, tuple)
        assert len(pt_outputs) == len(ort_outputs)
        for pt_output, ort_output in zip(pt_outputs, ort_outputs):
            _test_helpers.assert_values_are_close(pt_output, _from_dlpack(ort_output), rtol=rtol, atol=atol)
    else:
        _test_helpers.assert_values_are_close(pt_outputs, _from_dlpack(ort_outputs), rtol=rtol, atol=atol)


def _run_module_test(module_cls, dtype, gen_inputs_func, triton_op_count, **kwargs):
    pt_model = module_cls().to(DEVICE)
    ort_model = ORTModule(copy.deepcopy(pt_model), DebugOptions(save_onnx=True, onnx_prefix="triton_model"))

    def run_step(model, *tensors):
        prediction = model(*tensors)
        return prediction

    rtol = kwargs.get("rtol", 1e-04)
    atol = kwargs.get("atol", 1e-05)
    for _ in range(10):
        pt_inputs = gen_inputs_func(dtype)
        ort_inputs = copy.deepcopy(pt_inputs)
        pt_output = run_step(pt_model, *pt_inputs)
        ort_output = run_step(ort_model, *ort_inputs)
        _test_helpers.assert_values_are_close(pt_output, ort_output, rtol=rtol, atol=atol)
        _test_helpers.assert_gradients_match_and_reset_gradient(pt_model, ort_model, rtol=rtol, atol=atol)

    assert os.path.exists(os.path.join(os.getcwd(), "triton_model_torch_exported_training.onnx"))
    assert os.path.exists(os.path.join(os.getcwd(), "triton_model_optimized_training.onnx"))
    assert os.path.exists(os.path.join(os.getcwd(), "triton_model_optimized_pre_grad_training.onnx"))
    assert os.path.exists(os.path.join(os.getcwd(), "triton_model_execution_model_training.onnx"))
    model = onnx.load(os.path.join(os.getcwd(), "triton_model_execution_model_training.onnx"))
    actual_triton_op_count = 0
    for node in model.graph.node:
        if node.op_type == "TritonOp":
            actual_triton_op_count += 1
    assert actual_triton_op_count == triton_op_count
    os.remove(os.path.join(os.getcwd(), "triton_model_torch_exported_training.onnx"))
    os.remove(os.path.join(os.getcwd(), "triton_model_optimized_training.onnx"))
    os.remove(os.path.join(os.getcwd(), "triton_model_optimized_pre_grad_training.onnx"))
    os.remove(os.path.join(os.getcwd(), "triton_model_execution_model_training.onnx"))


@pytest.mark.parametrize("op_type", ["Add", "Sub", "Mul", "Div"])
@pytest.mark.parametrize("onnx_dtype", [TensorProto.FLOAT, TensorProto.FLOAT16])
@pytest.mark.parametrize("input_shapes", [([3, 4], [3, 4]), ([2, 3, 3, 3], [3, 1, 3])])
def test_elementwise_op(op_type, onnx_dtype, input_shapes):
    def _create_model(op_type, onnx_dtype):
        node = helper.make_node(op_type, ["X", "Y"], ["Z"], name="test")
        graph = helper.make_graph(
            [node],
            "test",
            [
                helper.make_tensor_value_info("X", onnx_dtype, None),
                helper.make_tensor_value_info("Y", onnx_dtype, None),
            ],
            [helper.make_tensor_value_info("Z", onnx_dtype, None)],
        )
        return helper.make_model(graph, producer_name="test")

    def _gen_inputs(dtype):
        return (
            torch.randn(*input_shapes[0], dtype=dtype, device=DEVICE),
            torch.randn(*input_shapes[1], dtype=dtype, device=DEVICE),
        )

    _run_op_test(op_type, onnx_dtype, _create_model, _gen_inputs)


@pytest.mark.parametrize("onnx_dtype", [TensorProto.FLOAT, TensorProto.FLOAT16])
@pytest.mark.parametrize("input_shape_and_axis", [([3, 4], -1), ([2, 3, 3, 3], 1)])
def test_softmax_op(onnx_dtype, input_shape_and_axis):
    kwargs = {"axis": input_shape_and_axis[1]}
    if onnx_dtype == TensorProto.FLOAT16:
        kwargs["rtol"] = 1e-03
        kwargs["atol"] = 1e-03

    def _create_model(op_type, onnx_dtype, **kwargs):
        node = helper.make_node(op_type, ["X"], ["Y"], name="test", **kwargs)
        graph = helper.make_graph(
            [node],
            "test",
            [helper.make_tensor_value_info("X", onnx_dtype, None)],
            [helper.make_tensor_value_info("Y", onnx_dtype, None)],
        )
        return helper.make_model(graph, producer_name="test")

    def _gen_inputs(dtype):
        return [torch.randn(*input_shape_and_axis[0], dtype=dtype, device=DEVICE)]

    _run_op_test("Softmax", onnx_dtype, _create_model, _gen_inputs, **kwargs)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_elementwise_module(dtype):
    class NeuralNetElementwise(torch.nn.Module):
        def forward(self, input1, input2, input3, input4):
            return input1 + input2 - input3 * input4

    def _gen_inputs(dtype):
        return (
            torch.randn(3, 4, dtype=dtype, device=DEVICE, requires_grad=True),
            torch.randn(4, dtype=dtype, device=DEVICE, requires_grad=True),
            torch.randn(3, 1, dtype=dtype, device=DEVICE, requires_grad=True),
            torch.randn(3, 4, dtype=dtype, device=DEVICE, requires_grad=True),
        )

    kwargs = {}
    if dtype == torch.float16:
        kwargs["rtol"] = 1e-03
        kwargs["atol"] = 1e-03
    _run_module_test(NeuralNetElementwise, dtype, _gen_inputs, 1, **kwargs)
