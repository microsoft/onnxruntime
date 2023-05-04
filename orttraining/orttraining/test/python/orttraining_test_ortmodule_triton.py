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


def _torch_pow(input, exponent):
    return torch.pow(input, exponent)


def _torch_sqrt(input):
    return torch.sqrt(input)


def _torch_exp(input):
    return torch.exp(input)


def _torch_cast(input, **kwargs):
    return input.to(_onnx_dtype_to_torch_dtype(kwargs.get("to")))


def _torch_where(condition, x, y):
    return torch.where(condition, x, y)


def _torch_sum(*inputs):
    result = inputs[0]
    for input in inputs[1:]:
        result += input
    return result


def _troch_dropout_gard(dy, mask, **kwargs):
    ratio = kwargs.get("ratio", 0.5)
    return torch.where(mask, dy / (1.0 - ratio), 0.0)


def _torch_softmax(input, **kwargs):
    axis = kwargs.get("axis", -1)
    return torch.softmax(input, axis)


def _torch_reduce(input, func, **kwargs):
    rank = len(input.shape)
    axes = kwargs.get("axes", [idx for idx in range(rank)])
    keepdims = kwargs.get("keepdims", True)
    axes = [axis if axis >= 0 else rank + axis for axis in axes]
    axes.sort(reverse=True)
    result = input
    for axis in axes:
        result = func(result, dim=axis, keepdim=keepdims)
        if func == torch.max or func == torch.min:
            result = result[0]
    return result


def _torch_reduce_sum(input, **kwargs):
    return _torch_reduce(input, torch.sum, **kwargs)


def _torch_reduce_mean(input, **kwargs):
    return _torch_reduce(input, torch.mean, **kwargs)


def _torch_reduce_max(input, **kwargs):
    return _torch_reduce(input, torch.max, **kwargs)


def _torch_reduce_min(input, **kwargs):
    return _torch_reduce(input, torch.min, **kwargs)


def _torch_layer_norm(input, weight, bias, **kwargs):
    rank = len(input.shape)
    axis = kwargs.get("axis", -1)
    if axis < 0:
        axis += rank
    normalized_shape = input.shape[axis:]
    return torch.nn.functional.layer_norm(input, normalized_shape, weight, bias)


class TorchFuncExecutor:
    _INFER_FUNC_MAP = {
        "Add": _torch_add,
        "Sub": _torch_sub,
        "Mul": _torch_mul,
        "Div": _torch_div,
        "Pow": _torch_pow,
        "Sqrt": _torch_sqrt,
        "Exp": _torch_exp,
        "Cast": _torch_cast,
        "Where": _torch_where,
        "Sum": _torch_sum,
        "DropoutGrad": _troch_dropout_gard,
        "ReduceSum": _torch_reduce_sum,
        "ReduceMean": _torch_reduce_mean,
        "ReduceMax": _torch_reduce_max,
        "ReduceMin": _torch_reduce_min,
        "Softmax": _torch_softmax,
        "LayerNormalization": _torch_layer_norm,
    }

    @classmethod
    def run(cls, op_type, *torch_tensors, **kwargs):
        if op_type not in cls._INFER_FUNC_MAP:
            raise NotImplementedError(f"Unsupported op type: {op_type}")
        return cls._INFER_FUNC_MAP[op_type](*torch_tensors, **kwargs)


def _run_op_test(op_type, onnx_dtype, create_model_func, gen_inputs_func, **kwargs):
    rtol = kwargs.get("rtol", 1e-03 if onnx_dtype == TensorProto.FLOAT16 else 1e-04)
    atol = kwargs.get("atol", 1e-03 if onnx_dtype == TensorProto.FLOAT16 else 1e-05)
    pt_inputs = gen_inputs_func(_onnx_dtype_to_torch_dtype(onnx_dtype))
    ort_inputs = copy.deepcopy(pt_inputs)
    ort_inputs = [tensor.to(torch.uint8) if tensor.dtype == torch.bool else tensor for tensor in ort_inputs]
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
        loss = prediction.sum()
        loss.backward()
        return prediction

    rtol = kwargs.get("rtol", 1e-03 if dtype == torch.float16 else 1e-04)
    atol = kwargs.get("atol", 1e-03 if dtype == torch.float16 else 1e-05)
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
@pytest.mark.parametrize("input_shapes", [([1024, 2], [1024, 2]), ([2, 3, 3, 3], [3, 1, 3]), ([2049], [1])])
def test_binary_elementwise_op(op_type, onnx_dtype, input_shapes):
    def _create_model(op_type, onnx_dtype):
        graph = helper.make_graph(
            [helper.make_node(op_type, ["X", "Y"], ["Z"], name="test")],
            "test",
            [
                helper.make_tensor_value_info("X", onnx_dtype, None),
                helper.make_tensor_value_info("Y", onnx_dtype, None),
            ],
            [helper.make_tensor_value_info("Z", onnx_dtype, None)],
        )
        return helper.make_model(graph, producer_name="test")

    def _gen_inputs(dtype):
        return [
            torch.randn(*input_shapes[0], dtype=dtype, device=DEVICE),
            torch.randn(*input_shapes[1], dtype=dtype, device=DEVICE),
        ]

    _run_op_test(op_type, onnx_dtype, _create_model, _gen_inputs)


@pytest.mark.parametrize("onnx_dtype", [TensorProto.FLOAT, TensorProto.FLOAT16])
@pytest.mark.parametrize("input_shapes", [([1024, 2], [1024, 2]), ([2, 3, 3, 3], [3, 1, 3], [2, 1, 3, 1])])
def test_sum_op(onnx_dtype, input_shapes):
    def _create_model(op_type, onnx_dtype):
        graph = helper.make_graph(
            [helper.make_node(op_type, [f"I{idx}" for idx in range(len(input_shapes))], ["O"], name="test")],
            "test",
            [helper.make_tensor_value_info(f"I{idx}", onnx_dtype, None) for idx in range(len(input_shapes))],
            [helper.make_tensor_value_info("O", onnx_dtype, None)],
        )
        return helper.make_model(graph, producer_name="test")

    def _gen_inputs(dtype):
        return [torch.randn(*input_shape, dtype=dtype, device=DEVICE) for input_shape in input_shapes]

    _run_op_test("Sum", onnx_dtype, _create_model, _gen_inputs)


@pytest.mark.parametrize("op_type", ["Sqrt", "Exp"])
@pytest.mark.parametrize("onnx_dtype", [TensorProto.FLOAT, TensorProto.FLOAT16])
@pytest.mark.parametrize("input_shape", [[1024, 4], [2, 3, 3, 3], [2049, 1]])
def test_unary_elementwise_op(op_type, onnx_dtype, input_shape):
    def _create_model(op_type, onnx_dtype):
        graph = helper.make_graph(
            [helper.make_node(op_type, ["X"], ["Y"], name="test")],
            "test",
            [helper.make_tensor_value_info("X", onnx_dtype, None)],
            [helper.make_tensor_value_info("Y", onnx_dtype, None)],
        )
        return helper.make_model(graph, producer_name="test")

    def _gen_inputs(dtype):
        return [torch.rand(*input_shape, dtype=dtype, device=DEVICE)]

    _run_op_test(op_type, onnx_dtype, _create_model, _gen_inputs)


@pytest.mark.parametrize("onnx_dtype", [TensorProto.FLOAT, TensorProto.FLOAT16])
@pytest.mark.parametrize("input_shape_and_exponent", [([1024, 2], 2.0), ([2, 3, 3, 3], 0.5), ([2049], 3.0)])
def test_pow_op(onnx_dtype, input_shape_and_exponent):
    def _create_model(op_type, onnx_dtype):
        graph = helper.make_graph(
            [helper.make_node(op_type, ["X", "Y"], ["Z"], name="test")],
            "test",
            [
                helper.make_tensor_value_info("X", onnx_dtype, None),
                helper.make_tensor_value_info("Y", onnx_dtype, None),
            ],
            [helper.make_tensor_value_info("Z", onnx_dtype, None)],
        )
        return helper.make_model(graph, producer_name="test")

    def _gen_inputs(dtype):
        return [
            torch.rand(*input_shape_and_exponent[0], dtype=dtype, device=DEVICE),
            torch.tensor(input_shape_and_exponent[1], dtype=dtype, device=DEVICE),
        ]

    _run_op_test("Pow", onnx_dtype, _create_model, _gen_inputs)


@pytest.mark.parametrize("onnx_dtype", [TensorProto.FLOAT, TensorProto.FLOAT16])
@pytest.mark.parametrize("input_shape", [[1024, 2], [2, 3, 3, 3], [1, 2050]])
def test_cast_op(onnx_dtype, input_shape):
    def _create_model(op_type, onnx_dtype, **kwargs):
        graph = helper.make_graph(
            [helper.make_node(op_type, ["X"], ["Y"], name="test", **kwargs)],
            "test",
            [
                helper.make_tensor_value_info("X", onnx_dtype, None),
            ],
            [helper.make_tensor_value_info("Y", kwargs.get("to"), None)],
        )
        return helper.make_model(graph, producer_name="test")

    def _gen_inputs(dtype):
        return [torch.randn(*input_shape, dtype=dtype, device=DEVICE)]

    kwargs = {"to": TensorProto.FLOAT16 if onnx_dtype == TensorProto.FLOAT else TensorProto.FLOAT}
    _run_op_test("Cast", onnx_dtype, _create_model, _gen_inputs, **kwargs)


@pytest.mark.parametrize("onnx_dtype", [TensorProto.FLOAT, TensorProto.FLOAT16])
@pytest.mark.parametrize("input_shapes", [([2, 1024], [2, 1024], [2, 1024]), ([2, 1, 3, 1], [2, 3, 3, 3], [3, 1, 3])])
def test_where_op(onnx_dtype, input_shapes):
    def _create_model(op_type, onnx_dtype):
        graph = helper.make_graph(
            [helper.make_node(op_type, ["C", "X", "Y"], ["Z"], name="test")],
            "test",
            [
                helper.make_tensor_value_info("C", TensorProto.BOOL, None),
                helper.make_tensor_value_info("X", onnx_dtype, None),
                helper.make_tensor_value_info("Y", onnx_dtype, None),
            ],
            [helper.make_tensor_value_info("Z", onnx_dtype, None)],
        )
        return helper.make_model(graph, producer_name="test")

    def _gen_inputs(dtype):
        return [
            torch.rand(*input_shapes[0], dtype=dtype, device=DEVICE) < 0.5,
            torch.randn(*input_shapes[1], dtype=dtype, device=DEVICE),
            torch.randn(*input_shapes[2], dtype=dtype, device=DEVICE),
        ]

    _run_op_test("Where", onnx_dtype, _create_model, _gen_inputs)


@pytest.mark.parametrize("onnx_dtype", [TensorProto.FLOAT, TensorProto.FLOAT16])
@pytest.mark.parametrize("input_shape_and_ratio", [([1024, 2], 0.2), ([25, 75], 0.6), ([1, 2049], 0.5)])
def test_dropout_op(onnx_dtype, input_shape_and_ratio):
    graph = helper.make_graph(
        [
            helper.make_node("Add", ["X1", "X2"], ["T1"], name="add1"),
            helper.make_node("Dropout", ["T1", "ratio"], ["Y1", "mask1"], name="dropout1"),
            helper.make_node("Add", ["Y1", "X3"], ["T2"], name="add2"),
            helper.make_node("Dropout", ["T2", "ratio"], ["Y2", "mask2"], name="dropout2"),
        ],
        "test",
        [
            helper.make_tensor_value_info("X1", onnx_dtype, None),
            helper.make_tensor_value_info("X2", onnx_dtype, None),
            helper.make_tensor_value_info("X3", onnx_dtype, None),
        ],
        [
            helper.make_tensor_value_info("Y1", onnx_dtype, None),
            helper.make_tensor_value_info("mask1", TensorProto.BOOL, None),
            helper.make_tensor_value_info("Y2", onnx_dtype, None),
            helper.make_tensor_value_info("mask2", TensorProto.BOOL, None),
        ],
        initializer=[helper.make_tensor("ratio", TensorProto.FLOAT, (), [input_shape_and_ratio[1]])],
    )
    model_str = helper.make_model(graph, producer_name="test").SerializeToString()
    torch_dtype = _onnx_dtype_to_torch_dtype(onnx_dtype)
    input_tensor = [
        torch.randn(*input_shape_and_ratio[0], dtype=torch_dtype, device=DEVICE),
        torch.randn(*input_shape_and_ratio[0], dtype=torch_dtype, device=DEVICE),
        torch.randn(*input_shape_and_ratio[0], dtype=torch_dtype, device=DEVICE),
    ]
    outputs = execute_triton_op("", hash(model_str), model_str, *[to_dlpack(t) for t in input_tensor])
    y1, mask1, y2, mask2 = tuple([_from_dlpack(o).detach().cpu().numpy().flatten() for o in outputs])
    x1 = (input_tensor[0] + input_tensor[1]).detach().cpu().numpy().flatten()
    x2 = y1 * mask1 + input_tensor[2].detach().cpu().numpy().flatten()

    def _check_output(x, y, mask, ratio):
        all_count = 0
        masked_count = 0
        for x_value, y_value, mask_value in zip(x, y, mask):
            if mask_value:
                assert abs(y_value - x_value / (1.0 - ratio)) < 0.05
            else:
                assert y_value == 0
            if not mask_value:
                masked_count += 1
            all_count += 1
        assert abs(masked_count / all_count - ratio) < 0.05

    _check_output(x1, y1, mask1, input_shape_and_ratio[1])
    _check_output(x2, y2, mask2, input_shape_and_ratio[1])
    assert any(mask1[i] != mask2[i] for i in range(len(mask1)))


@pytest.mark.parametrize("onnx_dtype", [TensorProto.FLOAT, TensorProto.FLOAT16])
@pytest.mark.parametrize("input_shape_and_ratio", [([1024, 2], 0.2), ([25, 75], 0.6), ([1, 2049], 0.5)])
def test_dropout_grad_op(onnx_dtype, input_shape_and_ratio):
    def _create_model(op_type, onnx_dtype, **kwargs):
        graph = helper.make_graph(
            [
                helper.make_node(op_type, ["dY", "mask", "ratio"], ["dX"], name="test", domain="com.microsoft"),
            ],
            "test",
            [
                helper.make_tensor_value_info("dY", onnx_dtype, None),
                helper.make_tensor_value_info("mask", TensorProto.BOOL, None),
            ],
            [helper.make_tensor_value_info("dX", onnx_dtype, None)],
            initializer=[helper.make_tensor("ratio", TensorProto.FLOAT, (), [input_shape_and_ratio[1]])],
        )
        return helper.make_model(graph, producer_name="test")

    def _gen_inputs(dtype):
        return [
            torch.randn(*input_shape_and_ratio[0], dtype=dtype, device=DEVICE),
            torch.rand(*input_shape_and_ratio[0], dtype=dtype, device=DEVICE) >= input_shape_and_ratio[1],
        ]

    kwargs = {"ratio": input_shape_and_ratio[1]}
    _run_op_test("DropoutGrad", onnx_dtype, _create_model, _gen_inputs, **kwargs)


@pytest.mark.parametrize("op_type", ["ReduceMax", "ReduceMean", "ReduceMin", "ReduceSum"])
@pytest.mark.parametrize("onnx_dtype", [TensorProto.FLOAT, TensorProto.FLOAT16])
@pytest.mark.parametrize(
    "input_shape_and_reduce_info",
    [
        ([1024, 2], [-1], True),
        ([2, 3, 3, 3], [1, 2], False),
        ([123, 4, 5, 6], [2], True),
        ([16, 8, 16, 8], [1, 3], True),
        ([16, 8, 16, 8], [0, 2], False),
    ],
)
def test_reduce_op(op_type, onnx_dtype, input_shape_and_reduce_info):
    def _create_model(op_type, onnx_dtype, **kwargs):
        reduce_inputs = ["X"]
        initializer = []
        if input_shape_and_reduce_info[1] is not None:
            reduce_inputs.append("axes")
            initializer.append(
                helper.make_tensor(
                    "axes",
                    onnx.TensorProto.INT64,
                    [len(input_shape_and_reduce_info[1])],
                    input_shape_and_reduce_info[1],
                )
            )
        node = (
            helper.make_node(op_type, reduce_inputs, ["Y"], name="test", keepdims=input_shape_and_reduce_info[2])
            if input_shape_and_reduce_info[2] is not None
            else helper.make_node(op_type, reduce_inputs, ["Y"], name="test")
        )
        graph = helper.make_graph(
            [node],
            "test",
            [helper.make_tensor_value_info("X", onnx_dtype, None)],
            [helper.make_tensor_value_info("Y", onnx_dtype, None)],
            initializer=initializer,
        )
        return helper.make_model(graph, producer_name="test")

    def _gen_inputs(dtype):
        return [torch.randn(*input_shape_and_reduce_info[0], dtype=dtype, device=DEVICE)]

    kwargs = {}
    if input_shape_and_reduce_info[1] is not None:
        kwargs["axes"] = input_shape_and_reduce_info[1]
    if input_shape_and_reduce_info[2] is not None:
        kwargs["keepdims"] = input_shape_and_reduce_info[2]
    _run_op_test(op_type, onnx_dtype, _create_model, _gen_inputs, **kwargs)


@pytest.mark.parametrize("onnx_dtype", [TensorProto.FLOAT, TensorProto.FLOAT16])
@pytest.mark.parametrize("input_shape_and_axis", [([2, 1024], -1), ([2, 3, 3, 3], 1), ([4, 2049], 1)])
def test_softmax_op(onnx_dtype, input_shape_and_axis):
    def _create_model(op_type, onnx_dtype, **kwargs):
        graph = helper.make_graph(
            [helper.make_node(op_type, ["X"], ["Y"], name="test", **kwargs)],
            "test",
            [helper.make_tensor_value_info("X", onnx_dtype, None)],
            [helper.make_tensor_value_info("Y", onnx_dtype, None)],
        )
        return helper.make_model(graph, producer_name="test")

    def _gen_inputs(dtype):
        return [torch.randn(*input_shape_and_axis[0], dtype=dtype, device=DEVICE)]

    kwargs = {"axis": input_shape_and_axis[1]}
    _run_op_test("Softmax", onnx_dtype, _create_model, _gen_inputs, **kwargs)


@pytest.mark.parametrize("onnx_dtype", [TensorProto.FLOAT, TensorProto.FLOAT16])
@pytest.mark.parametrize("input_shape_and_axis", [([2, 1024], -1), ([2, 3, 3, 3], 2), ([4, 2049], 1)])
def test_layer_norm_op(onnx_dtype, input_shape_and_axis):
    def _create_model(op_type, onnx_dtype, **kwargs):
        graph = helper.make_graph(
            [helper.make_node(op_type, ["X", "W", "B"], ["Y"], name="test", **kwargs)],
            "test",
            [
                helper.make_tensor_value_info("X", onnx_dtype, None),
                helper.make_tensor_value_info("W", onnx_dtype, None),
                helper.make_tensor_value_info("B", onnx_dtype, None),
            ],
            [helper.make_tensor_value_info("Y", onnx_dtype, None)],
        )
        return helper.make_model(graph, producer_name="test")

    def _gen_inputs(dtype):
        return [
            torch.randn(*input_shape_and_axis[0], dtype=dtype, device=DEVICE),
            torch.randn(*input_shape_and_axis[0][input_shape_and_axis[1] :], dtype=dtype, device=DEVICE),
            torch.randn(input_shape_and_axis[0][input_shape_and_axis[1] :], dtype=dtype, device=DEVICE),
        ]

    kwargs = {"axis": input_shape_and_axis[1]}
    _run_op_test("LayerNormalization", onnx_dtype, _create_model, _gen_inputs, **kwargs)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_elementwise_module(dtype):
    N, D, H, W = 8, 768, 12, 64

    class NeuralNetElementwise(torch.nn.Module):
        def forward(self, input1, input2, input3, input4):
            return input1 + input2 - input3 * input4

    def _gen_inputs(dtype):
        return [
            torch.randn(N, D, H, W, dtype=dtype, device=DEVICE, requires_grad=True),
            torch.randn(W, dtype=dtype, device=DEVICE, requires_grad=True),
            torch.randn(D, 1, 1, dtype=dtype, device=DEVICE, requires_grad=True),
            torch.randn(N, 1, H, W, dtype=dtype, device=DEVICE, requires_grad=True),
        ]

    _run_module_test(NeuralNetElementwise, dtype, _gen_inputs, 1)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("input_shapes_and_axis", [([2, 3, 3, 3], [3, 3], 2), ([2, 1024], [2, 1024], -1)])
def test_softmax_module(dtype, input_shapes_and_axis):
    class NeuralNetSoftmax(torch.nn.Module):
        def forward(self, input1, input2):
            return torch.softmax(input1 * input2, dim=input_shapes_and_axis[2])

    def _gen_inputs(dtype):
        return [
            torch.randn(*input_shapes_and_axis[0], dtype=dtype, device=DEVICE, requires_grad=True),
            torch.randn(*input_shapes_and_axis[1], dtype=dtype, device=DEVICE, requires_grad=True),
        ]

    _run_module_test(NeuralNetSoftmax, dtype, _gen_inputs, 2)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("input_shapes_and_axis", [([2, 3, 3, 3], [3, 3], 2), ([2, 1024], [2, 1024], -1)])
def test_layer_norm_module(dtype, input_shapes_and_axis):
    class NeuralNetLayerNorm(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer_norm = torch.nn.LayerNorm(
                *input_shapes_and_axis[0][input_shapes_and_axis[2] :], device=DEVICE, dtype=dtype
            )

        def forward(self, input1, input2):
            return self.layer_norm(input1 * input2)

    def _gen_inputs(dtype):
        return [
            torch.randn(*input_shapes_and_axis[0], dtype=dtype, device=DEVICE, requires_grad=True),
            torch.randn(*input_shapes_and_axis[1], dtype=dtype, device=DEVICE, requires_grad=True),
        ]

    _run_module_test(NeuralNetLayerNorm, dtype, _gen_inputs, 2)
