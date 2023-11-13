# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import copy
import json
import os
import random
import uuid

import _test_helpers
import onnx
import pytest
import torch
from onnx import TensorProto, helper
from torch._C import _from_dlpack
from torch.utils.dlpack import to_dlpack

from onnxruntime.training.ort_triton import call_triton_by_name, call_triton_by_onnx
from onnxruntime.training.ortmodule import DebugOptions, ORTModule

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


def _torch_gelu(input):
    return torch.nn.functional.gelu(input)


def _torch_quick_gelu(input, **kwargs):
    alpha = kwargs.get("alpha", 1.702)
    return input * torch.sigmoid(input * alpha)


def _torch_gelu_grad(dy, x):
    alpha = 0.70710678118654752440
    beta = 1.12837916709551257390 * 0.70710678118654752440 * 0.5
    cdf = 0.5 * (1 + torch.erf(x * alpha))
    pdf = beta * torch.exp(x * x * -0.5)
    return dy * (cdf + x * pdf)


def _torch_quick_gelu_grad(dy, x, **kwargs):
    alpha = kwargs.get("alpha", 1.702)
    sigmoid = torch.sigmoid(x * alpha)
    return dy * sigmoid * (1.0 + x * alpha * (1.0 - sigmoid))


class TorchFuncExecutor:
    _TORCH_FUNC_MAP = {  # noqa: RUF012
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
        "Gelu": _torch_gelu,
        "QuickGelu": _torch_quick_gelu,
        "GeluGrad": _torch_gelu_grad,
        "QuickGeluGrad": _torch_quick_gelu_grad,
    }

    @classmethod
    def run(cls, op_type, *torch_tensors, **kwargs):
        if op_type not in cls._TORCH_FUNC_MAP:
            raise NotImplementedError(f"Unsupported op type: {op_type}")
        return cls._TORCH_FUNC_MAP[op_type](*torch_tensors, **kwargs)


def _run_op_test(op_type, onnx_dtype, create_model_func, gen_inputs_func, **kwargs):
    rtol = kwargs.get("rtol", 1e-03 if onnx_dtype == TensorProto.FLOAT16 else 1e-04)
    atol = kwargs.get("atol", 1e-03 if onnx_dtype == TensorProto.FLOAT16 else 1e-05)
    pt_inputs = gen_inputs_func(_onnx_dtype_to_torch_dtype(onnx_dtype))
    ort_inputs = copy.deepcopy(pt_inputs)
    ort_inputs = [tensor.to(torch.uint8) if tensor.dtype == torch.bool else tensor for tensor in ort_inputs]
    if "::" in op_type:
        _, op_type = op_type.split("::")
    pt_outputs = TorchFuncExecutor.run(op_type, *pt_inputs, **kwargs)
    model_str = create_model_func(op_type, onnx_dtype, **kwargs).SerializeToString()
    unique_id = uuid.uuid1().int >> 64
    ort_outputs = call_triton_by_onnx(unique_id, model_str, *[to_dlpack(tensor) for tensor in ort_inputs])
    if isinstance(pt_outputs, tuple):
        assert isinstance(ort_outputs, tuple)
        assert len(pt_outputs) == len(ort_outputs)
        for pt_output, ort_output in zip(pt_outputs, ort_outputs):
            _test_helpers.assert_values_are_close(pt_output, _from_dlpack(ort_output), rtol=rtol, atol=atol)
    else:
        _test_helpers.assert_values_are_close(pt_outputs, _from_dlpack(ort_outputs), rtol=rtol, atol=atol)


def _run_step(model, *tensors):
    prediction = model(*tensors)
    loss = prediction.sum()
    loss.backward()
    return prediction


def _run_module_test(module_cls, dtype, gen_inputs_func, triton_op_count, **kwargs):
    pt_model = module_cls().to(DEVICE).to(dtype)
    ort_model = ORTModule(copy.deepcopy(pt_model), DebugOptions(save_onnx=True, onnx_prefix="triton_model"))
    rtol = kwargs.get("rtol", 1e-03 if dtype == torch.float16 else 1e-04)
    atol = kwargs.get("atol", 1e-03 if dtype == torch.float16 else 1e-05)
    for _ in range(10):
        pt_inputs = gen_inputs_func(dtype)
        ort_inputs = copy.deepcopy(pt_inputs)
        pt_output = _run_step(pt_model, *pt_inputs)
        ort_output = _run_step(ort_model, *ort_inputs)
        _test_helpers.assert_values_are_close(pt_output, ort_output, rtol=rtol, atol=atol)
        _test_helpers.assert_gradients_match_and_reset_gradient(pt_model, ort_model, rtol=rtol, atol=atol)
        for idx, pt_input in enumerate(pt_inputs):
            if pt_input.requires_grad:
                _test_helpers.assert_values_are_close(pt_input.grad, ort_inputs[idx].grad, rtol=rtol, atol=atol)

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


def _run_tunable_op_test(module_cls, dtype, gen_inputs_func, tunable_op, impl_count, **kwargs):
    os.environ["ORTMODULE_ENABLE_TUNING"] = "1"
    os.environ["ORTMODULE_TUNING_RESULTS_PATH"] = "./"
    pt_model = module_cls().to(DEVICE).to(dtype)
    ort_model = ORTModule(copy.deepcopy(pt_model))
    rtol = kwargs.get("rtol", 1e-03 if dtype == torch.float16 else 1e-04)
    atol = kwargs.get("atol", 1e-03 if dtype == torch.float16 else 1e-05)
    for _ in range(5):
        pt_inputs = gen_inputs_func(dtype)
        ort_inputs = copy.deepcopy(pt_inputs)
        pt_output = _run_step(pt_model, *pt_inputs)
        ort_output = _run_step(ort_model, *ort_inputs)
        _test_helpers.assert_values_are_close(pt_output, ort_output, rtol=rtol, atol=atol)
        _test_helpers.assert_gradients_match_and_reset_gradient(pt_model, ort_model, rtol=rtol, atol=atol)
    tunable_results_file = os.path.join(os.getcwd(), "tuning_results_training.json")
    assert os.path.exists(tunable_results_file)
    with open(tunable_results_file, encoding="UTF-8") as f:
        tunable_results = json.load(f)
    assert tunable_op in str(tunable_results)
    del os.environ["ORTMODULE_ENABLE_TUNING"]
    for i in range(impl_count - 1):
        new_tunable_results = copy.deepcopy(tunable_results)
        for k, v in new_tunable_results[0]["results"].items():
            if tunable_op in k:
                for param, impl in v.items():
                    v[param] = (impl + 1 + i) % impl_count
        with open(tunable_results_file, "w", encoding="UTF-8") as f:
            json.dump(new_tunable_results, f)
        ort_model = ORTModule(copy.deepcopy(pt_model))
        for _ in range(5):
            pt_inputs = gen_inputs_func(dtype)
            ort_inputs = copy.deepcopy(pt_inputs)
            pt_output = _run_step(pt_model, *pt_inputs)
            ort_output = _run_step(ort_model, *ort_inputs)
            _test_helpers.assert_values_are_close(pt_output, ort_output, rtol=rtol, atol=atol)
            _test_helpers.assert_gradients_match_and_reset_gradient(pt_model, ort_model, rtol=rtol, atol=atol)
    os.remove(tunable_results_file)
    del os.environ["ORTMODULE_TUNING_RESULTS_PATH"]


@pytest.mark.parametrize(
    "op",
    [
        ("Add", {}),
        ("Sub", {}),
        ("Mul", {}),
        ("Div", {}),
        ("com.microsoft::GeluGrad", {}),
        ("com.microsoft::QuickGeluGrad", {}),
        ("com.microsoft::QuickGeluGrad", {"alpha": 1.0}),
    ],
)
@pytest.mark.parametrize("onnx_dtype", [TensorProto.FLOAT, TensorProto.FLOAT16])
@pytest.mark.parametrize("input_shapes", [([1024, 2], [1024, 2]), ([2, 3, 3, 3], [3, 1, 3]), ([2049], [1])])
def test_binary_elementwise_op(op, onnx_dtype, input_shapes):
    def _create_model(op_type, onnx_dtype, **kwargs):
        domain = ""
        if "::" in op_type:
            domain, op_type = op_type.split("::")
        graph = helper.make_graph(
            [helper.make_node(op_type, ["X", "Y"], ["Z"], name="test", domain=domain, **kwargs)],
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

    _run_op_test(op[0], onnx_dtype, _create_model, _gen_inputs, **op[1])


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


@pytest.mark.parametrize(
    "op",
    [
        ("Sqrt", {}),
        ("Exp", {}),
        ("com.microsoft::Gelu", {}),
        ("com.microsoft::QuickGelu", {}),
        ("com.microsoft::QuickGelu", {"alpha": 1.0}),
    ],
)
@pytest.mark.parametrize("onnx_dtype", [TensorProto.FLOAT, TensorProto.FLOAT16])
@pytest.mark.parametrize("input_shape", [[1024, 4], [2, 3, 3, 3], [2049, 1]])
def test_unary_elementwise_op(op, onnx_dtype, input_shape):
    def _create_model(op_type, onnx_dtype, **kwargs):
        domain = ""
        if "::" in op_type:
            domain, op_type = op_type.split("::")
        graph = helper.make_graph(
            [helper.make_node(op_type, ["X"], ["Y"], name="test", domain=domain, **kwargs)],
            "test",
            [helper.make_tensor_value_info("X", onnx_dtype, None)],
            [helper.make_tensor_value_info("Y", onnx_dtype, None)],
        )
        return helper.make_model(graph, producer_name="test")

    def _gen_inputs(dtype):
        return [torch.rand(*input_shape, dtype=dtype, device=DEVICE)]

    _run_op_test(op[0], onnx_dtype, _create_model, _gen_inputs, **op[1])


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
    outputs = call_triton_by_onnx(hash(model_str), model_str, *[to_dlpack(t) for t in input_tensor])
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


@pytest.mark.parametrize("op_type", ["ReduceSum", "ReduceMean", "ReduceMax", "ReduceMin"])
@pytest.mark.parametrize("onnx_dtype", [TensorProto.FLOAT, TensorProto.FLOAT16])
@pytest.mark.parametrize(
    "input_shape_and_reduce_info",
    [
        ([2, 1024], [-1], True),
        ([1050, 3], [0], False),
        ([2, 3, 3, 3], [1, 2], True),
        ([123, 4, 5, 6], [2], False),
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
    if onnx_dtype == TensorProto.FLOAT16:
        kwargs["atol"] = 1e-2
        kwargs["rtol"] = 1e-2
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


@pytest.mark.parametrize("dtype", [torch.float, torch.float16])
@pytest.mark.parametrize(
    "input_info",
    [
        ([32, 64], False, [64, 16], False, 1.0),
        ([33, 68], False, [18, 68], True, 0.5),
        ([128, 64], True, [128, 32], False, 0.5),
        ([123, 234], True, [345, 123], True, -1.0),
        ([22, 33, 44], False, [44, 55], False, 1.0),
        ([22, 33, 44], False, [666, 44], True, 0.2),
        ([22, 33, 44], True, [33, 666], False, -0.2),
        ([64, 128], False, [16, 64, 128], True, 0.5),
        ([16, 32, 64], False, [16, 64, 32], False, 1.0),
        ([8, 16, 32, 16], True, [8, 16, 32, 32], True, 1.0),
    ],
)
def test_matmul(dtype, input_info):
    pt_inputs = [
        torch.rand(*input_info[0], dtype=dtype, device=DEVICE),
        torch.rand(*input_info[2], dtype=dtype, device=DEVICE),
    ]
    ort_inputs = copy.deepcopy(pt_inputs)
    kwargs = {}
    if input_info[1]:
        pt_inputs[0] = pt_inputs[0].transpose(-1, -2)
        kwargs["trans_a"] = True
    if input_info[3]:
        pt_inputs[1] = pt_inputs[1].transpose(-1, -2)
        kwargs["trans_b"] = True
    if input_info[4] != 1.0:
        kwargs["alpha"] = input_info[4]
    pt_output = torch.matmul(*pt_inputs) * input_info[4]
    alloc_out = random.choice([True, False])
    if alloc_out:
        ort_output = torch.empty(pt_output.shape, dtype=dtype, device=DEVICE)
        ort_inputs.append(ort_output)
        call_triton_by_name("triton_matmul_out", *[to_dlpack(tensor) for tensor in ort_inputs], **kwargs)
    else:
        ort_output = _from_dlpack(
            call_triton_by_name("triton_matmul", *[to_dlpack(tensor) for tensor in ort_inputs], **kwargs)
        )
    rtol = 1e-02 if dtype == torch.float16 else 1e-04
    atol = 1e-02 if dtype == torch.float16 else 1e-05
    _test_helpers.assert_values_are_close(pt_output, ort_output, rtol=rtol, atol=atol)


@pytest.mark.parametrize("dtype", [torch.float, torch.float16])
@pytest.mark.parametrize(
    "input_info",
    [
        ([64, 32], False, [32, 64], False, [64, 64], 1.0, 0.0),
        ([65, 129], False, [65, 129], True, [65, 1], 0.5, -0.5),
        ([127, 63], True, [127, 127], False, [127], -1.0, 0.2),
        ([256, 64], True, [128, 256], True, [1], 0.2, 0.5),
    ],
)
def test_gemm(dtype, input_info):
    pt_inputs = [
        torch.rand(*input_info[0], dtype=dtype, device=DEVICE),
        torch.rand(*input_info[2], dtype=dtype, device=DEVICE),
        torch.rand(*input_info[4], dtype=dtype, device=DEVICE),
    ]
    ort_inputs = copy.deepcopy(pt_inputs)
    kwargs = {}
    if input_info[1]:
        pt_inputs[0] = pt_inputs[0].transpose(-1, -2)
        kwargs["trans_a"] = True
    if input_info[3]:
        pt_inputs[1] = pt_inputs[1].transpose(-1, -2)
        kwargs["trans_b"] = True
    if input_info[5] != 1.0:
        kwargs["alpha"] = input_info[5]
    if input_info[6] != 1.0:
        kwargs["beta"] = input_info[6]
    pt_output = torch.matmul(pt_inputs[0], pt_inputs[1]) * input_info[5] + pt_inputs[2] * input_info[6]
    alloc_out = random.choice([True, False])
    if alloc_out:
        ort_output = torch.empty(pt_output.shape, dtype=dtype, device=DEVICE)
        ort_inputs.append(ort_output)
        call_triton_by_name("triton_gemm_out", *[to_dlpack(tensor) for tensor in ort_inputs], **kwargs)
    else:
        ort_output = _from_dlpack(
            call_triton_by_name("triton_gemm", *[to_dlpack(tensor) for tensor in ort_inputs], **kwargs)
        )
    rtol = 1e-02 if dtype == torch.float16 else 1e-04
    atol = 1e-02 if dtype == torch.float16 else 1e-05
    _test_helpers.assert_values_are_close(pt_output, ort_output, rtol=rtol, atol=atol)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_elementwise_module(dtype):
    n, d, h, w = 8, 768, 12, 64

    class NeuralNetElementwise(torch.nn.Module):
        def forward(self, input1, input2, input3, input4):
            return input1 + input2 - input3 * input4

    def _gen_inputs(dtype):
        return [
            torch.rand(n, d, h, w, dtype=dtype, device=DEVICE, requires_grad=True),
            torch.rand(w, dtype=dtype, device=DEVICE, requires_grad=True),
            torch.rand(d, 1, 1, dtype=dtype, device=DEVICE, requires_grad=True),
            torch.rand(n, 1, h, w, dtype=dtype, device=DEVICE, requires_grad=True),
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
@pytest.mark.parametrize(
    "input_shapes_and_axis", [([2, 1024], [2, 1024], -1), ([2, 2049], [2, 1], -1), ([2, 3, 3, 3], [3, 3], 2)]
)
def test_layer_norm_module(dtype, input_shapes_and_axis):
    pytest.skip("LayerNorm is disabled for now due to perf issue.")

    class NeuralNetLayerNorm(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer_norm = torch.nn.LayerNorm(
                input_shapes_and_axis[0][input_shapes_and_axis[2] :], device=DEVICE, dtype=dtype
            )

        def forward(self, input1, input2):
            return self.layer_norm(input1 * input2)

    def _gen_inputs(dtype):
        return [
            torch.randn(*input_shapes_and_axis[0], dtype=dtype, device=DEVICE, requires_grad=True),
            torch.randn(*input_shapes_and_axis[1], dtype=dtype, device=DEVICE, requires_grad=True),
        ]

    _run_module_test(NeuralNetLayerNorm, dtype, _gen_inputs, 2)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_dynamic_shapes_elementwise_module(dtype):
    class NeuralNetSymbolicShapesElementwise(torch.nn.Module):
        def forward(self, x, y, u, v):
            return x * y - (u + v)

    def _gen_inputs(dtype):
        dim1 = 64 * random.randint(2, 4)
        dim2 = 64 * random.randint(2, 4)
        return [
            torch.rand(16, dim1, dim2, dtype=dtype, device=DEVICE, requires_grad=True),
            torch.rand(16, 1, dim2, dtype=dtype, device=DEVICE, requires_grad=True),
            torch.rand(dim1, 1, dtype=dtype, device=DEVICE, requires_grad=True),
            torch.rand(16, dim1, dim2, dtype=dtype, device=DEVICE, requires_grad=True),
        ]

    _run_module_test(NeuralNetSymbolicShapesElementwise, dtype, _gen_inputs, 1)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_dynamic_shapes_reduction_module(dtype):
    class NeuralNetSymbolicShapesReduction(torch.nn.Module):
        def forward(self, x, y, z):
            return torch.softmax(x * y + z, dim=-1)

    def _gen_inputs(dtype):
        dim1 = 64 * random.randint(2, 4)
        dim2 = 64 * random.randint(2, 4)
        return [
            torch.rand(16, dim1, dim2, dtype=dtype, device=DEVICE, requires_grad=True),
            torch.rand(16, 1, dim2, dtype=dtype, device=DEVICE, requires_grad=True),
            torch.rand(dim1, 1, dtype=dtype, device=DEVICE, requires_grad=True),
        ]

    _run_module_test(NeuralNetSymbolicShapesReduction, dtype, _gen_inputs, 2)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("has_sum", [True, False])
def test_slice_scel_module(dtype, has_sum):
    class NeuralNetSliceScel(torch.nn.Module):
        def forward(self, logits, labels):
            shift_logits = logits[..., :-1, :].contiguous()
            labels = labels[..., 1:].contiguous()
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), labels.view(-1))
            return logits + loss if has_sum else loss

    def _gen_inputs(dtype):
        return [
            (torch.rand(4, 8, 16) * 0.01).to(dtype=dtype, device=DEVICE).requires_grad_(True),
            torch.randint(0, 16, (4, 8), dtype=torch.int64, device=DEVICE),
        ]

    _run_module_test(NeuralNetSliceScel, dtype, _gen_inputs, 2)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("input_shapes", [([128, 64], [64, 64]), ([16, 64, 128], [16, 128, 64])])
def test_matmul_tunable_op(dtype, input_shapes):
    class NeuralNetMatmul(torch.nn.Module):
        def forward(self, input1, input2):
            return torch.matmul(input1, input2)

    def _gen_inputs(dtype):
        return [
            torch.rand(*input_shapes[0], dtype=dtype, device=DEVICE, requires_grad=True),
            torch.rand(*input_shapes[1], dtype=dtype, device=DEVICE, requires_grad=True),
        ]

    _run_tunable_op_test(NeuralNetMatmul, dtype, _gen_inputs, "MatMulTunableOp", 2)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("m_n_k", [(64, 64, 64)])
def test_gemm_tunable_op(dtype, m_n_k):
    class NeuralNetGemm(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(m_n_k[2], m_n_k[1])

        def forward(self, input):
            return self.linear(input)

    def _gen_inputs(dtype):
        return [torch.rand(m_n_k[0], m_n_k[2], dtype=dtype, device=DEVICE, requires_grad=True)]

    _run_tunable_op_test(NeuralNetGemm, dtype, _gen_inputs, "GemmTunableOp", 2)
