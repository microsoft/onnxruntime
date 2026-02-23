# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import platform
from itertools import chain
from pathlib import Path
from unittest.mock import patch

import onnx
import pytest
import torch
from onnxscript import ir
from packaging import version

from olive.common.config_utils import validate_config
from olive.model import PyTorchModelHandler
from olive.model.config import IoConfig
from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.onnx.conversion import OnnxConversion, OnnxOpVersionConversion
from olive.passes.pytorch.autogptq import GptqQuantizer
from olive.passes.pytorch.rtn import Rtn
from test.utils import (
    ONNX_MODEL_PATH,
    get_hf_model,
    get_onnx_model,
    get_pytorch_model,
    get_tiny_phi3,
    pytorch_model_loader,
)


def _torch_is_older_than(version_str: str) -> bool:
    torch_version = version.parse(torch.__version__).release
    return torch_version < version.parse(version_str).release


@pytest.mark.parametrize(
    ("input_model", "use_dynamo_exporter", "dynamic"),
    [
        (get_hf_model(), True, True),
        (get_hf_model(), True, False),
        (get_pytorch_model(), True, True),
        (get_pytorch_model(), True, False),
    ],
)
def test_onnx_conversion_pass_with_exporters(input_model, use_dynamo_exporter: bool, dynamic: bool, tmp_path):
    # setup
    p = create_pass_from_dict(
        OnnxConversion, {"use_dynamo_exporter": use_dynamo_exporter, "dynamic": dynamic}, disable_search=True
    )
    output_folder = str(tmp_path / "onnx")
    onnx_model = p.run(input_model, output_folder)

    assert Path(onnx_model.model_path).exists()


@pytest.mark.parametrize("quantizer_pass", [Rtn, GptqQuantizer])
@pytest.mark.parametrize("use_dynamo_exporter", [True])
def test_onnx_conversion_pass_quant_model(quantizer_pass, use_dynamo_exporter: bool, tmp_path):
    if use_dynamo_exporter and platform.system() == "Windows":
        pytest.skip("FIXME: torch ops fails on Windows")

    if quantizer_pass == GptqQuantizer and not torch.cuda.is_available():
        pytest.skip("GptqQuantizer requires CUDA")

    if use_dynamo_exporter and version.parse(torch.__version__) != version.parse("2.8.0"):
        pytest.skip("Dynamo export requires 2.8. 2.9+ has issues with older transformers versions.")

    # setup
    base_model = get_tiny_phi3()
    pass_config = {"group_size": 16}
    if quantizer_pass == Rtn:
        pass_config["lm_head"] = True
        pass_config["embeds"] = True
    quantizer_pass = create_pass_from_dict(quantizer_pass, pass_config, disable_search=True)
    quantized_model = quantizer_pass.run(base_model, str(tmp_path / "quantized"))

    p = create_pass_from_dict(
        OnnxConversion, {"torch_dtype": "float32", "use_dynamo_exporter": use_dynamo_exporter}, disable_search=True
    )
    output_folder = str(tmp_path / "onnx")

    # run
    onnx_model = p.run(quantized_model, output_folder)

    # assert
    assert Path(onnx_model.model_path).exists()
    model_ir = ir.load(onnx_model.model_path)
    num_mnb = sum(node.op_type == "MatMulNBits" for node in model_ir.graph)
    # 2 layers X 1 qkv, 1 o, 1 gate_up, 1 down
    expected_num_mnb = 2 * 4
    if pass_config.get("lm_head", False):
        expected_num_mnb += 1
    assert num_mnb == expected_num_mnb

    num_gbq = sum(node.op_type == "GatherBlockQuantized" for node in model_ir.graph)
    expected_num_gbq = 1 if pass_config.get("embeds", False) else 0
    assert num_gbq == expected_num_gbq


@pytest.mark.parametrize("target_opset", [16, 17, 18])
def test_onnx_op_version_conversion_pass(target_opset, tmp_path):
    # Note: The test ONNX model is created with dynamo export at opset 20.
    # ONNX version converter cannot downgrade Gemm from opset 13+ to opset 9/10,
    # so we only test conversion to opset 16+.
    input_model = get_onnx_model()
    # setup
    p = create_pass_from_dict(
        OnnxOpVersionConversion,
        {"target_opset": target_opset},
        disable_search=True,
    )
    output_folder = str(tmp_path / "onnx")

    onnx_model = p.run(input_model, output_folder)

    # assert
    assert onnx_model.load_model().opset_import[0].version == target_opset


def get_io_config_phi2(model):
    input_names = [
        "input_ids",
        "attention_mask",
        *list(chain.from_iterable((f"past_key_values.{i}",) for i in range(32))),
    ]
    output_names = [
        "logits",
        *list(chain.from_iterable((f"present_key_values.{i}",) for i in range(32))),
    ]
    return {
        "input_names": input_names,
        "output_names": output_names,
    }


def get_dummy_inputs_phi2(model):
    def get_past_kv_inputs(batch_size: int, past_seq_len: int):
        num_heads, head_size = 31, 80
        torch_dtype = torch.float32
        return [(torch.rand(batch_size, past_seq_len, 1, num_heads, head_size, dtype=torch_dtype),) for _ in range(32)]

    input_ids = torch.randint(low=0, high=51200, size=(2, 8), dtype=torch.int64)
    attention_mask = torch.ones(2, 16, dtype=torch.int64)
    past_key_values = get_past_kv_inputs(2, 16)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "past_key_values": past_key_values,
    }


def get_io_config_llama2(model):
    input_names = [
        "input_ids",
        "attention_mask",
        "position_ids",
        *list(chain.from_iterable((f"past_key_values.{i}.key", f"past_key_values.{i}.value") for i in range(32))),
    ]
    output_names = [
        "logits",
        *list(chain.from_iterable((f"present.{i}.key", f"present.{i}.value") for i in range(32))),
    ]
    return {
        "input_names": input_names,
        "output_names": output_names,
    }


def get_dummy_inputs_llama2(_):
    def get_past_kv_inputs(batch_size: int, past_seq_len: int):
        num_heads = 32
        head_size = 80
        torch_dtype = torch.float32
        return [
            (
                torch.rand(batch_size, num_heads, past_seq_len, head_size, dtype=torch_dtype),
                torch.rand(batch_size, num_heads, past_seq_len, head_size, dtype=torch_dtype),
            )
            for _ in range(32)
        ]

    input_ids = torch.randint(low=0, high=51200, size=(2, 8), dtype=torch.int64)
    attention_mask = torch.ones(2, 16, dtype=torch.int64)
    past_key_values = get_past_kv_inputs(2, 16)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "past_key_values": past_key_values,
    }


@pytest.mark.parametrize(
    ("io_config_func", "dummy_inputs_func"),
    [
        (get_io_config_llama2, get_dummy_inputs_llama2),
        (get_io_config_phi2, get_dummy_inputs_phi2),
    ],
)
@patch("torch.onnx.export")
def test_onnx_conversion_with_past_key_values(mock_onnx_export, tmp_path, io_config_func, dummy_inputs_func):
    dummy_kwargs = None

    class MockOnnxProgram:
        def __init__(self, model_path):
            self.model = ir.serde.deserialize_model(onnx.load(model_path))

    def mock_onnx_export_func(*args, **kwargs):
        nonlocal dummy_kwargs
        # For dynamo export, inputs are passed via kwargs parameter
        dummy_kwargs = kwargs.get("kwargs", {})
        return MockOnnxProgram(ONNX_MODEL_PATH)

    output_folder = tmp_path / "onnx"
    output_folder.mkdir(parents=True, exist_ok=True)
    input_model = PyTorchModelHandler(
        model_loader=pytorch_model_loader,
        model_path=None,
        io_config=io_config_func,
        dummy_inputs_func=dummy_inputs_func,
    )
    mock_onnx_export.side_effect = mock_onnx_export_func
    # setup
    p = create_pass_from_dict(OnnxConversion, {"use_dynamo_exporter": True}, disable_search=True)
    _ = p.run(input_model, str(output_folder))
    assert "past_key_values" in dummy_kwargs  # pylint: disable=unsupported-membership-test


@pytest.mark.parametrize(
    "dynamic_shapes",
    [
        [{0: "axis_batch", 1: "x_axis"}, {0: "axis_batch", 1: "y_axis"}],
        {
            "input_x": {0: "axis_batch", 1: "x_axis"},
            "input_y": {0: "axis_batch", 1: "y_axis"},
        },
    ],
)
def test_dynamic_shapes_passes_validate_io_config_with_both_list_and_dict_format(dynamic_shapes):
    config = {"input_names": ["input_x", "input_y"], "output_names": ["logits"]}
    config["dynamic_shapes"] = dynamic_shapes
    io_config = validate_config(config, IoConfig)
    assert io_config.dynamic_shapes == dynamic_shapes


def _get_simulate_torch_float_tensor_inputs(return_tuple: bool = False):
    if return_tuple:
        return (
            torch.ones(5),
            (torch.zeros(5), torch.ones(5)),
            {"a": torch.zeros(5), "b": torch.ones(5)},
            torch.ones(4),
        )
    return {
        "y": {"a": torch.zeros(5), "b": torch.ones(5)},
        "w": torch.ones(5),
        "x": (torch.zeros(5), torch.ones(5)),
        "z": torch.ones(4),
    }


class SingnatureOnlyModel(torch.nn.Module):
    def forward(
        self,
        w: torch.Tensor,
        x: tuple[torch.Tensor, torch.Tensor],
        y: dict[str, torch.Tensor],
        z: torch.Tensor,
    ):
        pass


@pytest.mark.parametrize(
    ("dynamic_shapes", "expected_dynamic_shapes", "inputs"),
    [
        (
            [
                {0: "axis_batch", 1: "x_axis"},
                [{1: "x_axis"}, {0: "axis_batch"}],
                {"a": {0: "axis_batch"}, "b": {1: "x_axis"}},
                None,
            ],
            [
                {0: "axis_batch", 1: "x_axis"},
                ({1: "x_axis"}, {0: "axis_batch"}),
                {"a": {0: "axis_batch"}, "b": {1: "x_axis"}},
                None,
            ],
            _get_simulate_torch_float_tensor_inputs(return_tuple=True),
        ),
        (
            # We mess up the order of inputs and dynamic shapes from the model signature
            # to test that the validation can order it back.
            {
                "y": {"a": {0: "axis_batch"}, "b": {1: "x_axis"}},
                "w": {0: "axis_batch", 1: "x_axis"},
                "x": [{1: "x_axis"}, {0: "axis_batch"}],
                "z": None,
            },
            {
                "w": {0: "axis_batch", 1: "x_axis"},
                "x": ({1: "x_axis"}, {0: "axis_batch"}),
                "y": {"a": {0: "axis_batch"}, "b": {1: "x_axis"}},
                "z": None,
            },
            _get_simulate_torch_float_tensor_inputs(return_tuple=False),
        ),
    ],
    ids=["in_nested_tuple_inputs", "in_nested_dict_format"],
)
def test___validate_dynamic_shapes_follow_input_format_and_follow_order_of_model_sig(
    dynamic_shapes, expected_dynamic_shapes, inputs
):
    from olive.passes.onnx.conversion import _validate_dynamic_shapes

    if isinstance(dynamic_shapes, (tuple, list)):
        converted_dynamic_shapes, _, _ = _validate_dynamic_shapes(dynamic_shapes, inputs, {}, SingnatureOnlyModel())
    else:
        converted_dynamic_shapes, _, _ = _validate_dynamic_shapes(dynamic_shapes, (), inputs, SingnatureOnlyModel())
    assert converted_dynamic_shapes == expected_dynamic_shapes
