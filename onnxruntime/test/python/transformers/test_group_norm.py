# --------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# -------------------------------------------------------------------------
import statistics
from dataclasses import dataclass
from enum import Enum
from time import perf_counter
from typing import Optional, Tuple

import numpy
import torch
from onnx import TensorProto, helper

from onnxruntime import InferenceSession
from onnxruntime.transformers.io_binding_helper import CudaSession

torch.manual_seed(0)


class GroupNormOpType(Enum):
    GROUP_NORM = 1
    SKIP_GROUP_NORM = 2


@dataclass
class GroupNormConfig:
    batch_size: int
    height: int
    width: int
    channels: int
    epsilon: float = 1e-5
    num_groups: int = 32
    activation: bool = False
    channels_last: bool = True
    fp16: bool = False

    op_type: GroupNormOpType = GroupNormOpType.GROUP_NORM
    has_bias: bool = False
    has_add_out: bool = False
    broadcast_skip: int = 0  # 2 for (N, C), 4 for (N, 1, 1, C)

    def get_skip_symbolic_shape(self):
        skip_shape = {0: ["N", "H", "W", "C"], 2: ["N", "C"], 4: ["N", 1, 1, "C"]}
        return skip_shape[self.broadcast_skip]

    def get_skip_shape(self):
        skip_shape = {
            0: [self.batch_size, self.height, self.width, self.channels],
            2: [self.batch_size, self.channels],
            4: [self.batch_size, 1, 1, self.channels],
        }
        return skip_shape[self.broadcast_skip]

    @staticmethod
    def create(
        b: int,
        h: int,
        w: int,
        c: int,
        fp16: bool = False,
        activation: bool = False,
        template: int = 0,
        num_groups: int = 32,
    ):
        if template == 0:
            return GroupNormConfig(
                b, h, w, c, fp16=fp16, activation=activation, op_type=GroupNormOpType.GROUP_NORM, num_groups=num_groups
            )

        if template == 1:
            return GroupNormConfig(
                b,
                h,
                w,
                c,
                fp16=fp16,
                activation=activation,
                op_type=GroupNormOpType.SKIP_GROUP_NORM,
                has_bias=True,
                has_add_out=True,
                broadcast_skip=0,
                num_groups=num_groups,
            )

        if template == 2:
            return GroupNormConfig(
                b,
                h,
                w,
                c,
                fp16=fp16,
                activation=activation,
                op_type=GroupNormOpType.SKIP_GROUP_NORM,
                has_bias=True,
                has_add_out=True,
                broadcast_skip=2,
                num_groups=num_groups,
            )

        if template == 3:
            return GroupNormConfig(
                b,
                h,
                w,
                c,
                fp16=fp16,
                activation=activation,
                op_type=GroupNormOpType.SKIP_GROUP_NORM,
                has_bias=True,
                has_add_out=True,
                broadcast_skip=4,
                num_groups=num_groups,
            )

        if template == 4:  # No bias
            return GroupNormConfig(
                b,
                h,
                w,
                c,
                fp16=fp16,
                activation=activation,
                op_type=GroupNormOpType.SKIP_GROUP_NORM,
                has_bias=False,
                has_add_out=True,
                broadcast_skip=0,
                num_groups=num_groups,
            )

        if template == 5:  # No bias, no add_out
            return GroupNormConfig(
                b,
                h,
                w,
                c,
                fp16=fp16,
                activation=activation,
                op_type=GroupNormOpType.SKIP_GROUP_NORM,
                has_bias=False,
                has_add_out=False,
                broadcast_skip=0,
                num_groups=num_groups,
            )


def create_group_norm_graph(config: GroupNormConfig) -> bytes:
    inputs = ["input", "gamma", "beta"]
    outputs = ["output"]
    op_type = "GroupNorm"
    if config.op_type == GroupNormOpType.SKIP_GROUP_NORM:
        op_type = "SkipGroupNorm"
        inputs = [*inputs, "skip"]
        if config.has_bias:
            inputs = [*inputs, "bias"]
        if config.has_add_out:
            outputs = [*outputs, "add_out"]

    nodes = [
        helper.make_node(
            op_type,
            inputs,
            outputs,
            op_type + "_0",
            activation=int(config.activation),
            channels_last=int(config.channels_last),
            epsilon=config.epsilon,
            groups=config.num_groups,
            domain="com.microsoft",
        ),
    ]

    float_type = TensorProto.FLOAT16 if config.fp16 else TensorProto.FLOAT

    input_shapes = [
        helper.make_tensor_value_info("input", float_type, ["N", "H", "W", "C"]),
        helper.make_tensor_value_info("gamma", TensorProto.FLOAT, ["C"]),
        helper.make_tensor_value_info("beta", TensorProto.FLOAT, ["C"]),
    ]
    output_shapes = [
        helper.make_tensor_value_info("output", float_type, ["N", "H", "W", "C"]),
    ]

    if config.op_type == GroupNormOpType.SKIP_GROUP_NORM:
        input_shapes = [
            *input_shapes,
            helper.make_tensor_value_info("skip", float_type, config.get_skip_symbolic_shape()),
        ]
        if config.has_bias:
            input_shapes = [*input_shapes, helper.make_tensor_value_info("bias", float_type, ["C"])]
        if config.has_add_out:
            output_shapes = [*output_shapes, helper.make_tensor_value_info("add_out", float_type, ["N", "H", "W", "C"])]

    graph = helper.make_graph(
        nodes,
        "Group_Norm_Graph",
        input_shapes,
        output_shapes,
    )

    model = helper.make_model(graph)
    return model.SerializeToString()


def group_norm_ort(
    src: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    skip: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
    config: GroupNormConfig,
    measure_latency=False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[float]]:
    onnx_model_str = create_group_norm_graph(config)
    ort_session = InferenceSession(onnx_model_str, providers=["CUDAExecutionProvider"])

    session = CudaSession(ort_session, device=torch.device("cuda:0"))

    io_shape = {
        "input": [config.batch_size, config.height, config.width, config.channels],
        "gamma": [config.channels],
        "beta": [config.channels],
        "output": [config.batch_size, config.height, config.width, config.channels],
    }

    if config.op_type == GroupNormOpType.SKIP_GROUP_NORM:
        io_shape["skip"] = config.get_skip_shape()
        if config.has_bias:
            io_shape["bias"] = [config.channels]
        if config.has_add_out:
            io_shape["add_out"] = [config.batch_size, config.height, config.width, config.channels]

    session.allocate_buffers(io_shape)

    ort_inputs = {
        "input": src,
        "gamma": gamma,
        "beta": beta,
    }

    if config.op_type == GroupNormOpType.SKIP_GROUP_NORM:
        ort_inputs["skip"] = skip
        if config.has_bias:
            ort_inputs["bias"] = bias

    ort_outputs = session.infer(ort_inputs)
    output = ort_outputs["output"]

    add_out = (
        ort_outputs["add_out"] if config.op_type == GroupNormOpType.SKIP_GROUP_NORM and config.has_add_out else None
    )

    if measure_latency:
        latency_list = []
        for _ in range(10000):
            start_time = perf_counter()
            session.infer(ort_inputs)
            end_time = perf_counter()
            latency_list.append(end_time - start_time)
        average_latency = statistics.mean(latency_list)
        return output, add_out, average_latency

    return output, add_out, None


def group_norm_torch(
    src: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    skip: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
    config: GroupNormConfig,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    add_out = src
    if skip is not None:
        add_out = add_out + skip

    if bias is not None:
        if config.op_type == GroupNormOpType.SKIP_GROUP_NORM:
            add_out = add_out + bias.reshape(1, 1, 1, bias.shape[0])
        else:
            add_out = add_out + bias.reshape(bias.shape[0], 1, 1, bias.shape[1])

    x = add_out
    if config.channels_last:
        x = add_out.clone().permute(0, 3, 1, 2)  # from NHWC to NCHW

    weight = gamma.to(x.dtype)
    bias = beta.to(x.dtype)
    output = torch.nn.functional.group_norm(x, config.num_groups, weight=weight, bias=bias, eps=config.epsilon)

    if config.activation:
        torch.nn.functional.silu(output, inplace=True)

    if config.channels_last:
        output = output.permute(0, 2, 3, 1)  # from NCHW to NHWC

    return output, add_out


def run_parity(config, measure_latency=True):
    float_type = torch.float16 if config.fp16 else torch.float32

    input_tensor = torch.randn(
        config.batch_size,
        config.height,
        config.width,
        config.channels,
        device="cuda",
        dtype=float_type,
        requires_grad=False,
    )

    gamma = torch.randn(
        config.channels,
        device="cuda",
        dtype=torch.float32,
        requires_grad=False,
    )

    beta = torch.randn(
        config.channels,
        device="cuda",
        dtype=torch.float32,
        requires_grad=False,
    )

    skip = None
    bias = None
    if config.op_type == GroupNormOpType.SKIP_GROUP_NORM:
        skip = torch.randn(
            *config.get_skip_shape(),
            device="cuda",
            dtype=float_type,
            requires_grad=False,
        )
        if config.has_bias:
            bias = torch.randn(
                config.channels,
                device="cuda",
                dtype=float_type,
                requires_grad=False,
            )

    out_ort, ort_add_out, latency = group_norm_ort(
        input_tensor, gamma, beta, skip, bias, config, measure_latency=measure_latency
    )

    torch_out, torch_add_out = group_norm_torch(input_tensor, gamma, beta, skip, bias, config)

    average_diff = numpy.mean(numpy.abs(out_ort.detach().cpu().numpy() - torch_out.detach().cpu().numpy()))

    is_close = numpy.allclose(
        out_ort.detach().cpu().numpy(),
        torch_out.detach().cpu().numpy(),
        rtol=1e-1 if config.fp16 else 1e-3,
        atol=1e-1 if config.fp16 else 1e-3,
        equal_nan=True,
    )

    is_add_out_close = (
        numpy.allclose(
            ort_add_out.detach().cpu().numpy(),
            torch_add_out.detach().cpu().numpy(),
            rtol=1e-1 if config.fp16 else 1e-3,
            atol=1e-1 if config.fp16 else 1e-3,
            equal_nan=True,
        )
        if ort_add_out is not None
        else ""
    )

    # Compare results
    print(
        config.op_type.name,
        " B:",
        config.batch_size,
        " H:",
        config.height,
        " W:",
        config.width,
        " C:",
        config.channels,
        " G:",
        config.num_groups,
        " activation:",
        int(config.activation),
        " channels_last:",
        int(config.channels_last),
        " fp16:",
        int(config.fp16),
        f" Latency(μs): {int(latency * 1e6)}" if isinstance(latency, float) else "",
        " AvgDiff:",
        average_diff,
        " Pass:",
        is_close,
        is_add_out_close,
    )


def get_latent_height_width():
    default_size = [(512, 512), (768, 768), (1024, 1024)]
    small_img_size = [(512, 768), (768, 512)]
    xl_img_size = [
        (1152, 896),
        (896, 1152),
        (1216, 832),
        (832, 1216),
        (1344, 768),
        (768, 1344),
        (1536, 640),
        (640, 1536),
    ]
    return [(int(h / 8), int(w / 8)) for (h, w) in default_size + small_img_size + xl_img_size]


def get_channels():
    return [128, 256, 512, 1024, 2048, 320, 640, 960, 1920, 2560, 384, 768, 1536, 3072, 1152, 2304]


def run_activation(template: int, fp16, measure_latency=False):
    print("Test GroupNorm with Silu Activation for ", "fp16" if fp16 else "fp32")
    for b in [2]:
        for h, w in get_latent_height_width():
            for c in get_channels():
                config = GroupNormConfig.create(b, h, w, c, fp16=fp16, activation=True, template=template)
                run_parity(config, measure_latency=measure_latency)


def run_no_activation(template: int, fp16, measure_latency=False):
    print("Test GroupNorm without Activation for ", "fp16" if fp16 else "fp32")
    for b in [1, 2, 4]:
        for h, w in get_latent_height_width():
            for c in get_channels():
                config = GroupNormConfig.create(b, h, w, c, fp16=fp16, template=template)
                run_parity(config, measure_latency=measure_latency)


def run_all_groups(template: int, fp16, measure_latency=False):
    group_sizes = [1, 2, 4, 8, 16, 32]
    print("Test GroupNorm for different group sizes:", group_sizes)
    for group_size in group_sizes:
        for h, w in get_latent_height_width()[:3]:
            for c in get_channels()[:2]:
                config = GroupNormConfig.create(2, h, w, c, fp16=fp16, num_groups=group_size, template=template)
                run_parity(config, measure_latency=measure_latency)


def run_odd_channels(template: int, fp16, measure_latency=False):
    # Test some random number of channels that can be divisible by 2 * num_groups
    for h, w in get_latent_height_width():
        for c in [448, 704, 832, 1664, 2240, 2688, 2880, 3008]:
            config = GroupNormConfig.create(2, h, w, c, fp16=fp16, num_groups=32, template=template)
            run_parity(config, measure_latency=measure_latency)


def run_small_inputs(template: int, fp16):
    config = GroupNormConfig.create(1, 2, 2, 16, fp16=fp16, activation=False, num_groups=4, template=template)
    run_parity(config, measure_latency=False)

    config = GroupNormConfig.create(1, 1, 1, 64, fp16=fp16, activation=False, num_groups=8, template=template)
    run_parity(config, measure_latency=False)

    config = GroupNormConfig.create(1, 1, 1, 64, fp16=fp16, activation=True, num_groups=8, template=template)
    run_parity(config, measure_latency=False)


def run_performance(fp16):
    # Run perf test to tune parameters for given number of channels.
    for h, w in get_latent_height_width()[:3]:
        for c in [2304]:
            config = GroupNormConfig.create(2, h, w, c, fp16=fp16, num_groups=32, template=0)
            run_parity(config, measure_latency=True)


def run_all(template: int):
    for fp16 in [True, False]:
        run_small_inputs(template, fp16)
        run_odd_channels(template, fp16)
        run_all_groups(template, fp16)
        run_activation(template, fp16)
        run_no_activation(template, fp16)


def run_not_implemented():
    # Expect failure. Check whether the error message is expected.
    try:
        config = GroupNormConfig(1, 2, 2, 513, num_groups=3)
        run_parity(config)
    except RuntimeError as e:
        assert "GroupNorm in CUDA does not support the input: n=1 h=2 w=2 c=513 groups=3" in str(e)


if __name__ == "__main__":
    run_performance(True)

    run_not_implemented()

    for template in range(6):
        run_all(template)
