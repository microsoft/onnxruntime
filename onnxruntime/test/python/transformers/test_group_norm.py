# --------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# -------------------------------------------------------------------------
import statistics
from dataclasses import dataclass
from time import perf_counter

import numpy
import torch
from onnx import TensorProto, helper

from onnxruntime import InferenceSession
from onnxruntime.transformers.io_binding_helper import CudaSession

torch.manual_seed(0)


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


def create_group_norm_graph(config: GroupNormConfig) -> bytes:
    nodes = [
        helper.make_node(
            "GroupNorm",
            ["input", "gamma", "beta"],
            ["output"],
            "GroupNorm_0",
            activation=int(config.activation),
            channels_last=int(config.channels_last),
            epsilon=config.epsilon,
            groups=config.num_groups,
            domain="com.microsoft",
        ),
    ]

    float_type = TensorProto.FLOAT16 if config.fp16 else TensorProto.FLOAT

    graph = helper.make_graph(
        nodes,
        "Group_Norm_Graph",
        [
            helper.make_tensor_value_info("input", float_type, ["N", "H", "W", "C"]),
            helper.make_tensor_value_info("gamma", TensorProto.FLOAT, ["C"]),
            helper.make_tensor_value_info("beta", TensorProto.FLOAT, ["C"]),
        ],
        [
            helper.make_tensor_value_info("output", float_type, ["N", "H", "W", "C"]),
        ],
    )

    model = helper.make_model(graph)
    return model.SerializeToString()


def group_norm_ort(
    input: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    config: GroupNormConfig,
    measure_latency=False,
) -> torch.Tensor:
    onnx_model_str = create_group_norm_graph(config)
    ort_session = InferenceSession(onnx_model_str, providers=["CUDAExecutionProvider"])

    session = CudaSession(ort_session, device=torch.device("cuda:0"))

    io_shape = {
        "input": [config.batch_size, config.height, config.width, config.channels],
        "gamma": [config.channels],
        "beta": [config.channels],
        "output": [config.batch_size, config.height, config.width, config.channels],
    }

    session.allocate_buffers(io_shape)

    ort_inputs = {
        "input": input,
        "gamma": gamma,
        "beta": beta,
    }

    ort_outputs = session.infer(ort_inputs)
    output = ort_outputs["output"]

    if measure_latency:
        latency_list = []
        for _ in range(100):
            start_time = perf_counter()
            session.infer(ort_inputs)
            end_time = perf_counter()
            latency_list.append(end_time - start_time)
        average_latency = statistics.mean(latency_list)
        return output, average_latency

    return output, None


def group_norm_torch(
    input: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, config: GroupNormConfig
) -> torch.Tensor:
    if config.channels_last:
        input = input.permute(0, 3, 1, 2)  # from NHWC to NCHW

    weight = gamma.to(input.dtype)
    bias = beta.to(input.dtype)
    output = torch.nn.functional.group_norm(input, config.num_groups, weight=weight, bias=bias, eps=config.epsilon)

    if config.activation:
        torch.nn.functional.silu(output, inplace=True)

    if config.channels_last:
        output = output.permute(0, 2, 3, 1)  # from NCHW to NHWC

    return output


def run_parity(config, measure_latency=True):
    float_type = torch.float16 if config.fp16 else torch.float32

    intput = torch.randn(
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

    # Pytorch to compare
    out_ort, latency = group_norm_ort(intput, gamma, beta, config, measure_latency=measure_latency)
    ort_result = out_ort.detach().cpu().numpy()

    torch_out = group_norm_torch(intput, gamma, beta, config)
    torch_result = torch_out.detach().cpu().numpy()

    is_close = numpy.allclose(
        ort_result,
        torch_result,
        rtol=1e-1 if config.fp16 else 1e-3,
        atol=1e-1 if config.fp16 else 1e-3,
        equal_nan=True,
    )

    # Compare results
    print(
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
        config.activation,
        " channels_last:",
        config.channels_last,
        " fp16:",
        config.fp16,
        " Latency(ms):",
        latency * 1000 if isinstance(latency, float) else latency,
        " AvgDiff:",
        numpy.mean(numpy.abs(ort_result - torch_result)),
        " Pass:",
        is_close,
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


def run_activation(fp16, measure_latency=True):
    print("Test GroupNorm with Silu Activation for ", "fp16" if fp16 else "fp32")
    for b in [2]:
        for h, w in get_latent_height_width():
            for c in get_channels():
                config = GroupNormConfig(b, h, w, c, fp16=fp16, activation=True)
                run_parity(config, measure_latency=measure_latency)


def run_no_activation(fp16, measure_latency=True):
    print("Test GroupNorm without Activation for ", "fp16" if fp16 else "fp32")
    for b in [1, 2, 4]:
        for h, w in get_latent_height_width():
            for c in get_channels():
                config = GroupNormConfig(b, h, w, c, fp16=fp16)
                run_parity(config, measure_latency=measure_latency)


def run_all_groups(fp16, measure_latency=True):
    group_sizes = [1, 2, 4, 8, 16, 32]
    print("Test GroupNorm for different group sizes:", group_sizes)
    for group_size in group_sizes:
        for h, w in get_latent_height_width()[:3]:
            for c in get_channels()[:2]:
                config = GroupNormConfig(2, h, w, c, fp16=fp16, num_groups=group_size)
                run_parity(config, measure_latency=measure_latency)


def run_odd_channels(fp16, measure_latency=True):
    # Test some random number of channels that can be divisible by 2 * num_groups
    for h, w in get_latent_height_width():
        for c in [448, 704, 832, 1664, 2240, 2688, 2880, 3008]:
            config = GroupNormConfig(2, h, w, c, fp16=fp16, num_groups=32)
            run_parity(config, measure_latency=measure_latency)


def run_performance(fp16):
    for h, w in get_latent_height_width():
        for c in [1152, 2304, 2880]:
            config = GroupNormConfig(2, h, w, c, fp16=fp16, num_groups=32)
            run_parity(config, measure_latency=True)


def run_all():
    run_performance(True)

    measure_latency = False
    run_odd_channels(True, measure_latency=measure_latency)
    run_odd_channels(False, measure_latency=measure_latency)

    run_all_groups(True, measure_latency=measure_latency)
    run_all_groups(False, measure_latency=measure_latency)

    run_activation(True, measure_latency=measure_latency)
    run_activation(False, measure_latency=measure_latency)

    run_no_activation(True, measure_latency=measure_latency)
    run_no_activation(False, measure_latency=measure_latency)


if __name__ == "__main__":
    run_all()
