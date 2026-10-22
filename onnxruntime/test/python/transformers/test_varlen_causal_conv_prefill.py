# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import unittest

import numpy as np
import torch
from onnx import TensorProto, checker, helper

import onnxruntime as ort

ort.preload_dlls()


@unittest.skipUnless(
    torch.cuda.is_available() and "CUDAExecutionProvider" in ort.get_available_providers(),
    "CUDA is required",
)
class TestVarlenCausalConvPrefill(unittest.TestCase):
    def run_conv(self, x, weight, bias, state, lengths, capture, capacity, dilation, activation, alias):
        dtype = {
            torch.float32: TensorProto.FLOAT,
            torch.float16: TensorProto.FLOAT16,
            torch.bfloat16: TensorProto.BFLOAT16,
        }[x.dtype]
        batch, channels, pad = state.shape
        inputs = [
            helper.make_tensor_value_info("x", dtype, list(x.shape)),
            helper.make_tensor_value_info("weight", dtype, list(weight.shape)),
            helper.make_tensor_value_info("lengths", TensorProto.INT32, [batch + 1]),
            helper.make_tensor_value_info("state", dtype, list(state.shape)),
        ]
        tensors = {"x": x, "weight": weight, "state": state}
        names = ["x", "weight", "lengths", "", "state"]
        if bias is not None:
            inputs.append(helper.make_tensor_value_info("bias", dtype, [channels]))
            tensors["bias"] = bias
            names[3] = "bias"
        if capacity:
            inputs.append(helper.make_tensor_value_info("capture", TensorProto.INT32, [batch]))
            names.append("capture")
        output_shapes = {"y": list(x.shape), "final": [batch, channels, pad]}
        if capacity:
            output_shapes["updates"] = [batch, capacity, channels]
        outputs = [helper.make_tensor_value_info(name, dtype, shape) for name, shape in output_shapes.items()]
        node = helper.make_node(
            "VarlenCausalConvWithState",
            names,
            list(output_shapes),
            domain="com.microsoft",
            dilation=dilation,
            activation=activation,
            state_update_capacity=capacity,
        )
        model = helper.make_model(
            helper.make_graph([node], "varlen_conv", inputs, outputs),
            opset_imports=[helper.make_opsetid("", 23), helper.make_opsetid("com.microsoft", 1)],
            ir_version=10,
        )
        checker.check_model(model)
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        stream = torch.cuda.current_stream()
        session = ort.InferenceSession(
            model.SerializeToString(),
            sess_options=options,
            providers=[("CUDAExecutionProvider", {"user_compute_stream": str(stream.cuda_stream)})],
        )
        binding = session.io_binding()
        for name, tensor in tensors.items():
            binding.bind_input(name, "cuda", 0, dtype, tuple(tensor.shape), tensor.data_ptr())
        binding.bind_cpu_input("lengths", np.asarray(lengths, dtype=np.int32))
        if capacity:
            binding.bind_cpu_input("capture", np.asarray(capture, dtype=np.int32))
        result = {
            name: state if alias and name == "final" else torch.empty(shape, dtype=x.dtype, device=x.device)
            for name, shape in output_shapes.items()
        }
        for name, tensor in result.items():
            binding.bind_output(name, "cuda", 0, dtype, tuple(tensor.shape), tensor.data_ptr())
        session.run_with_iobinding(binding)
        stream.synchronize()
        return result

    def test_prefill_matches_general_kernel_and_aliased_continuation(self):
        generator = torch.Generator().manual_seed(73381)
        cases = [
            (length, dilation, capture, dtype)
            for length in (255, 256, 257, 1024)
            for dilation in (1, 3)
            for capture in (-1, 0, 7, 99)
            for dtype in (torch.float32, torch.float16, torch.bfloat16)
        ]
        for length, dilation, capture, dtype in cases:
            with self.subTest(length=length, dilation=dilation, capture=capture, dtype=dtype):
                channels = 129
                pad = 3 * dilation

                def random(shape, dtype=dtype):
                    return (torch.randn(shape, generator=generator) * 0.3).to(dtype).cuda()

                x = random((length, channels))
                weight = random((channels, 1, 4))
                bias = random((channels,))
                state = random((1, channels, pad))
                kwargs = dict(capacity=7, dilation=dilation, activation="silu")
                actual = self.run_conv(x, weight, bias, state.clone(), [0, length], [capture], alias=True, **kwargs)
                # Two requests force the general channel-tiled path, with identical arithmetic.
                expected = self.run_conv(
                    x.repeat(2, 1),
                    weight,
                    bias,
                    state.repeat(2, 1, 1),
                    [0, length, 2 * length],
                    [capture, capture],
                    alias=False,
                    **kwargs,
                )
                bits = torch.int32 if dtype == torch.float32 else torch.int16
                for name in actual:
                    rows = length if name == "y" else 1
                    self.assertTrue(torch.equal(actual[name].view(bits), expected[name][:rows].view(bits)), name)
                next_x = random((2, channels))
                actual_next = self.run_conv(next_x, weight, bias, actual["final"], [0, 2], [2], alias=True, **kwargs)
                expected_next = self.run_conv(
                    next_x, weight, bias, expected["final"][:1].clone(), [0, 2], [2], alias=False, **kwargs
                )
                for name in actual_next:
                    self.assertTrue(torch.equal(actual_next[name].view(bits), expected_next[name].view(bits)), name)

    def test_prefill_without_capture_or_bias(self):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            for activation in ("none", "silu"):
                with self.subTest(dtype=dtype, activation=activation):
                    x = torch.linspace(-1, 1, 257 * 33, device="cuda").to(dtype).reshape(257, 33)
                    weight = torch.full((33, 1, 4), 0.25, dtype=dtype, device="cuda")
                    state = torch.zeros((1, 33, 3), dtype=dtype, device="cuda")
                    args = dict(capacity=0, dilation=1, activation=activation)
                    actual = self.run_conv(x, weight, None, state.clone(), [0, 257], [], alias=True, **args)
                    expected = self.run_conv(
                        x.repeat(2, 1), weight, None, state.repeat(2, 1, 1), [0, 257, 514], [], alias=False, **args
                    )
                    for name in actual:
                        rows = 257 if name == "y" else 1
                        self.assertTrue(torch.equal(actual[name], expected[name][:rows]), name)


if __name__ == "__main__":
    unittest.main()
