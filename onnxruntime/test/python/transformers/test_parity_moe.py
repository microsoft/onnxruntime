# --------------------------------------------------------------------------
# Copyright 2020 The HuggingFace Inc. team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# --------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# -------------------------------------------------------------------------

import pytest
import unittest

import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from onnx import TensorProto, helper

import onnxruntime

torch.manual_seed(42)
numpy.random.seed(42)


ORT_DTYPE = TensorProto.FLOAT16
THRESHOLD = 3e-2


def value_string_of(numpy_array):
    arr = numpy_array.flatten()
    lines = ["f, ".join([str(v) for v in arr[i : min(i + 8, arr.size)]]) for i in range(0, arr.size, 8)]
    return "{\n    " + "f,\n    ".join(lines) + "f}"


def print_tensor(name, numpy_array):
    print(f"const std::vector<float> {name} = {value_string_of(numpy_array)};")


def create_moe_onnx_graph(
    num_rows,
    num_experts,
    hidden_size,
    inter_size,
    fc1_experts_weights,
    fc2_experts_weights,
    fc1_experts_bias,
    fc2_experts_bias,
):
    nodes = [
        helper.make_node(
            "MoE",
            [
                "input",
                "router_probs",
                "fc1_experts_weights",
                "fc2_experts_weights",
                "fc1_experts_bias",
                "fc2_experts_bias",
            ],
            ["output"],
            "MoE_0",
            k=1,
            activation_type="gelu",
            domain="com.microsoft",
        ),
    ]

    fc1_shape = [num_experts, hidden_size, inter_size]
    fc2_shape = [num_experts, inter_size, hidden_size]

    torch_type = torch.float16 if ORT_DTYPE == TensorProto.FLOAT16 else torch.float32

    initializers = [
        helper.make_tensor(
            "fc1_experts_weights",
            ORT_DTYPE,
            fc1_shape,
            fc1_experts_weights.to(torch_type).flatten().tolist(),
            raw=False,
        ),
        helper.make_tensor(
            "fc2_experts_weights",
            ORT_DTYPE,
            fc2_shape,
            fc2_experts_weights.to(torch_type).flatten().tolist(),
            raw=False,
        ),
    ]

    fc1_bias_shape = [num_experts, inter_size]
    fc2_bias_shape = [num_experts, hidden_size]
    initializers.extend(
        [
            helper.make_tensor(
                "fc1_experts_bias",
                ORT_DTYPE,
                fc1_bias_shape,
                fc1_experts_bias.to(torch_type).flatten().tolist(),
                raw=False,
            ),
            helper.make_tensor(
                "fc2_experts_bias",
                ORT_DTYPE,
                fc2_bias_shape,
                fc2_experts_bias.to(torch_type).flatten().tolist(),
                raw=False,
            ),
        ]
    )

    graph_inputs = [
        helper.make_tensor_value_info("input", ORT_DTYPE, [num_rows, hidden_size]),
    ]

    graph_inputs.append(
        helper.make_tensor_value_info(
            "router_probs",
            ORT_DTYPE,
            [num_rows, num_experts],
        )
    )

    graph_outputs = [
        helper.make_tensor_value_info("output", ORT_DTYPE, [num_rows, hidden_size]),
    ]

    graph = helper.make_graph(
        nodes,
        "MoE_Graph",
        graph_inputs,
        graph_outputs,
        initializers,
    )

    model = helper.make_model(graph)
    return model.SerializeToString()


def get_activation_fn(activation):
    if activation == "relu":
        return nn.ReLU
    elif activation == "gelu":
        return nn.GELU
    else:
        raise NotImplementedError


class MoEGate(nn.Module):
    def __init__(self, num_experts, in_features):
        super().__init__()
        self.wg_reduction = torch.nn.Linear(in_features, 16, bias=False)

        wg = torch.empty(num_experts, 16)
        torch.nn.init.orthogonal_(wg, gain=0.32)
        self.register_parameter("wg", torch.nn.Parameter(wg))

    def forward(self, input):
        input = self.wg_reduction(input)
        with torch.no_grad():
            wg_norm = self.wg.norm(p=2.0, dim=1, keepdim=True)
            self.wg.mul_(1.5 / wg_norm)
        logits = self._cosine(input, self.wg)
        return logits

    def _cosine(self, mat1, mat2, eps=1e-4):
        assert mat1.dim() == 2
        assert mat2.dim() == 2
        # mat1 = F.normalize(mat1, p=2.0, dim=1, eps=eps)
        mat2 = F.normalize(mat2.float(), p=2.0, dim=1, eps=eps)
        return mat1.float().matmul(mat2.transpose(0, 1)).type_as(mat1)


class MoERuntimeExperts(nn.Module):
    def __init__(
        self,
        num_experts,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
        bias=True,
        chunk_size=-1,
    ):
        super().__init__()
        # assert bias is False, "Current bias is not supported"
        assert drop == 0.0, "Current drop is not supported"
        assert chunk_size == -1, "Current chunk is not supported"

        self.weight1 = nn.Parameter(torch.rand(num_experts, in_features, hidden_features))
        self.weight2 = nn.Parameter(torch.rand(num_experts, hidden_features, out_features))

        self.bias1 = nn.Parameter(torch.rand(num_experts, hidden_features)) if bias else None
        self.bias2 = nn.Parameter(torch.rand(num_experts, in_features)) if bias else None

        self.act = act_layer()

    def forward(self, x, indices_s):
        x = x.unsqueeze(1)
        x = self.bmm(x, self.weight1, indices_s)
        if self.bias1 is not None:
            x = x + self.bias1[indices_s].unsqueeze(1)  # S x hidden_features
        x = self.act(x)
        x = self.bmm(x, self.weight2, indices_s)
        if self.bias2 is not None:
            x = x + self.bias2[indices_s].unsqueeze(1)  # S x 1 x in_features
        return x

    def bmm(self, x, weight, indices_s):
        x = torch.bmm(x, weight[indices_s])  # S x 1 x hidden_features
        return x


class MoE(nn.Module):
    def __init__(
        self,
        batch_size,
        num_rows,
        num_experts,
        in_features,
        hidden_features=None,
        out_features=None,
        eval_capacity=-1,
        activation="gelu",
    ):
        super().__init__()
        self.num_experts = num_experts
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.eval_capacity = eval_capacity  # -1 means we route all tokens

        self.gate = MoEGate(num_experts=num_experts, in_features=in_features)
        self.moe_experts = MoERuntimeExperts(
            num_experts=num_experts,
            in_features=in_features,
            hidden_features=hidden_features,
            out_features=out_features,
            act_layer=get_activation_fn(activation),
            bias=True,
        )

        self.moe_onnx_graph = create_moe_onnx_graph(
            batch_size * num_rows,
            num_experts,
            in_features,
            hidden_features,
            self.moe_experts.weight1,
            self.moe_experts.weight2,
            self.moe_experts.bias1,
            self.moe_experts.bias2,
        )

        self.ort_sess = self.create_ort_session()

        self.torch_input = torch.randn(batch_size, num_rows, in_features)

    def create_ort_session(self):
        from onnxruntime import InferenceSession, SessionOptions

        sess_options = SessionOptions()

        cuda_providers = ["CUDAExecutionProvider"]
        if cuda_providers[0] not in onnxruntime.get_available_providers():
            return None

        sess_options.log_severity_level = 2
        ort_session = InferenceSession(self.moe_onnx_graph, sess_options, providers=["CUDAExecutionProvider"])

        return ort_session

    def torch_forward(self):
        x = self.torch_input

        b, t, c = x.shape
        x = x.reshape(-1, c)
        logits = self.gate(x)
        gates = torch.nn.functional.softmax(logits, dim=1)
        ret = torch.max(gates, dim=1)
        indices_s = ret.indices  # dim: [bs], the index of the expert with highest softmax value
        scores = ret.values.unsqueeze(-1).unsqueeze(-1)  # S
        x = self.moe_experts(x, indices_s)

        x = x * scores
        x = x.reshape(b * t, c)

        return x, torch.sum(x)

    def onnx_forward(self):
        x = self.torch_input

        _, _, c = x.shape
        y = x.reshape(-1, c)
        logits = self.gate(y)

        np_type = numpy.float16 if ORT_DTYPE == TensorProto.FLOAT16 else numpy.float32

        ort_inputs = {
            "input": numpy.ascontiguousarray(y.detach().numpy().astype(np_type)),
            "router_probs": numpy.ascontiguousarray(logits.detach().numpy().astype(np_type)),
        }

        ort_output = None
        if self.ort_sess is not None:
            ort_output = self.ort_sess.run(None, ort_inputs)

        # print_tensor("input", ort_inputs["input"])
        # print_tensor("router_probs", ort_inputs["router_probs"])
        # print_tensor("fc1_experts_weights", self.moe_experts.weight1.detach().numpy())
        # print_tensor("fc2_experts_weights", self.moe_experts.weight2.detach().numpy())
        # print_tensor("fc1_experts_bias", self.moe_experts.bias1.detach().numpy())
        # print_tensor("fc2_experts_bias", self.moe_experts.bias2.detach().numpy())
        # print_tensor("output", ort_output[0])

        return ort_output

    def parity_check(self):
        torch_out = self.torch_forward()
        ort_out = self.onnx_forward()
        if ort_out is not None:
            # print("max diff", numpy.max(numpy.abs(torch_out[0].detach().numpy() - ort_out[0])))
            assert numpy.allclose(torch_out[0].detach().numpy(), ort_out[0], rtol=THRESHOLD, atol=THRESHOLD)


class TestMoE(unittest.TestCase):
    @pytest.mark.slow
    def test_moe_large(self):
        for batch_size in [1, 8]:
            for num_rows in [16, 64]:
                for num_experts in [16, 64]:
                    for in_features in [256]:
                        for hidden_features in [512]:
                            print(
                                f"batch_size={batch_size}, num_rows={num_rows}, num_experts={num_experts}, in_features={in_features}, hidden_features={hidden_features}"
                            )
                            rt = MoE(
                                batch_size=batch_size,
                                num_rows=num_rows,
                                num_experts=num_experts,
                                in_features=in_features,
                                hidden_features=hidden_features,
                                out_features=in_features,
                            )
                            rt.parity_check()

    def test_moe_small(self):
        rt = MoE(
            batch_size=2,
            num_rows=8,
            num_experts=4,
            in_features=16,
            hidden_features=32,
            out_features=16,
        )
        rt.parity_check()


if __name__ == "__main__":
    unittest.main()
