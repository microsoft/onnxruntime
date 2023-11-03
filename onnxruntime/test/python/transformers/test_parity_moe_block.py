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

import unittest

import numpy
import numpy as np
import onnx
import onnxruntime

import torch
import torch.nn as nn
import torch.nn.functional as F


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
    from onnx import TensorProto, helper

    nodes = [
        helper.make_node(
            "MoEBlock",
            [
                "input",
                "gated_output",
                "fc1_experts_weights",
                "fc2_experts_weights",
                "fc1_experts_bias",
                "fc2_experts_bias",
            ],
            ["output"],
            "MoEBlock_0",
            k=1,
            activation_type="gelu",
            domain="com.microsoft",
        ),
    ]

    fc1_shape = [num_experts, hidden_size, inter_size]
    fc2_shape = [num_experts, inter_size, hidden_size]

    initializers = [
        helper.make_tensor(
            "fc1_experts_weights",
            TensorProto.FLOAT16,
            fc1_shape,
            fc1_experts_weights.to(torch.float16).flatten().tolist(),
            raw=False,
        ),
        helper.make_tensor(
            "fc2_experts_weights",
            TensorProto.FLOAT16,
            fc2_shape,
            fc2_experts_weights.to(torch.float16).flatten().tolist(),
            raw=False,
        ),
    ]

    fc1_bias_shape = [num_experts, inter_size]
    fc2_bias_shape = [num_experts, hidden_size]
    initializers.extend(
        [
            helper.make_tensor(
                "fc1_experts_bias",
                TensorProto.FLOAT16,
                fc1_bias_shape,
                fc1_experts_bias.to(torch.float16).flatten().tolist(),
                raw=False,
            ),
            helper.make_tensor(
                "fc2_experts_bias",
                TensorProto.FLOAT16,
                fc2_bias_shape,
                fc2_experts_bias.to(torch.float16).flatten().tolist(),
                raw=False,
            ),
        ]
    )

    graph_inputs = [
        helper.make_tensor_value_info("input", TensorProto.FLOAT16, [num_rows, hidden_size]),
    ]

    graph_inputs.append(
        helper.make_tensor_value_info(
            "gated_output",
            TensorProto.FLOAT16,
            [num_rows, num_experts],
        )
    )

    graph_outputs = [
        helper.make_tensor_value_info("output", TensorProto.FLOAT16, [num_rows, hidden_size]),
    ]

    graph = helper.make_graph(
        nodes,
        "MoEBlock_Graph",
        graph_inputs,
        graph_outputs,
        initializers,
    )

    model = helper.make_model(graph)
    return model.SerializeToString()


def onnx_inference(
    onnx_model_path,
    ort_inputs,
):
    from onnxruntime import InferenceSession, SessionOptions

    sess_options = SessionOptions()
    sess_options.log_severity_level = 2
    ort_session = InferenceSession(onnx_model_path, sess_options, providers=["CUDAExecutionProvider"])

    ort_output = ort_session.run(None, ort_inputs)
    return ort_output


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

        self.weight1 = nn.Parameter(torch.Tensor(num_experts, in_features, hidden_features))
        self.weight2 = nn.Parameter(torch.Tensor(num_experts, hidden_features, out_features))

        self.bias1 = nn.Parameter(torch.Tensor(num_experts, hidden_features)) if bias else None
        self.bias2 = nn.Parameter(torch.Tensor(num_experts, in_features)) if bias else None

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
        S, _, __ = x.shape
        x = torch.bmm(x, weight[indices_s])  # S x 1 x hidden_features
        return x


class MoEBlock(nn.Module):
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
            num_rows,
            num_experts,
            in_features,
            hidden_features,
            self.moe_experts.weight1,
            self.moe_experts.weight2,
            self.moe_experts.bias1,
            self.moe_experts.bias2,
        )

        self.torch_input = torch.randn(batch_size, num_rows, in_features)

    def torch_forward(self):
        x = self.torch_input

        B, T, C = x.shape
        x = x.reshape(-1, C)
        logits = self.gate(x)
        gates = torch.nn.functional.softmax(logits, dim=1)
        ret = torch.max(gates, dim=1)
        indices_s = ret.indices  # dim: [bs], the index of the expert with highest softmax value
        scores = ret.values.unsqueeze(-1).unsqueeze(-1)  # S
        x = self.moe_experts(x, indices_s)

        x = x * scores
        x = x.reshape(B, T, C)
        #print(x)
        return x, torch.sum(x)

    def onnx_forward(self):
        x = self.torch_input

        _, _, C = x.shape
        y = x.reshape(-1, C)
        logits = self.gate(y)

        ort_inputs = {
            "input": numpy.ascontiguousarray(y.detach().numpy().astype(numpy.float16)),
            "gated_output": numpy.ascontiguousarray(logits.detach().numpy().astype(numpy.float16)),
        }
        ort_output = onnx_inference(self.moe_onnx_graph, ort_inputs)
        #print(ort_output)
        return ort_output


class TestMoEBlock(unittest.TestCase):
    def test_moe_block(self):
        rt = MoEBlock(
            batch_size=1,
            num_rows=64,
            num_experts=32,
            in_features=256,
            hidden_features=2048,
            out_features=256,
        )
        # TODO: assertion
        rt.torch_forward()
        rt.onnx_forward()


if __name__ == "__main__":
    unittest.main()
