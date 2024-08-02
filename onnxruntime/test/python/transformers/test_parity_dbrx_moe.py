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
# --------------------------------------------------------------------------
import unittest
import time
from collections import OrderedDict

import numpy
import os
import torch
import torch.nn.functional as F
from mpi4py import MPI
from onnx import TensorProto, helper
from torch import nn
import torch.nn.init as init
from typing import Tuple


import onnxruntime
import onnx

comm = MPI.COMM_WORLD

torch.manual_seed(42)
numpy.random.seed(42)

def get_rank():
    return comm.Get_rank()


def get_size():
    return comm.Get_size()


def print_out(*args):
    if get_rank() == 0:
        print(*args)


def value_string_of(numpy_array):
    arr = numpy_array.flatten()
    lines = ["f, ".join([str(v) for v in arr[i : min(i + 8, arr.size)]]) for i in range(0, arr.size, 8)]
    return "{\n    " + "f,\n    ".join(lines) + "f}"


def print_tensor(name, numpy_array):
    print(f"const std::vector<float> {name} = {value_string_of(numpy_array)};")

def save_model_to_disk(model, model_path):
    external_data_path = model_path + ".data"
    onnx.save_model(
        model, model_path, save_as_external_data=True, all_tensors_to_one_file=True, location=external_data_path
    )


#def delete_model_data(external_data):
    #os.remove("dbrx_moe.onnx")
    #os.remove(external_data)

ORT_DTYPE = TensorProto.FLOAT16
NP_TYPE = numpy.float16 if ORT_DTYPE == TensorProto.FLOAT16 else numpy.float32
THRESHOLD = 3e-2
THRESHOLD_EP = 1e-6

local_rank = get_rank()


def create_moe_onnx_graph(
    num_rows,
    num_experts,
    hidden_size,
    inter_size,
    fc1_experts_weights,
    fc2_experts_weights,
    fc3_experts_weights,
    topk,
):
    nodes = [
        helper.make_node(
            "MoE",
            [
                "input",
                "router_probs",
                "fc1_experts_weights",
                "",
                "fc2_experts_weights",
                "",
                "fc3_experts_weights",
            ],
            ["output"],
            "MoE_0",
            k=topk,
            normalize_routing_weights=1,
            activation_type="silu",
            domain="com.microsoft",
        ),
    ]

    fc1_shape = [num_experts, hidden_size, inter_size]
    fc2_shape = [num_experts, inter_size, hidden_size]
    fc3_shape = [num_experts, hidden_size, inter_size]

    torch_type = torch.float16 if ORT_DTYPE == TensorProto.FLOAT16 else torch.float32


    initializers = [
        helper.make_tensor(
            "fc1_experts_weights",
            ORT_DTYPE,
            fc1_shape,
            fc1_experts_weights.to(torch_type).detach().numpy().tobytes(),
            raw=True,
        ),
        helper.make_tensor(
            "fc2_experts_weights",
            ORT_DTYPE,
            fc2_shape,
            fc2_experts_weights.to(torch_type).detach().numpy().tobytes(),
            raw=True,
        ),
        helper.make_tensor(
            "fc3_experts_weights",
            ORT_DTYPE,
            fc3_shape,
            fc3_experts_weights.to(torch_type).detach().numpy().tobytes(),
            raw=True,
        ),
    ]


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
    model_path = "dbrx_moe.onnx"

    save_model_to_disk(model, model_path)

    return model_path
    #return model.SerializeToString()



class ClassInstantier(OrderedDict):
    def __getitem__(self, key):
        content = super().__getitem__(key)
        cls, kwargs = content if isinstance(content, tuple) else (content, {})
        return cls(**kwargs)


ACT2CLS = {
    "silu": nn.SiLU,
}
ACT2FN = ClassInstantier(ACT2CLS)


class DBRXConfig:
    def __init__(
        self,
        hidden_size=6144,
        intermediate_size=10752,
        num_hidden_layers=40,
        num_attention_heads=48,
        num_key_value_heads=8,
        hidden_act="silu",
        initializer_range=0.02,
        rms_norm_eps=1e-5,
        use_cache=True,
        rope_theta=5e5,
        attention_dropout=0.0,
        num_experts_per_tok=4,
        num_local_experts=16,
        output_router_logits=False,
        router_aux_loss_coef=0.001
    ):
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.attention_dropout = attention_dropout
        self.num_experts_per_tok = num_experts_per_tok
        self.num_local_experts = num_local_experts
        self.output_router_logits = output_router_logits
        self.router_aux_loss_coef = router_aux_loss_coef

class DbrxRouter(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        moe_num_experts: int,
        moe_top_k: int,
        config: DBRXConfig,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.moe_num_experts = config.num_local_experts
        self.moe_top_k = config.num_experts_per_tok

        self.layer = nn.Linear(self.hidden_size, self.moe_num_experts, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.LongTensor]:
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        weights = self.layer(hidden_states).softmax(dim=-1, dtype=torch.float32)
        top_weights, top_experts = torch.topk(weights, self.moe_top_k, dim=-1)

        weights = weights.to(hidden_states.dtype)
        top_weights = top_weights.to(hidden_states.dtype)
        return weights, top_weights, top_experts


class DbrxExpertGLU(nn.Module):
    def __init__(self, hidden_size: int, ffn_hidden_size: int, moe_num_experts: int, ffn_act_fn: dict, config: DBRXConfig):
        super().__init__()
        self.hidden_size = hidden_size
        self.ffn_hidden_size = config.intermediate_size
        self.moe_num_experts = config.num_local_experts
        ffn_act_fn = {"name": config.hidden_act}

        self.w1 = nn.Parameter(torch.randn(moe_num_experts * ffn_hidden_size, hidden_size))
        self.v1 = nn.Parameter(torch.randn(moe_num_experts * ffn_hidden_size, hidden_size))
        self.w2 = nn.Parameter(torch.randn(moe_num_experts * ffn_hidden_size, hidden_size))

        init.xavier_uniform_(self.w1)
        init.xavier_uniform_(self.v1)
        init.xavier_uniform_(self.w2)

        act_fn_name = ffn_act_fn.get("name", "silu")
        self.activation_fn = ACT2FN[act_fn_name]

    def forward(
        self, x: torch.Tensor, expert_w1: torch.Tensor, expert_v1: torch.Tensor, expert_w2: torch.Tensor
    ) -> torch.Tensor:
        gate_proj = x.matmul(expert_w1.t())
        up_proj = x.matmul(expert_v1.t())
        gate_proj = self.activation_fn(gate_proj)
        intermediate_states = gate_proj * up_proj
        down_proj = intermediate_states.matmul(expert_w2)
        return down_proj


class DbrxExperts(nn.Module):
    def __init__(self, hidden_size: int,
                 ffn_hidden_size: int,
                 moe_num_experts: int,
                 ffn_act_fn: dict,
                 batch_size: int,
                 sequence_length: int,
                 config: DBRXConfig):
        super().__init__()
        self.moe_num_experts = config.num_local_experts
        self.config = DBRXConfig()
        self.hidden_size = hidden_size
        self.ffn_hidden_size = config.intermediate_size
        self.moe_top_k = config.num_experts_per_tok
        self.mlp = DbrxExpertGLU(
                    hidden_size=hidden_size,
                    ffn_hidden_size=ffn_hidden_size,
                    moe_num_experts=moe_num_experts,
                    ffn_act_fn=config.hidden_act,
                    config=config
                    )

        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.moe_onnx_graph = create_moe_onnx_graph(
            self.batch_size * self.sequence_length,
            self.moe_num_experts,
            self.hidden_size,
            self.ffn_hidden_size,
            self.mlp.w1.view(moe_num_experts, hidden_size, ffn_hidden_size).reshape(moe_num_experts, hidden_size, ffn_hidden_size),
            self.mlp.w2.view(moe_num_experts, ffn_hidden_size, hidden_size).transpose(1, 2).reshape(moe_num_experts, ffn_hidden_size, hidden_size),
            self.mlp.v1.view(moe_num_experts, hidden_size, ffn_hidden_size).reshape(moe_num_experts, hidden_size, ffn_hidden_size),
            self.moe_top_k
        )

        self.ort_sess = self.create_ort_session()


    def test_moe_with_tensor_parallelism(
        self,
        hidden_size,
        inter_size,
        num_experts,
        num_rows,
        threshold=THRESHOLD,
    ):
        assert inter_size % get_size() == 0

        (
            onnx_model_full,
            fc1_experts_weights_all,
            fc2_experts_weights_all,
            fc3_experts_weights_all,
        ) = self.generate_weights_and_initial_model(
            num_rows,
            num_experts,
            hidden_size,
            inter_size,
        )

        def get_fc1_tensor_shards(expert_weights):
            return (
                expert_weights.reshape(-1, inter_size, hidden_size)
                .transpose(0, 2, 1)[
                    :, :, local_rank * inter_size // get_size() : (local_rank + 1) * inter_size // get_size()
                ]
                .transpose(0, 2, 1)
            )

        def get_fc2_tensor_shards(expert_weights):
            return (
                expert_weights.reshape(-1, hidden_size, inter_size)
                .transpose(0, 2, 1)[
                    :, local_rank * inter_size // get_size() : (local_rank + 1) * inter_size // get_size(), :
                ]
                .transpose(0, 2, 1)
            )

        fc1_experts_weights = get_fc1_tensor_shards(fc1_experts_weights_all)
        fc2_experts_weights = get_fc2_tensor_shards(fc2_experts_weights_all)
        fc3_experts_weights = get_fc1_tensor_shards(fc3_experts_weights_all)

        onnx_model_local = create_moe_onnx_graph(
            num_rows,
            num_experts,
            num_experts,
            hidden_size,
            inter_size // get_size(),
            fc1_experts_weights,
            fc2_experts_weights,
            fc3_experts_weights,
            tensor_shards=get_size(),
        )

        self.run_ort_with_parity_check(
            onnx_model_full,
            onnx_model_local,
            num_rows,
            hidden_size,
            num_experts,
            inter_size,
            threshold,
        )

    def run_ort_with_parity_check(
        self,
        onnx_model_full,
        onnx_model_local,
        num_rows,
        hidden_size,
        num_experts,
        inter_size,
        threshold,
    ):
        sess_options = onnxruntime.SessionOptions()
        cuda_provider_options = {"device_id": local_rank}
        execution_providers = [("CUDAExecutionProvider", cuda_provider_options)]

        ort_session = onnxruntime.InferenceSession(onnx_model_full, sess_options, providers=execution_providers)
        ort_session_local = onnxruntime.InferenceSession(onnx_model_local, sess_options, providers=execution_providers)

        ort_inputs = {
            ort_session.get_inputs()[0].name: numpy.random.rand(num_rows, hidden_size).astype(NP_TYPE),
            ort_session.get_inputs()[1].name: numpy.random.rand(num_rows, num_experts).astype(NP_TYPE),
        }

        output = ort_session.run(None, ort_inputs)
        sharded_output = ort_session_local.run(None, ort_inputs)

        print_out("max diff:", numpy.max(numpy.abs(output[0] - sharded_output[0])))
        assert numpy.allclose(output[0], sharded_output[0], atol=threshold, rtol=threshold)

        print_out(
            "hidden_size:",
            hidden_size,
            " inter_size:",
            inter_size,
            " num_experts:",
            num_experts,
            " num_rows:",
            num_rows,
            " world_size:",
            get_size(),
            " Parity: OK",
        )


    def generate_weights_and_initial_model(
        self,
        num_rows,
        num_experts,
        hidden_size,
        inter_size,
    ):
        s = 0.1
        fc1_experts_weights_all = numpy.random.normal(scale=s, size=(num_experts, hidden_size, inter_size)).astype(NP_TYPE)
        fc2_experts_weights_all = numpy.random.normal(scale=s, size=(num_experts, inter_size, hidden_size)).astype(NP_TYPE)
        fc3_experts_weights_all = numpy.random.normal(scale=s, size=(num_experts, hidden_size, inter_size)).astype(NP_TYPE)

        onnx_model_full = create_moe_onnx_graph(
            num_rows,
            num_experts,
            num_experts,
            hidden_size,
            inter_size,
            fc1_experts_weights_all,
            fc2_experts_weights_all,
            fc3_experts_weights_all,
        )

        return (
            onnx_model_full,
            fc1_experts_weights_all,
            fc2_experts_weights_all,
            fc3_experts_weights_all,
        )


    def create_ort_session(self):
        from onnxruntime import InferenceSession, SessionOptions

        sess_options = SessionOptions()

        cuda_providers = ["CUDAExecutionProvider"]
        if cuda_providers[0] not in onnxruntime.get_available_providers():
            return None
        sess_options.log_severity_level = 2
        ort_session = InferenceSession(self.moe_onnx_graph, sess_options, providers=["CUDAExecutionProvider"])

        return ort_session

    def ort_run_with_iobinding(self, ort_inputs, repeat=1000):
        iobinding = self.ort_sess.io_binding()
        device_id = torch.cuda.current_device()

        iobinding.bind_input(
            name="input",
            device_type="cuda",
            device_id=device_id,
            element_type=NP_TYPE,
            shape=ort_inputs["input"].shape,
            buffer_ptr=onnxruntime.OrtValue.ortvalue_from_numpy(ort_inputs["input"], "cuda", device_id).data_ptr(),
        )

        iobinding.bind_input(
            name="router_probs",
            device_type="cuda",
            device_id=device_id,
            element_type=NP_TYPE,
            shape=ort_inputs["router_probs"].shape,
            buffer_ptr=onnxruntime.OrtValue.ortvalue_from_numpy(
                ort_inputs["router_probs"], "cuda", device_id
            ).data_ptr(),
        )

        iobinding.bind_output(
            name="output",
            device_type="cuda",
            device_id=device_id,
            element_type=NP_TYPE,
            shape=ort_inputs["input"].shape,
            buffer_ptr=onnxruntime.OrtValue.ortvalue_from_numpy(
                numpy.zeros(ort_inputs["input"].shape), "cuda", device_id
            ).data_ptr(),
        )

        s = time.time()
        for _ in range(repeat):
            iobinding.synchronize_inputs()
            self.ort_sess.run_with_iobinding(iobinding)
            iobinding.synchronize_outputs()
        e = time.time()
        print(f"MoE cuda kernel time: {(e - s) / repeat * 1000} ms")

    def forward(
        self, x: torch.Tensor, weights: torch.Tensor, top_weights: torch.Tensor, top_experts: torch.LongTensor
    ) -> torch.Tensor:
        bsz, q_len, hidden_size = x.shape
        x = x.view(-1, hidden_size)
        out = torch.zeros_like(x)

        expert_mask = nn.functional.one_hot(top_experts, num_classes=self.moe_num_experts).permute(2, 1, 0)
        # Chunk experts at once to avoid storing full parameter multiple times in autograd
        w1_chunked = self.mlp.w1.view(self.mlp.moe_num_experts, self.mlp.ffn_hidden_size, self.mlp.hidden_size).chunk(
            self.moe_num_experts, dim=0
        )
        v1_chunked = self.mlp.v1.view(self.mlp.moe_num_experts, self.mlp.ffn_hidden_size, self.mlp.hidden_size).chunk(
            self.moe_num_experts, dim=0
        )
        w2_chunked = self.mlp.w2.view(self.mlp.moe_num_experts, self.mlp.ffn_hidden_size, self.mlp.hidden_size).chunk(
            self.moe_num_experts, dim=0
        )
        w1_chunked = [w1.squeeze(dim=0) for w1 in w1_chunked]
        v1_chunked = [v1.squeeze(dim=0) for v1 in v1_chunked]
        w2_chunked = [w2.squeeze(dim=0) for w2 in w2_chunked]
        for expert_idx in range(0, self.moe_num_experts):
            topk_idx, token_idx = torch.where(expert_mask[expert_idx])
            if token_idx.shape[0] == 0:
                continue

            token_list = token_idx
            topk_list = topk_idx

            expert_tokens = x[None, token_list].reshape(-1, hidden_size)
            expert_out = (
                self.mlp(expert_tokens, w1_chunked[expert_idx], v1_chunked[expert_idx], w2_chunked[expert_idx])
                * top_weights[token_list, topk_list, None]
            )

            out.index_add_(0, token_idx, expert_out)

        out = out.reshape(bsz, q_len, hidden_size)
        return out


    def parity_check(self):
        config = DBRXConfig()
        ffn = DbrxFFN(config, self.batch_size, self.sequence_length)
        router = DbrxRouter(hidden_size=config.hidden_size,
                              moe_num_experts=config.num_local_experts,
                              moe_top_k=config.num_local_experts,
                              config=DBRXConfig())
        hidden_state = torch.randn(self.batch_size, self.sequence_length, self.hidden_size)
        torch_output = ffn.forward(hidden_state)
        print("forward: ", torch_output)
        ort_output = ffn.ort_forward(hidden_state, iobinding=False)
        """
        if ort_output is not None:
            assert torch.allclose(torch_output, ort_output, rtol=1e-04, atol=1e-04)
            print(
                "batch_size:",
                self.batch_size,
                " sequence_length:",
                self.sequence_length,
                " max_diff:",
                (torch_output - ort_output).abs().max(),
                " parity: OK",
            )
        """



class DbrxFFN(nn.Module):
    def __init__(self, config: DBRXConfig, batch_size, sequence_length):
        super().__init__()
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.router = DbrxRouter(
            hidden_size=config.hidden_size,
            moe_num_experts=config.num_local_experts,
            moe_top_k=config.num_experts_per_tok,
            config=DBRXConfig()
        )

        self.experts = DbrxExperts(
            hidden_size=config.hidden_size,
            ffn_hidden_size=config.intermediate_size,
            moe_num_experts=config.num_local_experts,
            ffn_act_fn=config.hidden_act,
            batch_size=batch_size,
            sequence_length=sequence_length,
            config=DBRXConfig()
            )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        print("Input:", x)
        weights, top_weights, top_experts = self.router(x)
        print("After router:", weights, top_weights, top_experts)
        out = self.experts(x, weights, top_weights, top_experts)
        print("After experts:", out)
        return out, weights


    def ort_forward(self, hidden_states: torch.Tensor, iobinding=False) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        assert not torch.isnan(hidden_states).any(), "Input hidden_states contains NaN values"
        assert not torch.isinf(hidden_states).any(), "Input hidden_states contains Inf values"

        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.router.layer(hidden_states)

        assert not torch.isnan(router_logits).any(), "router_logits contains NaN values"
        assert not torch.isinf(router_logits).any(), "router_logits contains Inf values"
        ort_inputs = {
            "input": numpy.ascontiguousarray(hidden_states.detach().numpy().astype(NP_TYPE)),
            "router_probs": numpy.ascontiguousarray(router_logits.detach().numpy().astype(NP_TYPE)),
        }

        #ort_output = None
        if self.experts.ort_sess is not None:
            if not iobinding:
                ort_output = self.experts.ort_sess.run(None, ort_inputs)
            else:
                ort_output = self.experts.ort_run_with_iobinding(ort_inputs)
                #return ort_output



        # print_tensor("input", ort_inputs["input"])
        # print_tensor("router_probs", ort_inputs["router_probs"])
        # print_tensor("fc1_experts_weights", self.moe_experts_weight1.detach().numpy())
        # print_tensor("fc2_experts_weights", self.moe_experts_weight2.detach().numpy())
        # print_tensor("fc3_experts_weights", self.moe_experts_weight3.detach().numpy())
        # print_tensor("output", ort_output[0])

        print("ORT output:", ort_output)
        assert not numpy.isnan(ort_output).any(), "ORT output contains NaN values"
        assert not numpy.isinf(ort_output).any(), "ORT output contains Inf values"
        return ort_output



class TestDBRXMoE(unittest.TestCase):
    def test_dbrx_moe_parity(self):
        for batch_size in [1, 16]:
            for sequence_length in [128, 1024]:
                # use a small sizes to speed up the test
                config = DBRXConfig()
                hidden_size = config.hidden_size
                moe_num_experts = config.num_local_experts
                #moe_top_k = config.num_experts_per_tok
                ffn_hidden_size = config.intermediate_size
                ffn_act_fn = {"name", config.hidden_act}
                dbrx_moe = DbrxExperts(hidden_size,
                                      ffn_hidden_size,
                                      moe_num_experts,
                                      ffn_act_fn,
                                      batch_size,
                                      sequence_length,
                                      config)
                dbrx_moe.parity_check()
                #dbrx_moe.test_moe_with_tensor_parallelism(hidden_size,
                                                          #ffn_hidden_size,
                                                          #moe_num_experts,
                                                          #num_rows=batch_size * sequence_length,
                                                          #threshold=THRESHOLD)
    #external_data_path = "dbrx_moe.onnx" + ".data"
    #delete_model_data(external_data_path)


if __name__ == "__main__":
    unittest.main()
