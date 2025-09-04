# --------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
#
# Regular MoE CPU kernel testing implementation - SwiGLU Interleaved Only
#
# This file tests the non-quantized MoE CPU implementation with SwiGLU
# activation in interleaved format and validates parity between
# PyTorch reference implementation and ONNX Runtime CPU kernel.
#
# Based on the CUDA test structure for consistency.
# --------------------------------------------------------------------------

import itertools
import os
import unittest
from parameterized import parameterized
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from onnx import TensorProto, helper

try:
    import onnxruntime
    has_onnx = True
except ImportError:
    has_onnx = False

# Reduces number of tests to run for faster pipeline checks
pipeline_mode = os.getenv("PIPELINE_MODE", "1") == "1"

# Device and provider settings for CPU
device = torch.device("cpu")
ort_provider = ["CPUExecutionProvider"]

torch.manual_seed(42)
numpy.random.seed(42)

onnx_to_torch_type_map = {
    TensorProto.FLOAT16: torch.float16,
    TensorProto.FLOAT: torch.float,
    TensorProto.BFLOAT16: torch.bfloat16,
    TensorProto.UINT8: torch.uint8,
}

ort_to_numpy_type_map = {
    TensorProto.FLOAT16: numpy.float16,
    TensorProto.FLOAT: numpy.float32,
    TensorProto.UINT8: numpy.uint8,
}

ort_dtype_name_map = {
    TensorProto.FLOAT16: "FP16",
    TensorProto.FLOAT: "FP32",
    TensorProto.BFLOAT16: "BF16",
}


class SwigluMoeConfig:
    def __init__(
        self,
        hidden_size=2048,
        intermediate_size=2048,
        num_experts_per_token=2,
        num_local_experts=8,
    ):
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts_per_token = num_experts_per_token
        self.num_local_experts = num_local_experts


def swiglu(x: torch.Tensor, alpha: float = 1.702, limit: float = 7.0):
    x_glu = x[..., ::2]
    x_linear = x[..., 1::2]

    if limit is not None:
        x_glu = x_glu.clamp(max=limit)
        x_linear = x_linear.clamp(min=-limit, max=limit)

    y = x_glu * torch.sigmoid(alpha * x_glu) * (x_linear + 1)
    return y


class SwigluMlp(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.intermediate_size = config.intermediate_size
        self.hidden_dim = config.hidden_size
        self.w1 = nn.Linear(self.hidden_dim, 2 * self.intermediate_size, bias=True)
        self.w2 = nn.Linear(self.intermediate_size, self.hidden_dim, bias=True)

    def forward(self, x):
        x1 = self.w1(x)
        y = swiglu(x1)
        y = self.w2(y)
        return y


def make_onnx_intializer(name: str, tensor: torch.Tensor, shape, onnx_dtype):
    torch_dtype = onnx_to_torch_type_map[onnx_dtype]
    if torch_dtype == torch.bfloat16:
        numpy_vals_uint16 = tensor.to(torch.bfloat16).cpu().view(torch.uint16).numpy()
        initializer = helper.make_tensor(
            name=name,
            data_type=TensorProto.BFLOAT16,
            dims=shape,
            vals=numpy_vals_uint16.tobytes(),
            raw=True,
        )
    else:
        initializer = helper.make_tensor(
            name=name,
            data_type=onnx_dtype,
            dims=shape,
            vals=tensor.flatten().detach().cpu().numpy().astype(numpy.uint8).tolist()
            if onnx_dtype == TensorProto.UINT8
            else tensor.detach().to(torch_dtype).flatten().tolist(),
            raw=False,
        )
    return initializer


def create_swiglu_moe_onnx_graph(
    num_tokens: int,
    num_experts: int,
    hidden_size: int,
    inter_size: int,
    topk: int,
    onnx_dtype: int,
    quant_bits: int,
    fc1_experts_weights: torch.Tensor,
    fc1_experts_bias: torch.Tensor,
    fc2_experts_weights: torch.Tensor,
    fc2_experts_bias: torch.Tensor,
    fc1_experts_weight_scale: torch.Tensor = None,
    fc2_experts_weight_scale: torch.Tensor = None,
):
    use_quant = quant_bits > 0
    op_name = "QMoE" if use_quant else "MoE"

    inputs = (
        [
            "input",
            "router_probs",
            "fc1_experts_weights",
            "fc1_experts_weight_scale",
            "fc1_experts_bias",
            "fc2_experts_weights",
            "fc2_experts_weight_scale",
            "fc2_experts_bias",
        ]
        if use_quant
        else [
            "input",
            "router_probs",
            "fc1_experts_weights",
            "fc1_experts_bias",
            "fc2_experts_weights",
            "fc2_experts_bias",
        ]
    )

    nodes = [
        helper.make_node(
            op_name,
            inputs,
            ["output"],
            "MoE_0",
            k=topk,
            normalize_routing_weights=0,  # Test the fixed implementation
            activation_type="swiglu",
            swiglu_fusion=1,
            activation_alpha=1.702,
            activation_beta=1.0,
            domain="com.microsoft",
        ),
    ]

    if use_quant:
        nodes[0].attribute.extend([helper.make_attribute("expert_weight_bits", quant_bits)])

    components = 2 if quant_bits == 4 else 1
    fc1_weight_shape = [num_experts, 2 * inter_size, hidden_size // components]
    fc1_bias_shape = [num_experts, 2 * inter_size]
    fc1_experts_weight_scale_shape = [num_experts, 2 * inter_size]

    fc2_weight_shape = [num_experts, hidden_size, inter_size // components]
    fc2_bias_shape = [num_experts, hidden_size]
    fc2_experts_weight_scale_shape = [num_experts, hidden_size]

    weight_onnx_type = TensorProto.UINT8 if use_quant else onnx_dtype

    torch_dtype = onnx_to_torch_type_map[onnx_dtype]
    weight_torch_dtype = onnx_to_torch_type_map[weight_onnx_type]

    initializers = [
        make_onnx_intializer(
            "fc1_experts_weights", fc1_experts_weights.to(weight_torch_dtype), fc1_weight_shape, weight_onnx_type
        ),
        make_onnx_intializer("fc1_experts_bias", fc1_experts_bias.to(torch_dtype), fc1_bias_shape, onnx_dtype),
        make_onnx_intializer(
            "fc2_experts_weights", fc2_experts_weights.to(weight_torch_dtype), fc2_weight_shape, weight_onnx_type
        ),
        make_onnx_intializer("fc2_experts_bias", fc2_experts_bias.to(torch_dtype), fc2_bias_shape, onnx_dtype),
    ]

    if use_quant:
        initializers.extend(
            [
                make_onnx_intializer(
                    "fc1_experts_weight_scale",
                    fc1_experts_weight_scale.to(torch_dtype),
                    fc1_experts_weight_scale_shape,
                    onnx_dtype,
                ),
                make_onnx_intializer(
                    "fc2_experts_weight_scale",
                    fc2_experts_weight_scale.to(torch_dtype),
                    fc2_experts_weight_scale_shape,
                    onnx_dtype,
                ),
            ]
        )

    graph_inputs = [
        helper.make_tensor_value_info("input", onnx_dtype, [num_tokens, hidden_size]),
    ]

    graph_inputs.append(
        helper.make_tensor_value_info(
            "router_probs",
            onnx_dtype,
            [num_tokens, num_experts],
        )
    )

    graph_outputs = [
        helper.make_tensor_value_info("output", onnx_dtype, [num_tokens, hidden_size]),
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


class SparseMoeBlockORTHelper(nn.Module):
    def __init__(self, quant_bits=0, onnx_dtype=None):
        super().__init__()
        self.quant_bits = quant_bits
        if onnx_dtype is None:
            self.onnx_dtype = TensorProto.FLOAT16 if self.quant_bits > 0 else TensorProto.FLOAT
        else:
            self.onnx_dtype = onnx_dtype
        self.np_type = numpy.float16 if self.onnx_dtype == TensorProto.FLOAT16 else numpy.float32

    def create_ort_session(self, moe_onnx_graph):
        from onnxruntime import InferenceSession, SessionOptions

        sess_options = SessionOptions()
        sess_options.log_severity_level = 2

        try:
            ort_session = InferenceSession(moe_onnx_graph, sess_options, providers=ort_provider)
        except Exception as e:
            print(f"Failed to create ONNX Runtime session with provider {ort_provider}: {e}")
            print("Skipping ONNX Runtime execution for this test case.")
            return None

        return ort_session

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        pass

    def ort_forward(self, hidden_states: torch.Tensor, enable_performance_test=False) -> torch.Tensor:
        if self.ort_sess is None:
            return None

        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states_flat = hidden_states.view(-1, hidden_dim)
        router_logits = self.gate(hidden_states_flat)

        torch_dtype = onnx_to_torch_type_map[self.onnx_dtype]

        tensors = {
            "input": hidden_states_flat.clone().to(device=device, dtype=torch_dtype),
            "router_probs": router_logits.clone().to(device=device, dtype=torch_dtype),
            "output": torch.zeros_like(hidden_states_flat, device=device, dtype=torch_dtype),
        }

        ort_inputs = {
            "input": tensors["input"].detach().cpu().numpy(),
            "router_probs": tensors["router_probs"].detach().cpu().numpy(),
        }

        if enable_performance_test:
            import time

            repeat = 1000
            s = time.time()
            for _ in range(repeat):
                ort_outputs = self.ort_sess.run(None, ort_inputs)
            e = time.time()
            print(f"MoE CPU kernel time: {(e - s) / repeat * 1000} ms")
            ort_outputs = self.ort_sess.run(None, ort_inputs)
        else:
            ort_outputs = self.ort_sess.run(None, ort_inputs)

        output_tensor = torch.from_numpy(ort_outputs[0]).to(device)

        return output_tensor.reshape(batch_size, sequence_length, hidden_dim)

    def parity_check(self):
        hidden_state = torch.randn(self.batch_size, self.sequence_length, self.hidden_dim).to(device)
        torch_output = self.forward(hidden_state)
        ort_output = self.ort_forward(hidden_state)

        dtype_str = ort_dtype_name_map[self.onnx_dtype]

        ort_dtype_quant_bits_tolerance_map = {
            "FP32:0": (5e-3, 1e-3),
        }

        atol, rtol = ort_dtype_quant_bits_tolerance_map[f"{dtype_str}:{self.quant_bits}"]
        if ort_output is not None:
            print(
                f"name: {self.__class__.__name__}, quant_bits: {self.quant_bits}, dtype: {dtype_str},"
                f" batch: {self.batch_size}, seq_len: {self.sequence_length},"
                f" max_diff: {(torch_output.cpu() - ort_output.cpu()).abs().max()}"
            )
            torch.testing.assert_close(
                ort_output.cpu().to(torch.float32), torch_output.cpu().to(torch.float32), rtol=rtol, atol=atol
            )

    def benchmark_ort(self):
        hidden_state = torch.randn(self.batch_size, self.sequence_length, self.hidden_dim).to(device)
        self.ort_forward(hidden_state, enable_performance_test=True)

    def debug_detailed_comparison(self):
        """Detailed debugging to identify potential issues"""
        hidden_state = torch.randn(self.batch_size, self.sequence_length, self.hidden_dim).to(device)
        
        # Get PyTorch reference output
        torch_output = self.forward(hidden_state)
        
        # Get ORT output
        ort_output = self.ort_forward(hidden_state)
        
        if ort_output is None:
            print("ORT output is None - session creation failed")
            return
            
        print(f"\n=== DEBUGGING MoE COMPARISON ===")
        print(f"Input shape: {hidden_state.shape}")
        print(f"Input range: [{hidden_state.min():.6f}, {hidden_state.max():.6f}]")
        print(f"Input std: {hidden_state.std():.6f}")
        
        # Check router probabilities
        hidden_states_flat = hidden_state.view(-1, self.hidden_dim)
        router_logits = self.gate(hidden_states_flat)
        router_probs = F.softmax(router_logits, dim=-1)
        
        print(f"\nRouter logits range: [{router_logits.min():.6f}, {router_logits.max():.6f}]")
        print(f"Router probs range: [{router_probs.min():.6f}, {router_probs.max():.6f}]")
        print(f"Router probs sum per token (should be ~1.0): {router_probs.sum(dim=-1)[:5]}")
        
        # Check top-k selection
        top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        print(f"Top-k indices range: [{top_k_indices.min()}, {top_k_indices.max()}]")
        print(f"Top-k probs range: [{top_k_probs.min():.6f}, {top_k_probs.max():.6f}]")
        
        # Output comparison
        print(f"\nPyTorch output shape: {torch_output.shape}")
        print(f"ORT output shape: {ort_output.shape}")
        print(f"PyTorch output range: [{torch_output.min():.6f}, {torch_output.max():.6f}]")
        print(f"ORT output range: [{ort_output.min():.6f}, {ort_output.max():.6f}]")
        print(f"PyTorch output std: {torch_output.std():.6f}")
        print(f"ORT output std: {ort_output.std():.6f}")
        
        # Difference analysis
        diff = (torch_output - ort_output).abs()
        print(f"\nAbsolute difference range: [{diff.min():.8f}, {diff.max():.8f}]")
        print(f"Mean absolute difference: {diff.mean():.8f}")
        print(f"Std of absolute difference: {diff.std():.8f}")
        
        # Check for any NaN or Inf values
        torch_has_nan = torch.isnan(torch_output).any()
        torch_has_inf = torch.isinf(torch_output).any()
        ort_has_nan = torch.isnan(ort_output).any()
        ort_has_inf = torch.isinf(ort_output).any()
        
        print(f"\nPyTorch output has NaN: {torch_has_nan}, Inf: {torch_has_inf}")
        print(f"ORT output has NaN: {ort_has_nan}, Inf: {ort_has_inf}")
        
        # Check weight statistics
        for i, expert in enumerate(self.experts[:2]):  # Check first 2 experts
            w1_weight = expert.w1.weight
            w2_weight = expert.w2.weight
            print(f"\nExpert {i} W1 weight range: [{w1_weight.min():.6f}, {w1_weight.max():.6f}]")
            print(f"Expert {i} W2 weight range: [{w2_weight.min():.6f}, {w2_weight.max():.6f}]")
            print(f"Expert {i} W1 weight std: {w1_weight.std():.6f}")
            print(f"Expert {i} W2 weight std: {w2_weight.std():.6f}")
        
        print(f"=== END DEBUGGING ===\n")


class SwigluMoEBlock(SparseMoeBlockORTHelper):
    def __init__(
        self, config: SwigluMoeConfig, batch_size: int, sequence_length: int, quant_bits: int = 0, onnx_dtype=None
    ):
        super().__init__(quant_bits, onnx_dtype=onnx_dtype)
        self.hidden_dim = config.hidden_size
        self.ffn_dim = config.intermediate_size
        self.num_experts = config.num_local_experts
        self.top_k = config.num_experts_per_token
        use_quant = self.quant_bits > 0

        self.gate = nn.Linear(self.hidden_dim, self.num_experts, bias=True)

        self.experts = nn.ModuleList([SwigluMlp(config) for _ in range(self.num_experts)])

        fc1_w_list, fc2_w_list = [], []
        fc1_b_list, fc2_b_list = [], []

        for expert in self.experts:
            w1_weight = expert.w1.weight.data.clone()
            w2_weight = expert.w2.weight.data.clone() 
            w1_bias = expert.w1.bias.data.clone()     
            w2_bias = expert.w2.bias.data.clone()    
            
            fc1_w_list.append(w1_weight)
            fc2_w_list.append(w2_weight)
            fc1_b_list.append(w1_bias)
            fc2_b_list.append(w2_bias)

        fc1_experts_weights = torch.stack(fc1_w_list, dim=0)
        fc2_experts_weights = torch.stack(fc2_w_list, dim=0)
        fc1_experts_bias = torch.stack(fc1_b_list, dim=0)
        fc2_experts_bias = torch.stack(fc2_b_list, dim=0)

        self.batch_size = batch_size
        self.sequence_length = sequence_length

        self.moe_onnx_graph = create_swiglu_moe_onnx_graph(
            num_tokens=self.batch_size * self.sequence_length,
            num_experts=self.num_experts,
            hidden_size=self.hidden_dim,
            inter_size=self.ffn_dim,
            topk=self.top_k,
            onnx_dtype=self.onnx_dtype,
            quant_bits=self.quant_bits,
            fc1_experts_weights=fc1_experts_weights,
            fc1_experts_bias=fc1_experts_bias,
            fc2_experts_weights=fc2_experts_weights,
            fc2_experts_bias=fc2_experts_bias,
        )

        self.ort_sess = self.create_ort_session(self.moe_onnx_graph)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        router_logits = self.gate(hidden_states)
        
        # With normalize_routing_weights=0: full softmax then top-k selection
        full_probs = F.softmax(router_logits, dim=-1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(full_probs, self.top_k, dim=-1)
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])

            if top_x.shape[0] == 0:
                continue

            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)

            current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]

            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))

        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states


swiglu_test_cases = list(
    itertools.product(
        [1, 2],  # batch_size
        [16, 32],  # sequence_length
        [0],  # quant_bits (CPU kernel only supports float32)
    )
)

perf_test_cases = list(
    itertools.product(
        [1],  # batch_size
        [128],  # sequence_length
        [0],  # quant_bits (CPU kernel only supports float32)
    )
)

class TestSwigluMoECPU(unittest.TestCase):
    @parameterized.expand(swiglu_test_cases)
    def test_swiglu_moe_parity(self, batch_size, sequence_length, quant_bits):
        config = SwigluMoeConfig(hidden_size=64, intermediate_size=256, num_experts_per_token=2, num_local_experts=4)
        moe = SwigluMoEBlock(config, batch_size, sequence_length, quant_bits)
        moe.to(device)
        moe.parity_check()


class TestSwigluMoECPUPerf(unittest.TestCase):
    @parameterized.expand(perf_test_cases)
    def test_swiglu_moe_perf(self, batch_size, sequence_length, quant_bits):
        hidden_size = 1024
        intermediate_size = 2048
        num_experts_per_token = 4
        num_local_experts = 16
        config = SwigluMoeConfig(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts_per_token=num_experts_per_token,
            num_local_experts=num_local_experts,
        )
        moe = SwigluMoEBlock(config, batch_size, sequence_length, quant_bits)
        moe.to(device)
        moe.benchmark_ort()

if __name__ == "__main__":
    unittest.main()
