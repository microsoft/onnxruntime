# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import unittest

import numpy as np
from mpi4py import MPI
from onnx import TensorProto, helper

import onnxruntime

np.random.seed(3)

comm = MPI.COMM_WORLD


def get_rank():
    return comm.Get_rank()


def get_size():
    return comm.Get_size()


def print_out(*args):
    if get_rank() == 0:
        print(*args)


local_rank = get_rank()

ORT_DTYPE = TensorProto.FLOAT16
NP_TYPE = np.float16 if ORT_DTYPE == TensorProto.FLOAT16 else np.float32
THRESHOLD_TP = 3e-2
THRESHOLD_EP = 1e-6


def create_moe_onnx_graph(
    num_rows,
    num_experts,
    local_num_experts,
    hidden_size,
    inter_size,
    fc1_experts_weights,
    fc1_experts_bias,
    fc2_experts_weights,
    fc2_experts_bias,
    fc3_experts_weights,
    local_experts_start_index=0,
    topk=2,
    normalize_routing_weights=1,
    activation_type="gelu",
    tensor_shards=1,
):
    use_sharded_moe = num_experts > local_num_experts or tensor_shards > 1
    nodes = [
        (
            helper.make_node(
                "MoE",
                [
                    "input",
                    "router_probs",
                    "fc1_experts_weights",
                    "fc1_experts_bias",
                    "fc2_experts_weights",
                    "fc2_experts_bias",
                    "fc3_experts_weights",
                ],
                ["output"],
                "MoE_0",
                k=topk,
                normalize_routing_weights=normalize_routing_weights,
                activation_type=activation_type,
                domain="com.microsoft",
            )
            if not use_sharded_moe
            else helper.make_node(
                "ShardedMoE",
                [
                    "input",
                    "router_probs",
                    "fc1_experts_weights",
                    "fc1_experts_bias",
                    "fc2_experts_weights",
                    "fc2_experts_bias",
                    "fc3_experts_weights",
                ],
                ["output"],
                "MoE_0",
                k=topk,
                normalize_routing_weights=normalize_routing_weights,
                activation_type=activation_type,
                local_experts_start_index=local_experts_start_index,
                tensor_shards=tensor_shards,
                domain="com.microsoft",
            )
        ),
    ]

    fc1_shape = [local_num_experts, hidden_size, inter_size]
    fc2_shape = [local_num_experts, inter_size, hidden_size]
    fc3_shape = fc1_shape

    initializers = [
        helper.make_tensor(
            "fc1_experts_weights",
            ORT_DTYPE,
            fc1_shape,
            fc1_experts_weights.flatten(),
            raw=False,
        ),
        helper.make_tensor(
            "fc2_experts_weights",
            ORT_DTYPE,
            fc2_shape,
            fc2_experts_weights.flatten(),
            raw=False,
        ),
        helper.make_tensor(
            "fc3_experts_weights",
            ORT_DTYPE,
            fc3_shape,
            fc3_experts_weights.flatten(),
            raw=False,
        ),
    ]

    fc1_bias_shape = [local_num_experts, inter_size]
    fc2_bias_shape = [num_experts, hidden_size]
    initializers.extend(
        [
            helper.make_tensor(
                "fc1_experts_bias",
                ORT_DTYPE,
                fc1_bias_shape,
                fc1_experts_bias.flatten().tolist(),
                raw=False,
            ),
            helper.make_tensor(
                "fc2_experts_bias",
                ORT_DTYPE,
                fc2_bias_shape,
                fc2_experts_bias.flatten().tolist(),
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


def generate_weights_and_initial_model(
    num_rows,
    num_experts,
    hidden_size,
    inter_size,
):
    s = 0.1
    fc1_experts_weights_all = np.random.normal(scale=s, size=(num_experts, hidden_size, inter_size)).astype(NP_TYPE)
    fc2_experts_weights_all = np.random.normal(scale=s, size=(num_experts, inter_size, hidden_size)).astype(NP_TYPE)
    fc3_experts_weights_all = np.random.normal(scale=s, size=(num_experts, hidden_size, inter_size)).astype(NP_TYPE)
    fc1_experts_bias_all = np.random.normal(scale=s, size=(num_experts, inter_size)).astype(NP_TYPE)
    fc2_experts_bias_all = np.random.normal(scale=s, size=(num_experts, hidden_size)).astype(NP_TYPE)

    onnx_model_full = create_moe_onnx_graph(
        num_rows,
        num_experts,
        num_experts,
        hidden_size,
        inter_size,
        fc1_experts_weights_all,
        fc1_experts_bias_all,
        fc2_experts_weights_all,
        fc2_experts_bias_all,
        fc3_experts_weights_all,
    )

    return (
        onnx_model_full,
        fc1_experts_weights_all,
        fc1_experts_bias_all,
        fc2_experts_weights_all,
        fc2_experts_bias_all,
        fc3_experts_weights_all,
    )


def run_ort_with_parity_check(
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
        ort_session.get_inputs()[0].name: np.random.rand(num_rows, hidden_size).astype(NP_TYPE),
        ort_session.get_inputs()[1].name: np.random.rand(num_rows, num_experts).astype(NP_TYPE),
    }

    output = ort_session.run(None, ort_inputs)
    sharded_output = ort_session_local.run(None, ort_inputs)

    print_out("max diff:", np.max(np.abs(output[0] - sharded_output[0])))
    assert np.allclose(output[0], sharded_output[0], atol=threshold, rtol=threshold)

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


def test_moe_with_tensor_parallelism(
    hidden_size,
    inter_size,
    num_experts,
    num_rows,
    threshold=THRESHOLD_TP,
):
    assert inter_size % get_size() == 0

    (
        onnx_model_full,
        fc1_experts_weights_all,
        fc1_experts_bias_all,
        fc2_experts_weights_all,
        fc2_experts_bias_all,
        fc3_experts_weights_all,
    ) = generate_weights_and_initial_model(
        num_rows,
        num_experts,
        hidden_size,
        inter_size,
    )

    fc1_experts_weights = fc1_experts_weights_all[
        :, :, local_rank * inter_size // get_size() : (local_rank + 1) * inter_size // get_size()
    ]
    fc2_experts_weights = fc2_experts_weights_all[
        :, local_rank * inter_size // get_size() : (local_rank + 1) * inter_size // get_size(), :
    ]
    fc3_experts_weights = fc3_experts_weights_all[
        :, :, local_rank * inter_size // get_size() : (local_rank + 1) * inter_size // get_size()
    ]
    fc1_experts_bias = fc1_experts_bias_all[
        :, local_rank * inter_size // get_size() : (local_rank + 1) * inter_size // get_size()
    ]

    onnx_model_local = create_moe_onnx_graph(
        num_rows,
        num_experts,
        num_experts,
        hidden_size,
        inter_size // get_size(),
        fc1_experts_weights,
        fc1_experts_bias,
        fc2_experts_weights,
        fc2_experts_bias_all,
        fc3_experts_weights,
        tensor_shards=get_size(),
    )

    run_ort_with_parity_check(
        onnx_model_full,
        onnx_model_local,
        num_rows,
        hidden_size,
        num_experts,
        inter_size,
        threshold,
    )


def test_moe_with_expert_parallelism(
    hidden_size,
    inter_size,
    num_experts,
    num_rows,
    threshold=THRESHOLD_EP,
):
    local_experts_start_index = local_rank * num_experts // get_size()

    (
        onnx_model_full,
        fc1_experts_weights_all,
        fc1_experts_bias_all,
        fc2_experts_weights_all,
        fc2_experts_bias_all,
        fc3_experts_weights_all,
    ) = generate_weights_and_initial_model(
        num_rows,
        num_experts,
        hidden_size,
        inter_size,
    )

    fc1_experts_weights = fc1_experts_weights_all[
        local_experts_start_index : local_experts_start_index + num_experts // get_size(), :, :
    ]
    fc2_experts_weights = fc2_experts_weights_all[
        local_experts_start_index : local_experts_start_index + num_experts // get_size(), :, :
    ]
    fc3_experts_weights = fc3_experts_weights_all[
        local_experts_start_index : local_experts_start_index + num_experts // get_size(), :, :
    ]
    fc1_experts_bias = fc1_experts_bias_all[
        local_experts_start_index : local_experts_start_index + num_experts // get_size(), :
    ]

    onnx_model_local = create_moe_onnx_graph(
        num_rows,
        num_experts,
        num_experts // get_size(),
        hidden_size,
        inter_size,
        fc1_experts_weights,
        fc1_experts_bias,
        fc2_experts_weights,
        fc2_experts_bias_all,
        fc3_experts_weights,
        local_experts_start_index,
    )

    run_ort_with_parity_check(
        onnx_model_full,
        onnx_model_local,
        num_rows,
        hidden_size,
        num_experts,
        inter_size,
        threshold,
    )


class TestMoE(unittest.TestCase):
    def test_moe_parallelism(self):
        for hidden_size in [128, 1024]:
            for inter_size in [512, 2048]:
                for num_experts in [64]:
                    for num_rows in [1024]:
                        print_out("EP")
                        test_moe_with_expert_parallelism(
                            hidden_size,
                            inter_size,
                            num_experts,
                            num_rows,
                        )
                        print_out("TP")
                        test_moe_with_tensor_parallelism(
                            hidden_size,
                            inter_size,
                            num_experts,
                            num_rows,
                        )


if __name__ == "__main__":
    unittest.main()
