import os
from mpi4py import MPI
import onnxruntime
import numpy as np
from onnx import TensorProto, helper

np.random.seed(42)

comm = MPI.COMM_WORLD

def get_rank():
    return comm.Get_rank()


def get_size():
    return comm.Get_size()


def barrier():
    comm.Barrier()


def print_out(*args):
    if get_rank() == 0:
        print(*args)


def broadcast(data):
    comm = MPI.COMM_WORLD
    comm.broadcast(data, root=0)

local_rank = get_rank()

ORT_DTYPE = TensorProto.FLOAT16
NP_TYPE = np.float16 if ORT_DTYPE == TensorProto.FLOAT16 else np.float32
THRESHOLD = 3e-2

def create_moe_onnx_graph(
    num_rows,
    num_experts,
    local_num_experts,
    hidden_size,
    inter_size,
    fc1_experts_weights,
    fc2_experts_weights,
    fc1_experts_bias,
    fc2_experts_bias,
    local_experts_start_index = -1,
):
    use_sharded_moe = True if local_experts_start_index >= 0 else False
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
        ) if not use_sharded_moe else helper.make_node(
            "ShardedMoE",
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
            local_experts_start_index=local_experts_start_index,
            domain="com.microsoft",
        ),
    ]

    fc1_shape = [local_num_experts, hidden_size, inter_size]
    fc2_shape = [local_num_experts, inter_size, hidden_size]

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
    ]

    fc1_bias_shape = [local_num_experts, inter_size]
    fc2_bias_shape = [local_num_experts, hidden_size]
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

# two gpus
def main():
    hidden_size = 32
    inter_size = 64
    num_experts = 16
    num_rows = 32
    local_experts_start_index = local_rank * num_experts // get_size()

    # create weights and bias
    fc1_experts_weights_all = np.random.rand(num_experts, hidden_size, inter_size).astype(NP_TYPE)
    fc2_experts_weights_all = np.random.rand(num_experts, inter_size, hidden_size).astype(NP_TYPE)
    fc1_experts_bias_all = np.random.rand(num_experts, inter_size).astype(NP_TYPE)
    fc2_experts_bias_all = np.random.rand(num_experts, hidden_size).astype(NP_TYPE)

    # expert slicing by local rank
    fc1_experts_weights = fc1_experts_weights_all[local_experts_start_index:local_experts_start_index + num_experts // get_size(), :, :]
    fc2_experts_weights = fc2_experts_weights_all[local_experts_start_index:local_experts_start_index + num_experts // get_size(), :, :]
    fc1_experts_bias = fc1_experts_bias_all[local_experts_start_index:local_experts_start_index + num_experts // get_size(), :]
    fc2_experts_bias = fc2_experts_bias_all[local_experts_start_index:local_experts_start_index + num_experts // get_size(), :]

    # create onnx graph
    onnx_model = create_moe_onnx_graph(
        num_rows,
        num_experts,
        num_experts // get_size(),
        hidden_size,
        inter_size,
        fc1_experts_weights,
        fc2_experts_weights,
        fc1_experts_bias,
        fc2_experts_bias,
        local_experts_start_index,
    )

    sess_options = onnxruntime.SessionOptions()
    cuda_provider_options = {"enable_skip_layer_norm_strict_mode": False, "device_id": local_rank}
    provider_options = {"CUDAExecutionProvider": cuda_provider_options}
    execution_providers = [("CUDAExecutionProvider", cuda_provider_options)]

    ort_session = onnxruntime.InferenceSession(onnx_model, sess_options, providers=execution_providers)

    input_name = ort_session.get_inputs()[0].name
    router_probs_name = ort_session.get_inputs()[1].name
    input_data = np.random.rand(num_rows, hidden_size).astype(NP_TYPE)
    router_probs = np.random.rand(num_rows, num_experts).astype(NP_TYPE)
    ort_inputs = {input_name: input_data, router_probs_name: router_probs}

    output = ort_session.run(None, {input_name: input_data, router_probs_name: router_probs})


if __name__ == "__main__":
    main()