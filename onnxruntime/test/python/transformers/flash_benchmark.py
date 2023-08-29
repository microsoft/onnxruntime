# Install the newest triton version with
# pip install "git+https://github.com/openai/triton.git#egg=triton&subdirectory=python"
import math
import time

import torch
from onnx import TensorProto, helper

from onnxruntime import InferenceSession, SessionOptions


class Config:
    batch_size = 0
    sequence_length = 0
    kv_sequence_length = 0
    num_heads = 0
    head_size = 0

    def __init__(self, b, s, s2, n, h):
        self.batch_size = b
        self.sequence_length = s
        self.kv_sequence_length = s2
        self.num_heads = n
        self.head_size = h


def create_multihead_attention_graph(config):
    nodes = [
        helper.make_node(
            "MultiHeadAttention",
            [
                "query",
                "key",
                "value",
            ],
            ["output"],
            "MultiHeadAttention_0",
            num_heads=config.num_heads,
            domain="com.microsoft",
        ),
    ]

    graph = helper.make_graph(
        nodes,
        "MultiHeadAttention_Graph",
        [
            helper.make_tensor_value_info(
                "query",
                TensorProto.FLOAT16,
                [
                    config.batch_size,
                    config.sequence_length,
                    config.num_heads * config.head_size,
                ],
            ),
            helper.make_tensor_value_info(
                "key",
                TensorProto.FLOAT16,
                [
                    config.batch_size,
                    config.kv_sequence_length,
                    config.num_heads * config.head_size,
                ],
            ),
            helper.make_tensor_value_info(
                "value",
                TensorProto.FLOAT16,
                [
                    config.batch_size,
                    config.kv_sequence_length,
                    config.num_heads * config.head_size,
                ],
            ),
        ],
        [
            helper.make_tensor_value_info(
                "output",
                TensorProto.FLOAT16,
                [config.batch_size, config.sequence_length, config.num_heads * config.head_size],
            ),
        ],
    )

    model = helper.make_model(graph)
    return model.SerializeToString()


def flash_attn_func(q, k, v, config, causal=False):
    onnx_model_str = create_multihead_attention_graph(config)
    q = torch.reshape(q, (config.batch_size, config.sequence_length, -1))
    k = torch.reshape(k, (config.batch_size, config.kv_sequence_length, -1))
    v = torch.reshape(v, (config.batch_size, config.kv_sequence_length, -1))
    ort_inputs = {
        "query": q.detach().cpu().numpy(),
        "key": k.detach().cpu().numpy(),
        "value": v.detach().cpu().numpy(),
    }
    sess_options = SessionOptions()
    ort_session = InferenceSession(onnx_model_str, sess_options, providers=["CUDAExecutionProvider"])
    start = time.time()
    ort_session.run(None, ort_inputs)
    end = time.time()
    return end - start


def flops(batch, seqlen, headdim, nheads, causal):
    f = 4 * batch * seqlen**2 * nheads * headdim // (2 if causal else 1)
    return f


def efficiency(flop, time):
    return (flop / time / 10**12) if not math.isnan(time) else 0.0


repeats = 30
device = "cuda"
dtype = torch.float16

bs_seqlen_vals = [(32, 512), (16, 1024), (8, 2048), (4, 4096), (2, 8192), (1, 16384)]
causal_vals = [False]  # , True]
headdim_vals = [64, 128]
dim = 2048

time_f = {}
time_b = {}
time_f_b = {}
speed_f = {}
speed_b = {}
speed_f_b = {}
for causal in causal_vals:
    for headdim in headdim_vals:
        for batch_size, seqlen in bs_seqlen_vals:
            nheads = dim // headdim
            config = Config(batch_size, seqlen, seqlen, nheads, headdim)
            qkv = torch.randn(batch_size, seqlen, 3, nheads, headdim, device=device, dtype=dtype)
            q, k, v = qkv.unbind(dim=2)
            time_f = flash_attn_func(q, k, v, config, causal)
            print(f"### causal={causal}, headdim={headdim}, batch_size={batch_size}, seqlen={seqlen} ###")
            speed = efficiency(flops(batch_size, seqlen, headdim, nheads, causal), time_f)
            print(f"Flash2 fwd: {speed:.2f} TFLOPs/s")


# with open('flash2_attn_time.plk', 'wb') as fp:
#     pickle.dump((speed_f, speed_b, speed_f_b), fp, protocol=pickle.HIGHEST_PROTOCOL)
