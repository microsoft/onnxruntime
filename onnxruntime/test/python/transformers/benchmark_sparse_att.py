# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""
Benchmark performance of SparseAttention. Requires Nvidia GPU of Compute Capability 8.x.
Install required packages before running this script:
   pip install matplotlib pandas onnx torch onnxruntime-gpu
"""

#from typing import Optional

import torch
from test_sparse_attention import (
    # AttentionConfig,
    # GroupQueryAttentionConfig,
    OrtGroupQueryAttention,
    OrtSparseAttention,
    SparseAttentionConfig,
    TorchGroupQueryAttention,
    # create_group_query_attention_onnx_model,
    # create_session,
    # create_sparse_attention_onnx_model,
    # get_block_mask,
    # get_dense_mask,
    # group_query_attention_reference,
)

# from torch import Tensor
# from onnxruntime import InferenceSession, SessionOptions
# from onnxruntime.transformers.io_binding_helper import CudaSession, GpuBindingManager


def get_plot_algos(sm: int):
    # GQA with local windows only works in sm=8x
    if sm >= 80:
        return {
            "line_vals": ["torch_gqa", "ort_gqa", "ort_gqa_local", "ort_sparse_att"],
            "line_names": ["TORCH-GQA", "ORT-GQA-Dense", "ORT-GQA-Local", "ORT-SparseAtt"],
            "styles": [("red", "-"), ("blue", "-"), ("yellow", "-"), ("green", "-")],
        }
    else:
        return {
            "line_vals": ["torch_gqa", "ort_gqa", "ort_sparse_att"],
            "line_names": ["TORCH-GQA", "ORT-GQA-Dense", "ORT-SparseAtt"],
            "styles": [("red", "-"), ("blue", "-"), ("green", "-")],
        }


def plot_prompt_performance(
    sm: int,
    batch_size=4,
    num_heads=32,
    max_seq_len=8192,
    head_size=128,
    sparse_block_size=64,
    local_blocks=16,
    vert_stride=8,
    num_layout=8,
    dtype=torch.float16,
):
    import triton

    algos = get_plot_algos(sm)
    configs = [
        triton.testing.Benchmark(
            x_names=["sequence_length"],
            x_vals=[2**i for i in range(4, 14)],
            line_arg="provider",
            ylabel="ms",
            **algos,
            plot_name=f"prompt-sm{sm}-batch{batch_size}-head{num_heads}-d{head_size}-local{local_blocks}-vert{vert_stride}-{dtype}",
            args={"num_heads": num_heads, "batch_size": batch_size, "head_size": head_size, "dtype": dtype},
        )
    ]

    @triton.testing.perf_report(configs)
    def benchmark(batch_size, num_heads, sequence_length, head_size, provider, dtype=torch.float16, device="cuda"):
        warmup = 15
        repeat = 100

        config: SparseAttentionConfig = SparseAttentionConfig(
            batch_size=batch_size,
            sequence_length=sequence_length,
            max_sequence_length=max_seq_len,
            past_sequence_length=0,
            num_heads=num_heads,
            kv_num_heads=8,
            head_size=head_size,
            sparse_block_size=sparse_block_size,
            num_layout=num_layout,
            local_blocks=local_blocks,
            vert_stride=vert_stride,
        )

        if provider in ["ort_gqa", "ort_gqa_local"]:
            gqa_config = config.get_comparable_ort_gqa_config(use_local=(provider == "ort_gqa_local"))
            obj = OrtGroupQueryAttention(gqa_config)
        elif provider == "ort_sparse_att":
            obj = OrtSparseAttention(config)
        else:  # Torch GQA
            assert provider == "torch_gqa"
            if sequence_length > 2048:  # out of memory
                return 0
            gqa_config = config.get_comparable_torch_gqa_config(use_sparse=True)
            obj = TorchGroupQueryAttention(gqa_config)

        ms = triton.testing.do_bench(obj.infer, warmup=warmup, rep=repeat)
        return ms

    benchmark.run(save_path=".", print_data=True)


def plot_token_performance(
    sm: int,
    batch_size=4,
    num_heads=32,
    max_seq_len=8192,
    head_size=128,
    sparse_block_size=64,
    local_blocks=16,
    vert_stride=8,
    num_layout=8,
    dtype=torch.float16,
):
    import triton

    algos = get_plot_algos(sm)
    configs = [
        triton.testing.Benchmark(
            x_names=["past_sequence_length"],
            x_vals=[2**i for i in range(4, 13)] + [max_seq_len - 1],
            line_arg="provider",
            ylabel="ms",
            **algos,
            plot_name=f"token-sm{sm}-batch{batch_size}-head{num_heads}-d{head_size}-local{local_blocks}-vert{vert_stride}-{dtype}",
            args={"num_heads": num_heads, "batch_size": batch_size, "head_size": head_size, "dtype": dtype},
        )
    ]

    @triton.testing.perf_report(configs)
    def benchmark(batch_size, num_heads, past_sequence_length, head_size, provider, dtype=torch.float16, device="cuda"):
        warmup = 15
        repeat = 100

        config: SparseAttentionConfig = SparseAttentionConfig(
            batch_size=batch_size,
            sequence_length=1,
            max_sequence_length=max_seq_len,
            past_sequence_length=past_sequence_length,
            num_heads=num_heads,
            kv_num_heads=8,
            head_size=head_size,
            sparse_block_size=sparse_block_size,
            num_layout=num_layout,
            local_blocks=local_blocks,
            vert_stride=vert_stride,
        )

        if provider in ["ort_gqa", "ort_gqa_local"]:
            gqa_config = config.get_comparable_ort_gqa_config(use_local=(provider == "ort_gqa_local"))
            obj = OrtGroupQueryAttention(gqa_config)
        elif provider == "ort_sparse_att":
            obj = OrtSparseAttention(config)
        else:
            assert provider == "torch_gqa"
            if past_sequence_length > 2048:  # out of memory
                return 0
            gqa_config = config.get_comparable_torch_gqa_config(use_sparse=True)
            obj = TorchGroupQueryAttention(gqa_config)

        ms = triton.testing.do_bench(obj.infer, warmup=warmup, rep=repeat)
        return ms

    benchmark.run(save_path=".", print_data=True)


def run_performance_test(sm: int):
    """
    Run performance tests for prompt and token generation.

    Example results in Azure Standard_ND96isr_H100_v5 VM with NVIDIA H100-80GB-HBM3 GPU (sm=90):

    prompt-sm90-batch4-head32-d128-local16-vert8-torch.float16:
       sequence_length  TORCH-GQA  ORT-GQA-Dense  ORT-GQA-Local  ORT-SparseAtt
    0             16.0   0.079877       0.006362       0.006403       0.042758
    1             32.0   0.086920       0.016404       0.016686       0.044183
    2             64.0   0.090727       0.020429       0.020409       0.045343
    3            128.0   0.128148       0.032009       0.031984       0.051516
    4            256.0   0.323933       0.074110       0.073920       0.068308
    5            512.0   1.021856       0.162167       0.161951       0.109226
    6           1024.0   3.596002       0.452629       0.452780       0.231653
    7           2048.0  13.865088       1.499534       1.195749       0.515488
    8           4096.0   0.000000       5.454785       2.669682       1.163233
    9           8192.0   0.000000      22.068159       6.018604       2.772873

    token-sm90-batch4-head32-d128-local16-vert8-torch.float16:
       past_sequence_length  TORCH-GQA  ORT-GQA-Dense  ORT-GQA-Local  ORT-SparseAtt
    0                  16.0   0.104460       0.012652       0.012661       0.069549
    1                  32.0   0.113866       0.012776       0.012765       0.069024
    2                  64.0   0.124600       0.016791       0.012672       0.069397
    3                 128.0   0.108658       0.017900       0.018294       0.074844
    4                 256.0   0.115463       0.029409       0.029608       0.078911
    5                 512.0   0.149824       0.033968       0.033701       0.092998
    6                1024.0   0.234050       0.042930       0.042951       0.116920
    7                2048.0   0.390695       0.061462       0.043008       0.121555
    8                4096.0   0.000000       0.097505       0.042948       0.134757
    9                8191.0   0.000000       0.165861       0.043542       0.158796


    Example results in A100-SXM4-80GB (sm=80):

    prompt-sm80-batch4-head32-d128-local16-vert8-torch.float16:
       sequence_length  TORCH-GQA  ORT-GQA-Dense  ORT-GQA-Local  ORT-SparseAtt
    0             16.0   0.274839       0.008849       0.015198       0.054403
    1             32.0   0.272238       0.022875       0.048804       0.055898
    2             64.0   0.272420       0.027722       0.028318       0.073052
    3            128.0   0.273514       0.085971       0.062785       0.068287
    4            256.0   0.545428       0.108228       0.135093       0.095949
    5            512.0   1.678597       0.278193       0.248580       0.167271
    6           1024.0   6.021056       0.702882       0.701022       0.379936
    7           2048.0  23.512320       2.331175       1.863045       0.895726
    8           4096.0   0.000000       8.789178       4.526275       2.105048
    9           8192.0   0.000000      39.664131      10.046236       5.219436

    token-sm80-batch4-head32-d128-local16-vert8-torch.float16:
       past_sequence_length  TORCH-GQA  ORT-GQA-Dense  ORT-GQA-Local  ORT-SparseAtt
    0                  16.0   0.299303       0.020081       0.018587       0.082479
    1                  32.0   0.301700       0.018655       0.041943       0.084583
    2                  64.0   0.305700       0.017825       0.018420       0.085265
    3                 128.0   0.303379       0.023213       0.023152       0.090508
    4                 256.0   0.304119       0.034438       0.035257       0.100197
    5                 512.0   0.306051       0.063312       0.045373       0.114726
    6                1024.0   0.359197       0.092181       0.088628       0.145165
    7                2048.0   0.599463       0.101573       0.062101       0.159452
    8                4096.0   0.000000       0.196258       0.091019       0.180342
    9                8191.0   0.000000       0.334519       0.065158       0.213508


    Example results in Standard_NC4as_T4_v3 Azure VM with T4 GPU (sm=75):

    prompt-sm75-batch4-head32-d128-local16-vert8-torch.float16:
       sequence_length   TORCH-GQA  ORT-GQA-Dense  ORT-SparseAtt
    0             16.0    0.165154       3.003173       0.081945
    1             32.0    0.184173       2.994347       0.089064
    2             64.0    0.303300       3.023986       0.107418
    3            128.0    0.887795       3.073728       0.174213
    4            256.0    2.797654       3.246899       0.357869
    5            512.0   10.055048       3.814039       0.893903
    6           1024.0   37.849937       5.818439       2.658720
    7           2048.0  148.641785      13.638480       7.202690
    8           4096.0    0.000000      43.556847      17.680954
    9           8192.0    0.000000     161.628540      44.336670

    token-sm75-batch4-head32-d128-local16-vert8-torch.float16:
       past_sequence_length  TORCH-GQA  ORT-GQA-Dense  ORT-SparseAtt
    0                  16.0   0.144368       4.179228       0.137407
    1                  32.0   0.110353       2.996305       0.137509
    2                  64.0   0.145088       3.006860       0.165424
    3                 128.0   0.219500       3.036448       0.192001
    4                 256.0   0.347496       3.071341       0.249125
    5                 512.0   0.595842       3.135225       0.398726
    6                1024.0   1.081216       3.261110       0.612744
    7                2048.0   2.060307       3.515578       0.685670
    8                4096.0   0.000000       4.022986       0.819707
    9                8191.0   0.000000       5.024528       1.072912


    """
    with torch.no_grad():
        plot_prompt_performance(sm=sm)
        plot_token_performance(sm=sm)


if __name__ == "__main__":
    torch.set_printoptions(precision=6, edgeitems=3, linewidth=150, profile="default", sci_mode=False)

    major, minor = torch.cuda.get_device_capability()
    sm = major * 10 + minor

    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        run_performance_test(sm)
