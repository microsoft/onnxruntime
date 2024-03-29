# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------


import os
import sys
from dataclasses import dataclass
from itertools import product

import kernel_explorer as ke
import numpy as np
import pytest
from utils import dtype_to_suffix, matmul, softmax

max_batch_size = int(os.environ.get("KERNEL_EXPLORER_BATCHED_GEMM_MAX_BATCH_SIZE", 64))


def multinormal_distribution(num_distribution, num_element_per_dist):
    arrays = []
    for _ in range(num_distribution):
        mean = np.random.rand() - 0.5
        std = np.random.rand() + 0.5
        arrays.append(np.random.normal(mean, std, (num_element_per_dist,)))
    return np.array(arrays)


def get_ck_binding_name(dtype):
    dtype_suffix = "_" + dtype_to_suffix(dtype)
    return "GroupQueryAttentionCK" + dtype_suffix


dtypes = ["float16"]
batches = [1, max_batch_size]
seqlens = [128, 512]
total_seqlens = [128, 512]
num_heads = [32]
num_kv_heads = [32]
head_sizes = [128]

def maybe_pack_q_k_v_bnsh_for_device_on_host(q, k, v, dtype, qkv_format):
    q = q.astype(dtype)
    k = k.astype(dtype)
    v = v.astype(dtype)
    if qkv_format == ke.qkv_format.Q_K_V_BNSH:
        return q, k, v

    # BNSH to BSNH
    q = np.swapaxes(q, 2, 1)
    k = np.swapaxes(k, 2, 1)
    v = np.swapaxes(v, 2, 1)

    if qkv_format == ke.qkv_format.Q_K_V_BSNH:
        return np.ascontiguousarray(q), np.ascontiguousarray(k), np.ascontiguousarray(v)

    if qkv_format == ke.qkv_format.QKV_BSN3H:
        return np.ascontiguousarray(np.stack([q, k, v], axis=-2)), None, None

    if qkv_format == ke.qkv_format.Q_KV_BSNH_BSN2H:
        return np.ascontiguousarray(q), np.ascontiguousarray(np.stack([k, v], axis=-2)), None

    raise NotImplementedError

@dataclass
class GroupQueryAttentionMetric(ke.ComputeMetric):
    batch: int
    seqlen: int
    total_seqlen: int
    num_heads: int
    head_size: int
    num_kv_heads: int

    def report(self):
        common = (
            f"{self.dtype} B={self.batch} S={self.seqlen} T={self.total_seqlen} "
            f"N={self.num_heads} H={self.head_size} Nk={self.num_kv_heads}, "
            f"{self.name}"
        )
        if self.duration <= 0:
            return "not supported          " + common

        return f"{self.duration:>6.2f} us {self.tflops:>5.2f} tflops " + common


def profile_group_query_attention_func(
    f, dtype, batch, seqlen, total_seqlen, num_heads, head_size, num_kv_heads, scale, qkv_format
):
    v_head_size = head_size
    q_shape = [batch, num_heads, seqlen, head_size]
    k_shape = [batch, num_kv_heads, total_seqlen, head_size]
    v_shape = [batch, num_kv_heads, total_seqlen, v_head_size]
    out_shape = [batch, seqlen, num_heads, head_size]

    np.random.seed(42)
    q = multinormal_distribution(np.prod(q_shape[:-1]), q_shape[-1]).reshape(q_shape).astype(np.float64)
    k = multinormal_distribution(np.prod(k_shape[:-1]), k_shape[-1]).reshape(k_shape).astype(np.float64)
    v = multinormal_distribution(np.prod(v_shape[:-1]), v_shape[-1]).reshape(v_shape).astype(np.float64)

    out = np.empty(out_shape, dtype=dtype)
    host_q, host_k, host_v = maybe_pack_q_k_v_bnsh_for_device_on_host(q, k, v, dtype, qkv_format)
    dev_q = ke.DeviceArray(host_q)
    dev_k = ke.DeviceArray(host_k) if host_k is not None else None
    dev_v = ke.DeviceArray(host_v) if host_v is not None else None
    dev_out = ke.DeviceArray(out)

    my_func = f(
        batch,
        seqlen,
        total_seqlen,
        num_heads,
        head_size,
        num_kv_heads,
        scale,
        qkv_format,
        dev_q,
        dev_k,
        dev_v,
        dev_out,
    )

    for impl in my_func.ListOps():
        duration_ms = -1
        if my_func.SelectOp(impl):
            duration_ms = my_func.Profile()

        m, n, k, o, gemm_batch = seqlen, total_seqlen, head_size, head_size, batch * num_heads
        flops_per_batch = m * n * k * 2 + m * n * o * 2
        flops = flops_per_batch * gemm_batch

        ke.report(
            GroupQueryAttentionMetric(
                impl, dtype, duration_ms, flops, batch, seqlen, total_seqlen, num_heads, head_size, num_kv_heads
            )
        )


def profile_with_args(
    dtype, batch, seqlen, total_seqlen, num_heads, head_size, num_kv_heads, scale, qkv_format, *, sort=False
):
    with ke.benchmark(sort):
        args = (dtype, batch, seqlen, total_seqlen, num_heads, head_size, num_kv_heads, scale, qkv_format)
        if ke.is_composable_kernel_available():
            profile_group_query_attention_func(
                getattr(ke, get_ck_binding_name(dtype)), *args
            )
        profile_group_query_attention_func(
            getattr(ke, "GroupQueryAttentionTunable_" + dtype_to_suffix(dtype)), *args
        )


def profile():
    for args in product(dtypes, batches, seqlens, total_seqlens, num_heads, head_sizes, num_kv_heads):
        profile_with_args(*args, qkv_format=ke.qkv_format.Q_K_V_BNSH, scale=0.125, sort=True)
        print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    group = parser.add_argument_group("profile with args")
    group.add_argument("--sort", action="store_true")
    group.add_argument("dtype", choices=dtypes)
    group.add_argument("batch", type=int)
    group.add_argument("seqlen", type=int)
    group.add_argument("total_seqlen", type=int)
    group.add_argument("num_heads", type=int)
    group.add_argument("head_size", type=int)
    group.add_argument("num_kv_heads", type=int)
    group.add_argument("--scale", type=float, default=None, help="default to 1.0/sqrt(head_size)")
    group.add_argument(
        "--qkv_format",
        default="Q_K_V_BNSH",
        choices=[
            "Q_K_V_BNSH",  # non-packed, permuted
            "Q_K_V_BSNH",  # non-packed, non-permuted
            "Q_KV_BSNH_BSN2H",  # kv packed, non-permuted
            "QKV_BSN3H",  # qkv packed, non-permuted
        ],
    )

    if len(sys.argv) == 1:
        profile()
    else:
        args = parser.parse_args()
        print(args)
        profile_with_args(
            args.dtype,
            args.batch,
            args.seqlen,
            args.total_seqlen,
            args.num_heads,
            args.head_size,
            args.num_kv_heads,
            args.scale,
            getattr(ke.qkv_format, args.qkv_format),
            sort=args.sort,
        )
