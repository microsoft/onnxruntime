# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------


import math
import os
import sys
from dataclasses import dataclass
from itertools import product

import kernel_explorer as ke
import numpy as np
import pytest
import scipy
from utils import dtype_to_suffix


def multinormal_distribution(num_distribution, num_element_per_dist):
    arrays = []
    for i in range(num_distribution):
        mean = np.random.rand() - 0.5
        std = np.random.rand() + 0.5
        arrays.append(np.random.normal(mean, std, (num_element_per_dist,)))
    return np.array(arrays)


def get_ck_binding_name(dtype, masked: bool, biased: bool):
    dtype_suffix = "_" + dtype_to_suffix(dtype)
    ck_suffix = ""
    if masked:
        ck_suffix += "Masked"
    if biased:
        ck_suffix += "Biased"
    ck_suffix += dtype_suffix
    return "GemmSoftmaxGemmPermuteCK" + ck_suffix


dtypes = ["float16"]
batches = [1, 64]
seqlens = [128, 512]
total_seqlens = [128, 512]
num_heads = [8, 12]
head_sizes = [64]
mask_dims = [2, 3]


def _test_gemm_softmax_gemm_permute(f, dtype, batch, seqlen, total_seqlen, num_heads, head_size, mask_dim, scale):
    v_head_size = head_size
    q_shape = [batch, num_heads, seqlen, head_size]
    k_shape = [batch, num_heads, total_seqlen, head_size]
    v_shape = [batch, num_heads, total_seqlen, v_head_size]
    out_shape = [batch, seqlen, num_heads, head_size]

    attn_bias = None
    bias_shape = None
    # TODO: add bias

    attn_mask = None
    mask_shape = None
    mask_shape_broadcasted = None
    if mask_dim != 0:
        if mask_dim == 2:
            mask_shape = [batch, total_seqlen]
            mask_shape_broadcasted = [batch, 1, 1, total_seqlen]
        elif mask_dim == 3:
            mask_shape = [batch, seqlen, total_seqlen]
            mask_shape_broadcasted = [batch, 1, seqlen, total_seqlen]
        elif mask_dim == 4:
            max_seqlen = 1024
            mask_shape = [batch, 1, max_seqlen, max_seqlen]
            mask_shape_broadcasted = [batch, 1, max_seqlen, max_seqlen]
        else:
            raise ValueError

    np.random.seed(42)
    Q = multinormal_distribution(math.prod(q_shape[:-1]), q_shape[-1]).reshape(q_shape).astype(np.float64)
    K = multinormal_distribution(math.prod(k_shape[:-1]), k_shape[-1]).reshape(k_shape).astype(np.float64)
    V = multinormal_distribution(math.prod(v_shape[:-1]), v_shape[-1]).reshape(v_shape).astype(np.float64)
    if mask_shape is not None:
        attn_mask = (np.random.randint(0, 100, size=mask_shape) < 95).astype(np.int32)

    pre_softmax_attn_scores = Q @ np.swapaxes(K, 2, 3)
    if attn_mask is not None:
        filter_value = -10000.0
        converted_mask = (1 - attn_mask.reshape(mask_shape_broadcasted)) * filter_value
        pre_softmax_attn_scores = pre_softmax_attn_scores + converted_mask
    attn_scores = scipy.special.softmax(pre_softmax_attn_scores * scale, axis=-1)
    attn = attn_scores @ V
    ref = np.swapaxes(attn, 2, 1)  # permute 0213

    out = np.empty(out_shape, dtype=dtype)
    host_Q = Q.astype(dtype)
    host_K = K.astype(dtype)
    host_V = V.astype(dtype)
    dev_Q = ke.DeviceArray(host_Q)
    dev_K = ke.DeviceArray(host_K)
    dev_V = ke.DeviceArray(host_V)
    dev_out = ke.DeviceArray(out)
    dev_attn_mask = None
    if attn_mask is not None:
        dev_attn_mask = ke.DeviceArray(attn_mask)

    my_gemm_softmax_gemm_permute = f(
        batch, seqlen, total_seqlen, num_heads, head_size, mask_dim, scale, dev_Q, dev_K, dev_V, dev_attn_mask, dev_out
    )

    print()  # write an empty line in case pytest ... -s -v
    failures = {}
    for impl in my_gemm_softmax_gemm_permute.ListOps():
        if not my_gemm_softmax_gemm_permute.SelectOp(impl):
            print("Unsupport", impl)
            continue
        print("  Support", impl)

        my_gemm_softmax_gemm_permute.Run()
        dev_out.UpdateHostNumpyArray()

        try:
            is_strict = int(os.environ.get("KERNEL_EXPLORER_STRICT_TEST", "0"))
            if is_strict:
                # NOTE: this will always fail, just for manual checking with:
                #  KERNEL_EXPLORER_STRICT_TEST=1 pytest ... -s -v
                np.testing.assert_allclose(out, ref)
            else:
                is_zero_tol, atol, rtol = 1e-3, 1e-2, 1e-2
                not_close_to_zeros = np.abs(ref) > is_zero_tol
                np.testing.assert_allclose(out[not_close_to_zeros], ref[not_close_to_zeros], atol=atol, rtol=rtol)
        except Exception as err:
            header = "*" * 30 + impl + "*" * 30
            print(header)
            print(err)
            print("*" * len(header))
            failures[impl] = str(err)

    if failures:
        raise Exception(failures)


@pytest.mark.parametrize("mask_dim", mask_dims)
@pytest.mark.parametrize("head_size", head_sizes)
@pytest.mark.parametrize("nhead", num_heads)
@pytest.mark.parametrize("total_seqlen", total_seqlens)
@pytest.mark.parametrize("seqlen", seqlens)
@pytest.mark.parametrize("batch", [16])
@pytest.mark.parametrize("dtype", dtypes)
def test_gemm_softmax_gemm_permute_generic(dtype, batch, seqlen, total_seqlen, nhead, head_size, mask_dim):
    func = getattr(ke, "GemmSoftmaxGemmPermuteGeneric_" + dtype_to_suffix(dtype))
    scale = 1.0 / math.sqrt(head_size)
    _test_gemm_softmax_gemm_permute(func, dtype, batch, seqlen, total_seqlen, nhead, head_size, mask_dim, scale)


@pytest.mark.parametrize("mask_dim", mask_dims)
@pytest.mark.parametrize("head_size", head_sizes)
@pytest.mark.parametrize("nhead", num_heads)
@pytest.mark.parametrize("total_seqlen", total_seqlens)
@pytest.mark.parametrize("seqlen", seqlens)
@pytest.mark.parametrize("batch", batches)
@pytest.mark.parametrize("dtype", dtypes)
def test_gemm_softmax_gemm_permute_ck(dtype, batch, seqlen, total_seqlen, nhead, head_size, mask_dim):
    func = getattr(ke, get_ck_binding_name(dtype, mask_dim != 0, False))
    scale = 1.0 / math.sqrt(head_size)
    _test_gemm_softmax_gemm_permute(func, dtype, batch, seqlen, total_seqlen, nhead, head_size, mask_dim, scale)


@pytest.mark.parametrize("mask_dim", [2, 3])
@pytest.mark.parametrize("head_size", [64])
@pytest.mark.parametrize("nhead", [8])
@pytest.mark.parametrize("total_seqlen", [128])
@pytest.mark.parametrize("seqlen", [64])
@pytest.mark.parametrize("batch", [16])
@pytest.mark.parametrize("dtype", ["float16"])
def test_gemm_softmax_gemm_permute_tunable(dtype, batch, seqlen, total_seqlen, nhead, head_size, mask_dim):
    func = getattr(ke, "GemmSoftmaxGemmPermuteTunable_" + dtype_to_suffix(dtype))
    scale = 1.0 / math.sqrt(head_size)
    _test_gemm_softmax_gemm_permute(func, dtype, batch, seqlen, total_seqlen, nhead, head_size, mask_dim, scale)


@dataclass
class GemmSoftmaxGemmPermuteMetric(ke.ComputeMetric):
    batch: int
    seqlen: int
    total_seqlen: int
    num_heads: int
    head_size: int
    mask_dim: int

    def report(self):
        common = (
            f"{self.dtype} B={self.batch} S={self.seqlen} T={self.total_seqlen} "
            + f"N={self.num_heads} H={self.head_size} mask_{self.mask_dim}d, "
            + f"{self.name}"
        )
        if self.duration <= 0:
            return "not supported          " + common

        return f"{self.duration:>6.2f} us {self.tflops:>5.2f} tflops " + common


def profile_gemm_softmax_gemm_permute_func(
    f, dtype, batch, seqlen, total_seqlen, num_heads, head_size, mask_dim, scale
):
    v_head_size = head_size
    q_shape = [batch, num_heads, seqlen, head_size]
    k_shape = [batch, num_heads, total_seqlen, head_size]
    v_shape = [batch, num_heads, total_seqlen, v_head_size]
    out_shape = [batch, seqlen, num_heads, head_size]

    attn_bias = None
    bias_shape = None
    # TODO: add bias

    attn_mask = None
    mask_shape = None
    if mask_dim != 0:
        if mask_dim == 2:
            mask_shape = [batch, total_seqlen]
        elif mask_dim == 3:
            mask_shape = [batch, seqlen, total_seqlen]
        elif mask_dim == 4:
            max_seqlen = 1024
            mask_shape = [batch, 1, max_seqlen, max_seqlen]
        else:
            raise ValueError

    np.random.seed(42)
    Q = multinormal_distribution(math.prod(q_shape[:-1]), q_shape[-1]).reshape(q_shape).astype(np.float64)
    K = multinormal_distribution(math.prod(k_shape[:-1]), k_shape[-1]).reshape(k_shape).astype(np.float64)
    V = multinormal_distribution(math.prod(v_shape[:-1]), v_shape[-1]).reshape(v_shape).astype(np.float64)
    if mask_shape is not None:
        attn_mask = (np.random.randint(0, 100, size=mask_shape) < 95).astype(np.int32)

    out = np.empty(out_shape, dtype=dtype)
    host_Q = Q.astype(dtype)
    host_K = K.astype(dtype)
    host_V = V.astype(dtype)
    dev_Q = ke.DeviceArray(host_Q)
    dev_K = ke.DeviceArray(host_K)
    dev_V = ke.DeviceArray(host_V)
    dev_out = ke.DeviceArray(out)
    dev_attn_mask = None
    if attn_mask is not None:
        dev_attn_mask = ke.DeviceArray(attn_mask)

    my_gemm_softmax_gemm_permute = f(
        batch, seqlen, total_seqlen, num_heads, head_size, mask_dim, scale, dev_Q, dev_K, dev_V, dev_attn_mask, dev_out
    )

    for impl in my_gemm_softmax_gemm_permute.ListOps():
        duration_ms = -1
        if my_gemm_softmax_gemm_permute.SelectOp(impl):
            duration_ms = my_gemm_softmax_gemm_permute.Profile()

        m, n, k, o, gemm_batch = seqlen, total_seqlen, head_size, head_size, batch * num_heads
        flops_per_batch = m * n * k * 2 + m * n * o * 2
        flops_count_bias_and_softmax = True  # set to false to be aligned with ck
        if flops_count_bias_and_softmax:
            flops_per_batch += 2 * n + 1
        if flops_count_bias_and_softmax and attn_bias is not None:
            flops_per_batch += m * n
        if flops_count_bias_and_softmax and attn_mask is not None:
            flops_per_batch += m * n
        flops = flops_per_batch * gemm_batch

        ke.report(
            GemmSoftmaxGemmPermuteMetric(
                impl, dtype, duration_ms, flops, batch, seqlen, total_seqlen, num_heads, head_size, mask_dim
            )
        )


def profile_with_args(dtype, batch, seqlen, total_seqlen, num_heads, head_size, mask_dim, scale, *, sort=False):
    biased = False
    with ke.benchmark(sort):
        args = (dtype, batch, seqlen, total_seqlen, num_heads, head_size, mask_dim, scale)
        profile_gemm_softmax_gemm_permute_func(
            getattr(ke, "GemmSoftmaxGemmPermuteGeneric_" + dtype_to_suffix(dtype)), *args
        )
        profile_gemm_softmax_gemm_permute_func(getattr(ke, get_ck_binding_name(dtype, mask_dim != 0, biased)), *args)
        profile_gemm_softmax_gemm_permute_func(
            getattr(ke, "GemmSoftmaxGemmPermuteTunable_" + dtype_to_suffix(dtype)), *args
        )


def profile():
    for args in product(dtypes, batches, seqlens, total_seqlens, num_heads, head_sizes, mask_dims):
        profile_with_args(*args, scale=0.125, sort=True)
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
    group.add_argument("mask_dim", type=int, choices=[0, 2, 3, 4], default=2, help="0 for mask disabled")
    group.add_argument("scale", type=float, default=None, help="default to 1.0/sqrt(head_size)")

    if len(sys.argv) == 1:
        profile()
    else:
        args = parser.parse_args()
        profile_with_args(
            args.dtype,
            args.batch,
            args.seqlen,
            args.total_seqlen,
            args.num_heads,
            args.head_size,
            args.mask_dim,
            args.scale,
            sort=args.sort,
        )
