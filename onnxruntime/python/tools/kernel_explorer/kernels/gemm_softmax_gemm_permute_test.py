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


def get_ck_binding_name(dtype, biased: bool, masked: bool):
    dtype_suffix = "_" + dtype_to_suffix(dtype)
    ck_suffix = ""
    if biased:
        ck_suffix += "Biased"
    if masked:
        ck_suffix += "Masked"
    ck_suffix += dtype_suffix
    return "GemmSoftmaxGemmPermuteCK" + ck_suffix


dtypes = ["float16"]
batches = [1, max_batch_size]
seqlens = [128, 512]
total_seqlens = [128, 512]
num_heads = [8, 12]
head_sizes = [64]
biaseds = [False, True]
mask_dims = [0, 2, 3, 4]


def get_biased_id(biased):
    return "biased" if biased else "nobias"


def get_mask_dim_id(dim):
    if dim == 0:
        return "nomask"
    return f"mask_{dim}d"


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


def _test_gemm_softmax_gemm_permute(
    f, dtype, batch, seqlen, total_seqlen, num_heads, head_size, biased, mask_dim, scale, qkv_format
):
    v_head_size = head_size
    q_shape = [batch, num_heads, seqlen, head_size]
    k_shape = [batch, num_heads, total_seqlen, head_size]
    v_shape = [batch, num_heads, total_seqlen, v_head_size]
    out_shape = [batch, seqlen, num_heads, head_size]

    attn_bias = None
    bias_shape = [batch, num_heads, seqlen, total_seqlen] if biased else None

    attn_mask = None
    mask_shape = None
    mask_shape_broadcasted = None
    max_seqlen = None
    if mask_dim != 0:
        if mask_dim == 2:
            mask_shape = [batch, total_seqlen]
            mask_shape_broadcasted = [batch, 1, 1, total_seqlen]
        elif mask_dim == 3:
            mask_shape = [batch, seqlen, total_seqlen]
            mask_shape_broadcasted = [batch, 1, seqlen, total_seqlen]
        elif mask_dim == 4:
            max_seqlen = ((seqlen - 1) // 1024 + 1) * 1024  # round up to multiple of 1024
            mask_shape = [batch, 1, max_seqlen, max_seqlen]
        else:
            raise ValueError

    np.random.seed(42)
    q = multinormal_distribution(np.prod(q_shape[:-1]), q_shape[-1]).reshape(q_shape).astype(np.float64)
    k = multinormal_distribution(np.prod(k_shape[:-1]), k_shape[-1]).reshape(k_shape).astype(np.float64)
    v = multinormal_distribution(np.prod(v_shape[:-1]), v_shape[-1]).reshape(v_shape).astype(np.float64)
    if bias_shape is not None:
        attn_bias = np.random.uniform(-0.5, 0.5, size=bias_shape)
    if mask_shape is not None:
        attn_mask = (np.random.randint(0, 100, size=mask_shape) < 95).astype(np.int32)

    pre_softmax_attn_scores = matmul(q, np.swapaxes(k, 2, 3))
    pre_softmax_attn_scores = pre_softmax_attn_scores * scale
    if attn_bias is not None:
        pre_softmax_attn_scores = pre_softmax_attn_scores + attn_bias
    if attn_mask is not None:
        filter_value = -10000.0
        if mask_dim == 4:
            # equivalent to past_sequence_length = max_sequence_length - seqlen
            converted_mask = (1 - attn_mask[:, :, -seqlen:, :total_seqlen]) * filter_value
        else:
            converted_mask = (1 - attn_mask.reshape(mask_shape_broadcasted)) * filter_value
        pre_softmax_attn_scores = pre_softmax_attn_scores + converted_mask
    attn_scores = softmax(pre_softmax_attn_scores, axis=-1)
    attn = matmul(attn_scores, v)
    ref = np.swapaxes(attn, 2, 1)  # permute 0213

    out = np.empty(out_shape, dtype=dtype)
    host_q, host_k, host_v = maybe_pack_q_k_v_bnsh_for_device_on_host(q, k, v, dtype, qkv_format)
    host_attn_bias = attn_bias.astype(dtype) if attn_bias is not None else None
    dev_q = ke.DeviceArray(host_q)
    dev_k = ke.DeviceArray(host_k) if host_k is not None else None
    dev_v = ke.DeviceArray(host_v) if host_v is not None else None
    dev_out = ke.DeviceArray(out)
    dev_attn_bias = ke.DeviceArray(host_attn_bias) if host_attn_bias is not None else None
    dev_attn_mask = ke.DeviceArray(attn_mask) if attn_mask is not None else None

    my_gemm_softmax_gemm_permute = f(
        batch,
        seqlen,
        total_seqlen,
        max_seqlen,
        num_heads,
        head_size,
        mask_dim,
        scale,
        qkv_format,
        dev_q,
        dev_k,
        dev_v,
        dev_attn_bias,
        dev_attn_mask,
        dev_out,
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
                is_zero_tol, atol, rtol = 1e-3, 2e-2, 1e-2
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


@pytest.mark.parametrize("mask_dim", mask_dims, ids=get_mask_dim_id)
@pytest.mark.parametrize("biased", biaseds, ids=get_biased_id)
@pytest.mark.parametrize("head_size", head_sizes)
@pytest.mark.parametrize("nhead", num_heads)
@pytest.mark.parametrize("total_seqlen", total_seqlens)
@pytest.mark.parametrize("seqlen", seqlens)
@pytest.mark.parametrize("batch", [16])
@pytest.mark.parametrize("dtype", ["float16", "float32"])
def test_gemm_softmax_gemm_permute_generic(dtype, batch, seqlen, total_seqlen, nhead, head_size, biased, mask_dim):
    f = getattr(ke, "GemmSoftmaxGemmPermuteGeneric_" + dtype_to_suffix(dtype))
    scale = 1.0 / np.sqrt(head_size)
    _test_gemm_softmax_gemm_permute(
        f, dtype, batch, seqlen, total_seqlen, nhead, head_size, biased, mask_dim, scale, ke.qkv_format.Q_K_V_BNSH
    )


@pytest.mark.parametrize("mask_dim", [2], ids=get_mask_dim_id)
@pytest.mark.parametrize("biased", [False], ids=get_biased_id)
@pytest.mark.parametrize("head_size", [64])
@pytest.mark.parametrize("nhead", [8])
@pytest.mark.parametrize("total_seqlen", [128])
@pytest.mark.parametrize("seqlen", [64])
@pytest.mark.parametrize("batch", [16])
@pytest.mark.parametrize("dtype", ["float16", "float32"])
def test_gemm_softmax_gemm_permute_generic_nested_tunable(
    dtype, batch, seqlen, total_seqlen, nhead, head_size, biased, mask_dim
):
    f = getattr(ke, "GemmSoftmaxGemmPermuteGenericNestedTunable_" + dtype_to_suffix(dtype))
    scale = 1.0 / np.sqrt(head_size)
    _test_gemm_softmax_gemm_permute(
        f, dtype, batch, seqlen, total_seqlen, nhead, head_size, biased, mask_dim, scale, ke.qkv_format.Q_K_V_BNSH
    )


@pytest.mark.skipif(not ke.is_composable_kernel_available(), reason="ck is not enabled")
@pytest.mark.parametrize("mask_dim", mask_dims, ids=get_mask_dim_id)
@pytest.mark.parametrize("biased", biaseds, ids=get_biased_id)
@pytest.mark.parametrize("head_size", head_sizes)
@pytest.mark.parametrize("nhead", num_heads)
@pytest.mark.parametrize("total_seqlen", total_seqlens)
@pytest.mark.parametrize("seqlen", seqlens)
@pytest.mark.parametrize("batch", batches)
@pytest.mark.parametrize("dtype", dtypes)
def test_gemm_softmax_gemm_permute_ck(dtype, batch, seqlen, total_seqlen, nhead, head_size, biased, mask_dim):
    f = getattr(ke, get_ck_binding_name(dtype, biased, mask_dim != 0))
    scale = 1.0 / np.sqrt(head_size)
    _test_gemm_softmax_gemm_permute(
        f, dtype, batch, seqlen, total_seqlen, nhead, head_size, biased, mask_dim, scale, ke.qkv_format.Q_K_V_BNSH
    )


@pytest.mark.parametrize("mask_dim", [2], ids=get_mask_dim_id)
@pytest.mark.parametrize("biased", [False], ids=get_biased_id)
@pytest.mark.parametrize("head_size", [64])
@pytest.mark.parametrize("nhead", [8])
@pytest.mark.parametrize("total_seqlen", [128])
@pytest.mark.parametrize("seqlen", [64])
@pytest.mark.parametrize("batch", [16])
@pytest.mark.parametrize("dtype", ["float16"])
def test_gemm_softmax_gemm_permute_tunable(dtype, batch, seqlen, total_seqlen, nhead, head_size, biased, mask_dim):
    f = getattr(ke, "GemmSoftmaxGemmPermuteTunable_" + dtype_to_suffix(dtype))
    scale = 1.0 / np.sqrt(head_size)
    _test_gemm_softmax_gemm_permute(
        f, dtype, batch, seqlen, total_seqlen, nhead, head_size, biased, mask_dim, scale, ke.qkv_format.Q_K_V_BNSH
    )


stabel_diffusion_configs = [
    [2, 64, 64, 8, 160, "QKV_BSN3H"],
    [2, 256, 256, 8, 160, "QKV_BSN3H"],
    [2, 1024, 1024, 8, 80, "QKV_BSN3H"],
    [2, 4096, 4096, 8, 40, "QKV_BSN3H"],
    [2, 64, 77, 8, 160, "Q_KV_BSNH_BSN2H"],
    [2, 256, 77, 8, 160, "Q_KV_BSNH_BSN2H"],
    [2, 1024, 77, 8, 80, "Q_KV_BSNH_BSN2H"],
    [2, 4096, 77, 8, 40, "Q_KV_BSNH_BSN2H"],
    [1, 4096, 4096, 1, 512, "Q_K_V_BNSH"],
]


@pytest.mark.skipif(not ke.is_composable_kernel_available(), reason="ck is not enabled")
@pytest.mark.parametrize("mask_dim", [0], ids=get_mask_dim_id)
@pytest.mark.parametrize("biased", [False], ids=get_biased_id)
@pytest.mark.parametrize("batch, seqlen, total_seqlen, nhead, head_size, qkv_format_name", stabel_diffusion_configs)
@pytest.mark.parametrize("dtype", dtypes)
def test_gemm_softmax_gemm_permute_ck_sd(
    dtype, batch, seqlen, total_seqlen, nhead, head_size, biased, mask_dim, qkv_format_name
):
    qkv_format = getattr(ke.qkv_format, qkv_format_name)
    f = getattr(ke, get_ck_binding_name(dtype, biased, mask_dim != 0))
    scale = 1.0 / np.sqrt(head_size)
    _test_gemm_softmax_gemm_permute(
        f, dtype, batch, seqlen, total_seqlen, nhead, head_size, biased, mask_dim, scale, qkv_format
    )


@dataclass
class GemmSoftmaxGemmPermuteMetric(ke.ComputeMetric):
    batch: int
    seqlen: int
    total_seqlen: int
    num_heads: int
    head_size: int
    biased: bool
    mask_dim: int

    def report(self):
        bias_str = " biased" if self.biased else ""
        mask_str = f" mask_{self.mask_dim}d" if self.mask_dim != 0 else ""
        common = (
            f"{self.dtype} B={self.batch} S={self.seqlen} T={self.total_seqlen} "
            f"N={self.num_heads} H={self.head_size}{bias_str}{mask_str}, "
            f"{self.name}"
        )
        if self.duration <= 0:
            return "not supported          " + common

        return f"{self.duration:>6.2f} us {self.tflops:>5.2f} tflops " + common


def profile_gemm_softmax_gemm_permute_func(
    f, dtype, batch, seqlen, total_seqlen, num_heads, head_size, biased, mask_dim, scale, qkv_format
):
    v_head_size = head_size
    q_shape = [batch, num_heads, seqlen, head_size]
    k_shape = [batch, num_heads, total_seqlen, head_size]
    v_shape = [batch, num_heads, total_seqlen, v_head_size]
    out_shape = [batch, seqlen, num_heads, head_size]

    attn_bias = None
    bias_shape = [batch, num_heads, seqlen, total_seqlen] if biased else None

    attn_mask = None
    mask_shape = None
    max_seqlen = None
    if mask_dim != 0:
        if mask_dim == 2:
            mask_shape = [batch, total_seqlen]
        elif mask_dim == 3:
            mask_shape = [batch, seqlen, total_seqlen]
        elif mask_dim == 4:
            max_seqlen = ((seqlen - 1) // 1024 + 1) * 1024  # round up to multiple of 1024
            mask_shape = [batch, 1, max_seqlen, max_seqlen]
        else:
            raise ValueError

    np.random.seed(42)
    q = multinormal_distribution(np.prod(q_shape[:-1]), q_shape[-1]).reshape(q_shape).astype(np.float64)
    k = multinormal_distribution(np.prod(k_shape[:-1]), k_shape[-1]).reshape(k_shape).astype(np.float64)
    v = multinormal_distribution(np.prod(v_shape[:-1]), v_shape[-1]).reshape(v_shape).astype(np.float64)
    if bias_shape is not None:
        attn_bias = np.random.uniform(-2, 2, size=bias_shape)
    if mask_shape is not None:
        attn_mask = (np.random.randint(0, 100, size=mask_shape) < 95).astype(np.int32)

    out = np.empty(out_shape, dtype=dtype)
    host_q, host_k, host_v = maybe_pack_q_k_v_bnsh_for_device_on_host(q, k, v, dtype, qkv_format)
    host_attn_bias = attn_bias.astype(dtype) if attn_bias is not None else None
    dev_q = ke.DeviceArray(host_q)
    dev_k = ke.DeviceArray(host_k) if host_k is not None else None
    dev_v = ke.DeviceArray(host_v) if host_v is not None else None
    dev_out = ke.DeviceArray(out)
    dev_attn_bias = ke.DeviceArray(host_attn_bias) if host_attn_bias is not None else None
    dev_attn_mask = ke.DeviceArray(attn_mask) if attn_mask is not None else None

    my_gemm_softmax_gemm_permute = f(
        batch,
        seqlen,
        total_seqlen,
        max_seqlen,
        num_heads,
        head_size,
        mask_dim,
        scale,
        qkv_format,
        dev_q,
        dev_k,
        dev_v,
        dev_attn_bias,
        dev_attn_mask,
        dev_out,
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
                impl, dtype, duration_ms, flops, batch, seqlen, total_seqlen, num_heads, head_size, biased, mask_dim
            )
        )


def profile_with_args(
    dtype, batch, seqlen, total_seqlen, num_heads, head_size, biased, mask_dim, scale, qkv_format, *, sort=False
):
    with ke.benchmark(sort):
        args = (dtype, batch, seqlen, total_seqlen, num_heads, head_size, biased, mask_dim, scale, qkv_format)
        if qkv_format == ke.qkv_format.Q_K_V_BNSH:
            profile_gemm_softmax_gemm_permute_func(
                getattr(ke, "GemmSoftmaxGemmPermuteGeneric_" + dtype_to_suffix(dtype)), *args
            )
        if ke.is_composable_kernel_available():
            profile_gemm_softmax_gemm_permute_func(
                getattr(ke, get_ck_binding_name(dtype, biased, mask_dim != 0)), *args
            )
        profile_gemm_softmax_gemm_permute_func(
            getattr(ke, "GemmSoftmaxGemmPermuteTunable_" + dtype_to_suffix(dtype)), *args
        )


def profile():
    for batch, seqlen, total_seqlen, nhead, head_size, qkv_format_name in stabel_diffusion_configs:
        profile_with_args(
            "float16",
            batch,
            seqlen,
            total_seqlen,
            nhead,
            head_size,
            biased=False,
            mask_dim=0,
            qkv_format=getattr(ke.qkv_format, qkv_format_name),
            scale=0.125,
            sort=True,
        )
        print()

    for args in product(dtypes, batches, seqlens, total_seqlens, num_heads, head_sizes, biaseds, mask_dims):
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
    group.add_argument("biased", type=int, choices=[0, 1], default=0)
    group.add_argument("mask_dim", type=int, choices=[0, 2, 3, 4], default=2, help="0 for mask disabled")
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
        profile_with_args(
            args.dtype,
            args.batch,
            args.seqlen,
            args.total_seqlen,
            args.num_heads,
            args.head_size,
            args.biased,
            args.mask_dim,
            args.scale,
            getattr(ke.qkv_format, args.qkv_format),
            sort=args.sort,
        )
