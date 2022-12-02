# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import sys
from itertools import product

import kernel_explorer as ke
import numpy as np
import pytest
from utils import get_gemm_basic_sizes, get_gemm_bert_sizes, get_gemm_bound, transab_to_suffix


def dtype_to_suffix(dtype):
    return {
        "float32": "float",
        "float16": "half",
    }[dtype]


def _test_strided_batched_gemm(func, dtype: str, transa: bool, transb: bool, m: int, n: int, k: int, batch: int):
    assert dtype in ["float32", "float16"]

    a_shape = (k, m) if transa else (m, k)
    b_shape = (n, k) if transb else (k, n)

    np.random.seed(0)
    a = (np.random.rand(batch, *a_shape) + 0.5).astype(dtype).astype("float64")
    b = (np.random.rand(batch, *b_shape) + 0.5).astype(dtype).astype("float64")
    ref_c = (a.swapaxes(1, 2) if transa else a) @ (b.swapaxes(1, 2) if transb else b)

    bounds = [get_gemm_bound(dtype, a[i], b[i], ref_c[i], transa, transb) for i in range(batch)]

    a = a.astype(dtype)
    b = b.astype(dtype)

    my_c = np.zeros((batch, m, n), dtype=dtype)
    dev_a = ke.DeviceArray(a)
    dev_b = ke.DeviceArray(b)
    dev_c = ke.DeviceArray(my_c)

    opa = ke.blas_op.T if transa else ke.blas_op.N
    opb = ke.blas_op.T if transb else ke.blas_op.N
    lda = a_shape[1]
    ldb = b_shape[1]
    ldc = n
    stride_a = m * k
    stride_b = k * n
    stride_c = m * n
    alpha = 1.0
    beta = 0.0
    my_gemm = func(
        opa, opb, m, n, k, alpha, dev_a, lda, stride_a, dev_b, ldb, stride_b, beta, dev_c, ldc, stride_c, batch
    )

    failures = {}
    print(
        f"dtype={dtype} {transab_to_suffix((transa, transb))} m={m:<5} n={n:<5} k={k:<5} batch={batch:<3} max bound: {max(bounds)}"
    )

    for impl in my_gemm.ListOps():
        if not my_gemm.SelectOp(impl):
            continue

        my_gemm.Run()
        dev_c.UpdateHostNumpyArray()

        for i in range(batch):
            try:
                np.testing.assert_allclose(my_c[i], ref_c[i], rtol=bounds[i])
            except Exception as err:
                header = "*" * 30 + impl + "*" * 30
                print(header, bounds[i])
                print(err)
                print("*" * len(header))
                failures[impl] = str(err)

    if failures:
        raise Exception(failures)


dtypes = ["float32", "float16"]
all_transabs = list(product([True, False], repeat=2))


@pytest.mark.parametrize("batch", [1, 64])
@pytest.mark.parametrize("m, n, k", get_gemm_basic_sizes(full=False) + get_gemm_bert_sizes(full=False))
@pytest.mark.parametrize("transa, transb", all_transabs)
@pytest.mark.parametrize("dtype", dtypes)
def test_rocblas_gemm_all_cases(dtype, transa, transb, m, n, k, batch):
    wrapper_name = "RocBlasStridedBatchedGemm_" + dtype_to_suffix(dtype)
    _test_strided_batched_gemm(getattr(ke, wrapper_name), dtype, transa, transb, m, n, k, batch)


@pytest.mark.parametrize("batch", [1, 64])
@pytest.mark.parametrize("m, n, k", get_gemm_bert_sizes(full=False))
# FIXME: ck has bug when any of m,n,k < 8
# @pytest.mark.parametrize("m, n, k", get_gemm_basic_sizes(full=False) + get_gemm_bert_sizes(full=False))
@pytest.mark.parametrize("transa, transb", all_transabs)
@pytest.mark.parametrize("dtype", dtypes)
def test_ck_gemm_bert_cases(dtype, transa, transb, m, n, k, batch):
    wrapper_name = "CKStridedBatchedGemm_{}_{}".format(dtype_to_suffix(dtype), transab_to_suffix((transa, transb)))
    _test_strided_batched_gemm(getattr(ke, wrapper_name), dtype, transa, transb, m, n, k, batch)


# Tunable is basically wrapped around of rocblas and ck gemm, so no need for full tests
@pytest.mark.parametrize("batch", [1, 64])
@pytest.mark.parametrize("m, n, k", get_gemm_bert_sizes(full=False))
@pytest.mark.parametrize("transa, transb", all_transabs)
@pytest.mark.parametrize("dtype", dtypes)
def test_gemm_tunable_bert_cases(dtype, transa, transb, m, n, k, batch):
    wrapper_name = "StridedBatchedGemmTunable_{}_{}".format(dtype_to_suffix(dtype), transab_to_suffix((transa, transb)))
    _test_strided_batched_gemm(getattr(ke, wrapper_name), dtype, transa, transb, m, n, k, batch)


def profile_gemm_func(f, dtype: str, transa: bool, transb: bool, m: int, n: int, k: int, batch: int):
    a_shape = (k, m) if transa else (m, k)
    b_shape = (n, k) if transb else (k, n)

    np.random.seed(0)
    a = (np.random.rand(batch, *a_shape) * 2 - 1).astype(dtype)
    b = (np.random.rand(batch, *b_shape) * 2 - 1).astype(dtype)
    my_c = np.zeros((batch, m, n), dtype=dtype)

    dev_a = ke.DeviceArray(a)
    dev_b = ke.DeviceArray(b)
    dev_c = ke.DeviceArray(my_c)

    opa = ke.blas_op.T if transa else ke.blas_op.N
    opb = ke.blas_op.T if transb else ke.blas_op.N
    lda = a_shape[1]
    ldb = b_shape[1]
    ldc = n
    stride_a = m * k
    stride_b = k * n
    stride_c = m * n
    alpha = 1.0
    beta = 0.0
    my_gemm = f(opa, opb, m, n, k, alpha, dev_a, lda, stride_a, dev_b, ldb, stride_b, beta, dev_c, ldc, stride_c, batch)
    for impl in my_gemm.ListOps():
        if not my_gemm.SelectOp(impl):
            print(
                f"{impl:<50} {dtype} {transab_to_suffix((transa, transb))}",
                f"m={m:<4} n={n:<4} k={k:<4} batch={batch:<3} not supported",
            )
            sys.stdout.flush()
            continue
        time_ms = my_gemm.Profile()
        time_us = time_ms * 1000
        tflops = batch * (m * k * n * 2) / (time_ms * 1e-3) / 1e12
        print(
            f"{impl:<50} {dtype} {transab_to_suffix((transa, transb))}",
            f"m={m:<4} n={n:<4} k={k:<4} batch={batch:<3} {time_us:>8.4f} us {tflops:>5.2f} tflops",
        )


def profile_with_args(dtype, transa, transb, m, n, k, batch):
    dtype_suffix = "_" + dtype_to_suffix(dtype)
    transab_suffix = "_" + transab_to_suffix((transa, transb))
    fn_rocblas = getattr(ke, "RocBlasStridedBatchedGemm" + dtype_suffix)
    fn_ck = getattr(ke, "CKStridedBatchedGemm" + dtype_suffix + transab_suffix)
    fn_tunable = getattr(ke, "StridedBatchedGemmTunable" + dtype_suffix + transab_suffix)
    profile_gemm_func(fn_rocblas, dtype, transa, transb, m, n, k, batch)
    profile_gemm_func(fn_ck, dtype, transa, transb, m, n, k, batch)
    profile_gemm_func(fn_tunable, dtype, transa, transb, m, n, k, batch)
    print()


def profile():
    for dtype in dtypes:
        for m, n, k in get_gemm_bert_sizes(full=False):
            for batch in [1, 32, 64]:
                profile_with_args(dtype, False, False, m, n, k, batch)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    group = parser.add_argument_group("profile with args")
    group.add_argument("dtype", choices=dtypes)
    group.add_argument("transa", choices="NT")
    group.add_argument("transb", choices="NT")
    group.add_argument("m", type=int)
    group.add_argument("n", type=int)
    group.add_argument("k", type=int)
    group.add_argument("batch", type=int)
    if len(sys.argv) == 1:
        profile()
    else:
        args = parser.parse_args()
        profile_with_args(args.dtype, args.transa == "T", args.transb == "T", args.m, args.n, args.k, args.batch)
