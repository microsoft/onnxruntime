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


def _test_gemm(func, dtype: str, m: int, n: int, k: int, transa=False, transb=False):
    assert dtype in ["float32", "float16"]

    a_shape = (k, m) if transa else (m, k)
    b_shape = (n, k) if transb else (k, n)

    np.random.seed(0)
    a = (np.random.rand(*a_shape) + 0.5).astype(dtype).astype("float64")
    b = (np.random.rand(*b_shape) + 0.5).astype(dtype).astype("float64")
    ref_c = (a.T if transa else a) @ (b.T if transb else b)

    bound = get_gemm_bound(dtype, a, b, ref_c, transa, transb)

    a = a.astype(dtype)
    b = b.astype(dtype)

    my_c = np.zeros((m, n), dtype=dtype)
    dev_a = ke.DeviceArray(a)
    dev_b = ke.DeviceArray(b)
    dev_c = ke.DeviceArray(my_c)

    opa = ke.blas_op.T if transa else ke.blas_op.N
    opb = ke.blas_op.T if transb else ke.blas_op.N
    lda = a_shape[1]
    ldb = b_shape[1]
    alpha = 1.0
    beta = 0.0
    my_gemm = func(opa, opb, m, n, k, alpha, dev_a, lda, dev_b, ldb, beta, dev_c, n)

    failures = {}
    print(f"dtype={dtype} {transab_to_suffix((transa, transb))} m={m:<5} n={n:<5} k={k:<5} bound: {bound}")

    for impl in my_gemm.ListOps():
        if not my_gemm.SelectOp(impl):
            continue

        my_gemm.Run()
        dev_c.UpdateHostNumpyArray()

        try:
            np.testing.assert_allclose(my_c, ref_c, rtol=bound)
        except Exception as err:
            header = "*" * 30 + impl + "*" * 30
            print(header)
            print(err)
            print("*" * len(header))
            failures[impl] = str(err)

    if failures:
        raise Exception(failures)


dtypes = ["float32", "float16"]
all_transabs = list(product([True, False], repeat=2))


@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("size", get_gemm_basic_sizes(full=True) + get_gemm_bert_sizes(full=False))
@pytest.mark.parametrize("transab", all_transabs)
def test_rocblas_gemm_all_cases(dtype, size, transab):
    _test_gemm(getattr(ke, "RocblasGemm_" + dtype_to_suffix(dtype)), dtype, *size, *transab)


@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("size", get_gemm_basic_sizes(full=False) + get_gemm_bert_sizes(full=False))
@pytest.mark.parametrize("transab", all_transabs)
def test_ck_gemm_bert_cases(dtype, size, transab):
    wrapper_name = "CKGemm_{}_{}".format(dtype_to_suffix(dtype), transab_to_suffix(transab))
    _test_gemm(getattr(ke, wrapper_name), dtype, *size, *transab)


# Tunable is basically wrapped around of rocblas and ck gemm, so no need for full tests
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("size", get_gemm_basic_sizes(full=False) + get_gemm_bert_sizes(full=False))
@pytest.mark.parametrize("transab", all_transabs)
def test_gemm_tunable_bert_cases(dtype, size, transab):
    wrapper_name = "GemmTunable_{}_{}".format(dtype_to_suffix(dtype), transab_to_suffix(transab))
    _test_gemm(getattr(ke, wrapper_name), dtype, *size, *transab)


def profile_gemm_func(f, transa: bool, transb: bool, dtype: str, m: int, n: int, k: int):
    a_shape = (k, m) if transa else (m, k)
    b_shape = (n, k) if transb else (k, n)

    np.random.seed(0)
    a = (np.random.rand(*a_shape) * 2 - 1).astype(dtype)
    b = (np.random.rand(*b_shape) * 2 - 1).astype(dtype)
    my_c = np.zeros((m, n), dtype=dtype)

    dev_a = ke.DeviceArray(a)
    dev_b = ke.DeviceArray(b)
    dev_c = ke.DeviceArray(my_c)

    opa = ke.blas_op.T if transa else ke.blas_op.N
    opb = ke.blas_op.T if transb else ke.blas_op.N
    lda = a_shape[1]
    ldb = b_shape[1]
    alpha = 1.0
    beta = 0.0
    my_gemm = f(opa, opb, m, n, k, alpha, dev_a, lda, dev_b, ldb, beta, dev_c, n)
    for impl in my_gemm.ListOps():
        if not my_gemm.SelectOp(impl):
            print(f"{impl:<50} {dtype} {transab_to_suffix((transa, transb))} m={m:<4} n={n:<4} k={k:<4} not supported")
            sys.stdout.flush()
            continue
        time_ms = my_gemm.Profile()
        time_us = time_ms * 1000
        tflops = (m * k * n * 2) / (time_ms * 1e-3) / 1e12
        print(
            f"{impl:<50} {dtype} {transab_to_suffix((transa, transb))}",
            f"m={m:<4} n={n:<4} k={k:<4} {time_us:>8.4f} us {tflops:>5.2f} tflops",
        )


def profile_with_args(transa, transb, dtype, m, n, k):
    dtype_suffix = "_" + dtype_to_suffix(dtype)
    profile_gemm_func(getattr(ke, "RocblasGemm" + dtype_suffix), transa, transb, dtype, m, n, k)
    transab_suffix = "_" + transab_to_suffix((transa, transb))
    profile_gemm_func(getattr(ke, "CKGemm" + dtype_suffix + transab_suffix), transa, transb, dtype, m, n, k)
    profile_gemm_func(getattr(ke, "GemmTunable" + dtype_suffix + transab_suffix), transa, transb, dtype, m, n, k)


def profile():
    for dtype in dtypes:
        for m, n, k in get_gemm_bert_sizes(full=True):
            profile_with_args(False, False, dtype, m, n, k)
            print()
        print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    group = parser.add_argument_group("profile with args")
    group.add_argument("transa", choices="NT")
    group.add_argument("transb", choices="NT")
    group.add_argument("dtype", choices=dtypes)
    group.add_argument("m", type=int)
    group.add_argument("n", type=int)
    group.add_argument("k", type=int)
    if len(sys.argv) == 1:
        profile()
    else:
        args = parser.parse_args()
        profile_with_args(args.transa == "T", args.transb == "T", args.dtype, args.m, args.n, args.k)
