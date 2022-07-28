# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from itertools import product

import kernel_explorer as ke
import numpy as np
import pytest


def dtype_to_suffix(dtype):
    return {
        "float32": "float",
        "float16": "half",
    }[dtype]


def _test_gemm(func, dtype: str, m: int, n: int, k: int, transa=False, transb=False):
    a_shape = (k, m) if transa else (m, k)
    b_shape = (n, k) if transb else (k, n)

    np.random.seed(0)
    a = (np.random.rand(*a_shape) * 2 - 1).astype(dtype)
    b = (np.random.rand(*b_shape) * 2 - 1).astype(dtype)
    ref_c = (a.T if transa else a) @ (b.T if transb else b)

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
    for impl in my_gemm.ListImpls():
        if not my_gemm.SelectImpl(impl):
            continue

        my_gemm.Run()
        dev_c.UpdateHostNumpyArray()

        rtol = 1e-3 if dtype == "float16" else 1e-5
        atol = 1e-3 if dtype == "float16" else 1e-5

        try:
            np.testing.assert_allclose(ref_c, my_c, rtol=rtol, atol=atol)
        except Exception as err:
            failures[impl] = str(err)

    if failures:
        raise Exception(failures)


def get_basic_cases():
    dtypes = ["float32", "float16"]
    transabs = product([True, False], repeat=2)
    basic_sizes = product([1, 3, 4, 16, 127, 128, 129, 133, 1024], repeat=3)
    return list(product(dtypes, basic_sizes, transabs))


def get_bert_cases():
    dtypes = ["float32", "float16"]
    transabs = [(False, False)]
    bert_sizes = [
        # m, n, k
        (384, 64, 384),
        (384, 768, 768),
        (384, 3072, 768),
        (384, 1024, 1024),
        (384, 4096, 1024),
        (384, 768, 3072),
        (384, 1024, 4096),
    ]
    return list(product(dtypes, bert_sizes, transabs))


@pytest.mark.parametrize(
    "dtype, size, transab",
    get_basic_cases() + get_bert_cases(),
)
def test_rocblas_gemm_all_cases(dtype, size, transab):
    _test_gemm(getattr(ke, "RocblasGemm_" + dtype_to_suffix(dtype)), dtype, *size, *transab)


@pytest.mark.parametrize(
    "dtype, size, transab",
    get_bert_cases(),
)
def test_ck_gemm_bert_cases(dtype, size, transab):
    _test_gemm(getattr(ke, "CKGemm_" + dtype_to_suffix(dtype)), dtype, *size, *transab)


def profile_gemm_func(f, dtype, m, n, k):
    a_shape = (m, k)
    b_shape = (k, n)

    np.random.seed(0)
    a = (np.random.rand(*a_shape) * 2 - 1).astype(dtype)
    b = (np.random.rand(*b_shape) * 2 - 1).astype(dtype)
    my_c = np.zeros((m, n), dtype=dtype)

    dev_a = ke.DeviceArray(a)
    dev_b = ke.DeviceArray(b)
    dev_c = ke.DeviceArray(my_c)

    opa = ke.blas_op.N
    opb = ke.blas_op.N
    lda = a_shape[1]
    ldb = b_shape[1]
    alpha = 1.0
    beta = 0.0
    my_gemm = f(opa, opb, m, n, k, alpha, dev_a, lda, dev_b, ldb, beta, dev_c, n)
    for impl in my_gemm.ListImpls():
        if not my_gemm.SelectImpl(impl):
            print(f"{impl:<50} {dtype} m={m:<4} k={k:<4} n={n:<4}, not supported")
            continue
        time_ms = my_gemm.Profile()
        time_us = time_ms * 1000
        tflops = (m * k * n * 2) / (time_ms * 1e-3) / 1e12
        print(f"{impl:<50} {dtype} m={m:<4} k={k:<4} n={n:<4}, {time_us:>8.4f} us, {tflops:>5.2f} tflops")


def profile():
    for dtype, (m, n, k), _ in get_bert_cases():
        suffix = dtype_to_suffix(dtype)
        profile_gemm_func(getattr(ke, "RocblasGemm_" + suffix), dtype, m, n, k)
        profile_gemm_func(getattr(ke, "CKGemm_" + suffix), dtype, m, n, k)
        print()


if __name__ == "__main__":
    profile()
