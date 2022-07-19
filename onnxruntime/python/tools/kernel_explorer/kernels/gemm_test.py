# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from itertools import product

import kernel_explorer as ke
import numpy as np
import pytest


def dtype_to_bytes(dtype):
    type_map = {
        "float16": 2,
        "float32": 4,
    }
    return type_map[dtype]


def dtype_to_funcs(dtype):
    type_map = {
        "float16": list(filter(lambda x: "RocblasGemm_half" in x, dir(ke))),
        "float32": list(filter(lambda x: "RocblasGemm_float" in x, dir(ke))),
    }
    return type_map[dtype]


def _test_gemm(func, dtype: str, m: int, n: int, k: int, transa=False, transb=False):
    a_shape = (k, m) if transa else (m, k)
    b_shape = (n, k) if transb else (k, n)

    np.random.seed(0)
    a = (np.random.rand(*a_shape) * 2 - 1).astype(dtype)
    b = (np.random.rand(*b_shape) * 2 - 1).astype(dtype)

    my_c = np.zeros((m, n), dtype=dtype)
    dev_a = ke.DeviceArray(a)
    dev_b = ke.DeviceArray(b)
    dev_c = ke.DeviceArray(my_c)

    f = getattr(ke, func)

    opa = ke.blas_op.T if transa else ke.blas_op.N
    opb = ke.blas_op.T if transb else ke.blas_op.N
    lda = a_shape[1]
    ldb = b_shape[1]
    alpha = 1.0
    beta = 0.0
    my_gemm = f(opa, opb, m, n, k, alpha, dev_a, lda, dev_b, ldb, beta, dev_c, n)
    my_gemm.Run()
    dev_c.UpdateHostNumpyArray()

    rtol = 1e-3 if dtype == "float16" else 1e-5
    atol = 1e-3 if dtype == "float16" else 1e-5

    print(a, b)
    ref_c = (a.T if transa else a) @ (b.T if transb else b)
    np.testing.assert_allclose(ref_c, my_c, rtol=rtol, atol=atol)


dtypes = ["float32", "float16"]
transabs = product([True, False], repeat=2)
basic_sizes = product([1, 3, 4, 16, 127, 128, 129, 133, 1024], repeat=3)


@pytest.mark.parametrize(
    "dtype, size, transab",
    product(dtypes, basic_sizes, transabs),
)
def test_gemm_all_cases(dtype, size, transab):
    for f in dtype_to_funcs(dtype):
        _test_gemm(f, dtype, *size, *transab)


def profile_gemm_func(func, dtype, m, n, k):
    a_shape = (m, k)
    b_shape = (k, n)

    np.random.seed(0)
    a = (np.random.rand(*a_shape) * 2 - 1).astype(dtype)
    b = (np.random.rand(*b_shape) * 2 - 1).astype(dtype)
    my_c = np.zeros((m, n), dtype=dtype)

    dev_a = ke.DeviceArray(a)
    dev_b = ke.DeviceArray(b)
    dev_c = ke.DeviceArray(my_c)

    f = getattr(ke, func)
    opa = ke.blas_op.N
    opb = ke.blas_op.N
    lda = a_shape[1]
    ldb = b_shape[1]
    alpha = 1.0
    beta = 0.0
    my_gemm = f(opa, opb, m, n, k, alpha, dev_a, lda, dev_b, ldb, beta, dev_c, n)
    time_ms = my_gemm.Profile()

    time_us = time_ms * 1000
    tflops = (m * k * n * 2) / (time_ms * 1e-3) / 1e12

    print(f"RocBLAS GEMM {dtype} m={m:<4} k={k:<4} n={n:<4}, {time_us:>8.4f} us, {tflops:>5.2f} tflops")


def profile():
    dtypes = ["float32", "float16"]
    bert_sizes = [
        # m, k, n
        (384, 384, 64),
        (384, 768, 768),
        (384, 768, 3072),
        (384, 1024, 1024),
        (384, 1024, 4096),
        (384, 3072, 768),
        (384, 4096, 1024),
    ]
    for dtype in dtypes:
        for f in dtype_to_funcs(dtype):
            for m, k, n in bert_sizes:
                profile_gemm_func(f, dtype, m, k, n)
            print()
        print()


if __name__ == "__main__":
    profile()
