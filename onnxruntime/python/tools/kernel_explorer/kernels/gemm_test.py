# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import sys
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
    my_gemm = f(opb, opa, n, m, k, alpha, dev_b, ldb, dev_a, lda, beta, dev_c, n)
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


def profile_gemm_func(size, dtype, func):
    m, k, n = size

    np.random.seed(0)
    a = np.random.rand(m, k).astype(dtype)
    b = np.random.rand(k, n).astype(dtype)
    c = np.zeros((m, n)).astype(dtype)

    dev_a = ke.DeviceArray(a)
    dev_b = ke.DeviceArray(b)
    dev_c = ke.DeviceArray(c)

    f = getattr(ke, func)
    va = f(ke.blas_op.N, ke.blas_op.N, n, m, k, 1.0, dev_b, n, dev_a, k, 0.0, dev_c, n)
    t = va.Profile()

    print(dtype, size, f, f"{t*1000:.2f} us")  # , f"{size*3*(dtype_to_bytes(dtype))*1e3/t/1e9:.2f} GB/s")


def profile():
    sizes = product(list(range(64, 512 + 1, 64)), repeat=3)
    dtypes = ["float16", "float32"]
    for dt in dtypes:
        for f in dtype_to_funcs(dt):
            for s in sizes:
                profile_gemm_func(s, dt, f)
            print()
        print()


if __name__ == "__main__":
    profile()
