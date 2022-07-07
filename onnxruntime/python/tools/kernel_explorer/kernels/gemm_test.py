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


def _test_gemm(size, dtype, func):
    m, k, n = size

    np.random.seed(0)
    a = (np.random.rand(m, k).astype(dtype) - 0.5) * 2
    b = (np.random.rand(k, n).astype(dtype) - 0.5) * 2
    c = np.zeros((m, n)).astype(dtype)

    dev_a = ke.DeviceArray(a)
    dev_b = ke.DeviceArray(b)
    dev_c = ke.DeviceArray(c)

    f = getattr(ke, func)
    va = f(ke.blas_op.N, ke.blas_op.N, n, m, k, 1.0, dev_b, n, dev_a, k, 0.0, dev_c, n)
    va.Run()

    dev_c.UpdateHostNumpyArray()

    ref_c = a @ b

    rtol = 1e-3 if dtype == "float16" else 1e-5
    atol = 1e-3 if dtype == "float16" else 1e-5
    np.testing.assert_allclose(ref_c, c, rtol=rtol, atol=atol)


@pytest.mark.parametrize(
    "size",
    # product([5, 9], repeat=3)
    product([1, 3, 4, 16, 124, 125, 126, 127, 128, 129, 130, 131, 132], repeat=3),
)
def test_gemm_all_sizes(size):
    # FIXME: float16 is causing high roundoff error here!
    # dtypes = ["float16", "float32"]
    dtypes = ["float32"]
    for dtype in dtypes:
        for f in dtype_to_funcs(dtype):
            _test_gemm(size, dtype, f)


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
