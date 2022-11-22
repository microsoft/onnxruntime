# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import re
import sys
from itertools import product

import kernel_explorer as ke
import numpy as np
import pytest
from utils import get_gemm_basic_sizes, get_gemm_bert_sizes, get_gemm_bound, transab_to_suffix


def dtype_to_funcs(dtype):
    type_map = {
        "float16": list(filter(lambda x: re.search("GemmFastGelu.*_half", x), dir(ke))),
        "float32": list(filter(lambda x: re.search("GemmFastGelu.*_float", x), dir(ke))),
    }
    return type_map[dtype]


def fast_gelu(x, bias):
    x = x + bias
    y = 0.5 * x * (1 + np.tanh(0.797885 * x + 0.035677 * x * x * x))
    return y


# TODO The test method needs update.
def _test_gemmfastgelu(func, dtype: str, m: int, n: int, k: int, transa=False, transb=False):
    assert dtype in ["float16", "float32"]

    a_shape = (k, m) if transa else (m, k)
    b_shape = (n, k) if transb else (k, n)

    np.random.seed(0)
    a = (np.random.rand(*a_shape)).astype(dtype).astype("float64")
    b = (np.random.rand(*b_shape)).astype(dtype).astype("float64")
    bias = (np.random.rand(n)).astype(dtype)
    temp_c = (a.T if transa else a) @ (b.T if transb else b)

    bound = get_gemm_bound(dtype, a, b, temp_c, transa, transb)

    temp_c = temp_c.astype(dtype)
    ref_c = fast_gelu(temp_c, bias)

    a = a.astype(dtype)
    b = b.astype(dtype)

    my_c = np.zeros((m, n), dtype=dtype)
    dev_a = ke.DeviceArray(a)
    dev_b = ke.DeviceArray(b)
    dev_bias = ke.DeviceArray(bias)
    dev_c = ke.DeviceArray(my_c)

    opa = ke.blas_op.T if transa else ke.blas_op.N
    opb = ke.blas_op.T if transb else ke.blas_op.N
    lda = a_shape[1]
    ldb = b_shape[1]
    alpha = 1.0
    beta = 0.0
    my_func = getattr(ke, func)
    my_op = my_func(opa, opb, m, n, k, alpha, dev_a, lda, dev_b, ldb, dev_bias, beta, dev_c, n)

    if my_op.IsSupported():
        my_op.Run()
        dev_c.UpdateHostNumpyArray()

        print(
            f"{func:<50} : dtype={dtype} {transab_to_suffix((transa, transb))} m={m:<5} n={n:<5} k={k:<5} bound: {bound}"
        )

        np.testing.assert_allclose(my_c, ref_c, rtol=max(bound, 1e-2))


dtypes = ["float16", "float32"]
all_transabs = list(product([True, False], repeat=2))


@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("size", get_gemm_basic_sizes(full=False) + get_gemm_bert_sizes(full=False))
@pytest.mark.parametrize("transab", all_transabs)
def test_gemmfastgelu_bert_cases(dtype, size, transab):
    for func in dtype_to_funcs(dtype):
        _test_gemmfastgelu(func, dtype, *size, *transab)


def profile_gemmfastgelu_func(func, dtype: str, m: int, n: int, k: int, transa: bool, transb: bool):
    a_shape = (k, m) if transa else (m, k)
    b_shape = (n, k) if transb else (k, n)

    np.random.seed(0)
    a = (np.random.rand(*a_shape) * 2 - 1).astype(dtype)
    b = (np.random.rand(*b_shape) * 2 - 1).astype(dtype)
    my_c = np.zeros((m, n), dtype=dtype)
    bias = np.random.rand(n).astype(dtype)

    dev_a = ke.DeviceArray(a)
    dev_b = ke.DeviceArray(b)
    dev_bias = ke.DeviceArray(bias)
    dev_c = ke.DeviceArray(my_c)

    opa = ke.blas_op.T if transa else ke.blas_op.N
    opb = ke.blas_op.T if transb else ke.blas_op.N
    lda = a_shape[1]
    ldb = b_shape[1]
    alpha = 1.0
    beta = 0.0
    my_func = getattr(ke, func)
    my_op = my_func(opa, opb, m, n, k, alpha, dev_a, lda, dev_b, ldb, dev_bias, beta, dev_c, n)

    if my_op.IsSupported():
        my_op.Run()
        dev_c.UpdateHostNumpyArray()

        time_ms = my_op.Profile()
        time_us = time_ms * 1000
        # only counts gemm tflops because fastgelu is low order term (7 * n).
        tflops = (m * k * n * 2) / (time_ms * 1e-3) / 1e12
        print(
            f"{func:<50} {dtype} {transab_to_suffix((transa, transb))}",
            f"m={m:<4} n={n:<4} k={k:<4} {time_us:>8.4f} us {tflops:>5.2f} tflops",
        )


def profile_with_args(transa, transb, dtype, m, n, k):
    for func in dtype_to_funcs(dtype):
        profile_gemmfastgelu_func(func, dtype, m, n, k, transa, transb)


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
