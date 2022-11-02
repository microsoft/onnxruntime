# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import sys
from itertools import product

import kernel_explorer as ke
import numpy as np
import pytest


def dtype_to_suffix(dtype):
    return {
        "float16": "half",
    }[dtype]


def transab_to_suffix(transab):
    return {
        (True, True): "TT",
        (True, False): "TN",
        (False, True): "NT",
        (False, False): "NN",
    }[tuple(transab)]

def add_add_fast_gelu(x, bias0, bias1):
    x = x + bias0 + bias1
    y = 0.5 * x * (1 + np.tanh(0.797885 * x + 0.035677 * x * x * x))
    return y

def _test_gemmfastgelu(func, dtype: str, m: int, n: int, k: int, transa=False, transb=False):
    assert dtype in ["float16"]

    a_shape = (k, m) if transa else (m, k)
    b_shape = (n, k) if transb else (k, n)

    np.random.seed(0)
    a = (np.random.rand(*a_shape) - 0.5).astype(dtype).astype("float64")
    b = (np.random.rand(*b_shape) + 0.5).astype(dtype).astype("float64")
    bias0 = np.random.rand(n).astype(dtype)
    bias1 = np.zeros(n, dtype=dtype)
    temp_c = (a.T if transa else a) @ (b.T if transb else b)
    ref_c = add_add_fast_gelu(temp_c, bias0, bias1)
    # ref_c = temp_c

    # The machine epsilon, unit roundoff, the smallest positive floating point number n such that the floating point
    # number that represents 1 + n is greater than 1.
    machine_eps = 2.0 ** -(24 if dtype == "float32" else 11)

    # The following implements error bound 5.7 in paper I. C. Ipsen and H. Zhou, “Probabilistic error analysis for
    # Inner Products,” SIAM Journal on Matrix Analysis and Applications, vol. 41, no. 4, pp. 1726–1741, 2020.
    # NOTE: the bound is not tight for float16 when k is large
    absa_mul_absb = np.abs(a.T if transa else a) @ np.abs(b.T if transb else b)
    coeff = np.max(absa_mul_absb / np.abs(ref_c))
    gamma_2k = (1.0 + machine_eps) ** (2 * k) - 1.0
    bound_5_7 = coeff * np.sqrt(np.log(2 / 1e-10) * machine_eps * gamma_2k / 2)
    bound = bound_5_7

    a = a.astype(dtype)
    b = b.astype(dtype)

    my_c = np.zeros((m, n), dtype=dtype)
    dev_a = ke.DeviceArray(a)
    dev_b = ke.DeviceArray(b)
    dev_bias0 = ke.DeviceArray(bias0)
    dev_bias1 = ke.DeviceArray(bias1)
    dev_c = ke.DeviceArray(my_c)

    opa = ke.blas_op.T if transa else ke.blas_op.N
    opb = ke.blas_op.T if transb else ke.blas_op.N
    lda = a_shape[1]
    ldb = b_shape[1]
    alpha = 1.0
    beta = 0.0
    my_gemmfastgelu = func(opa, opb, m, n, k, alpha, dev_a, lda, dev_b, ldb, dev_bias0, dev_bias1, beta, dev_c, n)

    failures = {}
    print(f"dtype={dtype} {transab_to_suffix((transa, transb))} m={m:<5} n={n:<5} k={k:<5} bound: {0.005}")

    for impl in my_gemmfastgelu.ListOps():
        if not my_gemmfastgelu.SelectOp(impl):
            continue

        my_gemmfastgelu.Run()
        dev_c.UpdateHostNumpyArray()

        try:
            np.testing.assert_allclose(my_c, ref_c, rtol=0)
        except Exception as err:
            header = "*" * 30 + impl + "*" * 30
            print(header)
            print(err)
            print("*" * len(header))
            failures[impl] = str(err)

    if failures:
        raise Exception(failures)


dtypes = ["float16"]
all_transabs = list(product([True, False], repeat=2))
all_basic_sizes = list(product([1, 3, 4, 16, 127, 128, 129, 133, 1024], repeat=3))


def get_bert_sizes(full=True):
    bert_base_sizes = [
        # m, n, k
        (384, 768, 768),
        (384, 768, 768 * 3),
        (384, 768, 768 * 4),
        (384, 768 * 4, 768),
        (384, 1024, 1024),
        (384, 1024, 1024 * 3),
        (384, 1024, 1024 * 4),
        (384, 1024 * 4, 1024),
    ]

    # we then multiply m with the batch size
    if full:
        batch_sizes = [1, 64]
    else:
        batch_sizes = [1]
    bert_sizes = []
    for bsz in batch_sizes:
        bert_sizes.extend([(m * bsz, n, k) for m, n, k in bert_base_sizes])
    return bert_sizes

# ck has various impls to be tested, use the full basic cases will result too many cases to test.
# So we use a reduced combination here.
reduced_basic_sizes = list(product([1, 4, 127, 133], [3, 16, 128], [3, 129, 1024]))


@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("size", reduced_basic_sizes + get_bert_sizes(full=False))
@pytest.mark.parametrize("transab", all_transabs)
def test_ck_gemmfastgelu_bert_cases(dtype, size, transab):
    wrapper_name = "CKGemmFastGelu_{}_{}".format(dtype_to_suffix(dtype), transab_to_suffix(transab))
    _test_gemmfastgelu(getattr(ke, wrapper_name), dtype, *size, *transab)


def profile_gemmfastgelu_func(f, transa: bool, transb: bool, dtype: str, m: int, n: int, k: int):
    a_shape = (k, m) if transa else (m, k)
    b_shape = (n, k) if transb else (k, n)

    np.random.seed(0)
    a = (np.random.rand(*a_shape) * 2 - 1).astype(dtype)
    b = (np.random.rand(*b_shape) * 2 - 1).astype(dtype)
    my_c = np.zeros((m, n), dtype=dtype)
    bias0 = np.random.rand(n).astype(dtype)
    bias1 = np.zeros(n, dtype=dtype)

    dev_a = ke.DeviceArray(a)
    dev_b = ke.DeviceArray(b)
    dev_bias0 = ke.DeviceArray(bias0)
    dev_bias1 = ke.DeviceArray(bias1)
    dev_c = ke.DeviceArray(my_c)

    opa = ke.blas_op.T if transa else ke.blas_op.N
    opb = ke.blas_op.T if transb else ke.blas_op.N
    lda = a_shape[1]
    ldb = b_shape[1]
    alpha = 1.0
    beta = 0.0
    my_gemmfastgelu = f(opa, opb, m, n, k, alpha, dev_a, lda, dev_b, ldb, dev_bias0, dev_bias1, beta, dev_c, n)
    for impl in my_gemmfastgelu.ListOps():
        if not my_gemmfastgelu.SelectOp(impl):
            print(f"{impl:<50} {dtype} {transab_to_suffix((transa, transb))} m={m:<4} n={n:<4} k={k:<4} not supported")
            sys.stdout.flush()
            continue
        time_ms = my_gemmfastgelu.Profile()
        time_us = time_ms * 1000
        tflops = (m * k * n * 2) / (time_ms * 1e-3) / 1e12
        print(
            f"{impl:<50} {dtype} {transab_to_suffix((transa, transb))}",
            f"m={m:<4} n={n:<4} k={k:<4} {time_us:>8.4f} us {tflops:>5.2f} tflops",
        )


def profile_with_args(transa, transb, dtype, m, n, k):
    dtype_suffix = "_" + dtype_to_suffix(dtype)
    transab_suffix = "_" + transab_to_suffix((transa, transb))
    profile_gemmfastgelu_func(getattr(ke, "CKgemmfastgelu" + dtype_suffix + transab_suffix), transa, transb, dtype, m, n, k)


def profile():
    for dtype in dtypes:
        for m, n, k in get_bert_sizes(full=True):
            profile_with_args(False, False, dtype, m, n, k)
            print()
        print()


if __name__ == "__main__":
    # import argparse

    # parser = argparse.ArgumentParser()
    # group = parser.add_argument_group("profile with args")
    # group.add_argument("transa", choices="NT")
    # group.add_argument("transb", choices="NT")
    # group.add_argument("dtype", choices=dtypes)
    # group.add_argument("m", type=int)
    # group.add_argument("n", type=int)
    # group.add_argument("k", type=int)
    # if len(sys.argv) == 1:
    #     profile()
    # else:
    #     args = parser.parse_args()
    #     profile_with_args(args.transa == "T", args.transb == "T", args.dtype, args.m, args.n, args.k)
    test_ck_gemmfastgelu_bert_cases("float16", (3840, 4096, 4096), (False, False))
