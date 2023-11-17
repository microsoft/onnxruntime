# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import os
from itertools import product

import numpy as np
import scipy.special
from ml_dtypes import float8_e4m3fnuz


def dtype_to_bytes(dtype):
    type_map = {
        "float8_e4m3fn": 1,
        "float8_e4m3fnuz": 1,
        "float8_e5m2": 1,
        "float8_e5m2fnuz": 1,
        "float16": 2,
        "float32": 4,
        "float64": 8,
    }
    return type_map[dtype]


def transab_to_suffix(transab):
    return {
        (True, True): "TT",
        (True, False): "TN",
        (False, True): "NT",
        (False, False): "NN",
    }[tuple(transab)]


def dtype_to_suffix(dtype):
    return {
        "float32": "float",
        "float16": "half",
        "float8_e4m3fn": "fp8e4m3fn",
        "float8_e4m3fnuz": "fp8e4m3fnuz",
    }[dtype]


def get_gemm_bound(
    dtype: str,
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    transa: bool,
    transb: bool,
    a_b_positive=False,  # if both a and b are positive matrix, we can skip coeff computation
):
    k = b.shape[1] if transb else b.shape[0]
    # The machine epsilon, unit roundoff, the smallest positive floating point number n such that the floating point
    # number that represents 1 + n is greater than 1.
    machine_eps = 2.0 ** -(24 if dtype == "float32" else 11)

    # The following implements error bound 5.7 in paper I. C. Ipsen and H. Zhou, “Probabilistic error analysis for
    # Inner Products,” SIAM Journal on Matrix Analysis and Applications, vol. 41, no. 4, pp. 1726-1741, 2020.
    # NOTE: the bound is not tight for float16 when k is large
    if a_b_positive:
        coeff = 1.0
    else:
        absa_mul_absb = np.abs(a.T if transa else a) @ np.abs(b.T if transb else b)
        coeff = np.max(absa_mul_absb / np.abs(c))
    gamma_2k = (1.0 + machine_eps) ** (2 * k) - 1.0
    bound_5_7 = coeff * np.sqrt(np.log(2 / 1e-10) * machine_eps * gamma_2k / 2)
    bound = bound_5_7

    return bound


def get_gemm_bert_sizes(full=True):
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


def get_gemm_basic_sizes(full=True):
    if full:
        return list(product([1, 3, 4, 16, 127, 128, 129, 133, 1024], repeat=3))

    # ck has various impls to be tested, use the full basic cases will result too many cases to test.
    # So we use a reduced combination here.
    return list(product([1, 4, 127, 133], [3, 16, 128], [3, 129, 1024]))


def softmax(x, *, is_log_softmax=False, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=1)
    if is_log_softmax:
        return x - np.log(np.sum(np.exp(x), axis=axis, keepdims=1))
    return (np.exp(x)) / np.sum(np.exp(x), axis=axis, keepdims=1)


def _matmul(a, b):
    if os.getenv("KERNEL_EXPLORER_TEST_USE_CUPY", "0") == "1":
        import cupy as cp

        return (cp.asarray(a) @ cp.asarray(b)).get()
    else:
        return a @ b


def matmul(a, b, transa=False, transb=False):
    return _matmul(a.T if transa else a, b.T if transb else b)


def fast_gelu(x, bias):
    x = x + bias
    y = 0.5 * x * (1 + np.tanh(0.797885 * x + 0.035677 * x * x * x))
    return y


def gelu(x, bias):
    x = x + bias
    return 0.5 * x * (1 + scipy.special.erf(x / np.sqrt(2)))


def relu(x, bias):
    x = x + bias
    return np.max(x, 0, keepdims=True)


def root_mean_square(x, axis, epsilon):
    rms = np.sqrt(np.mean(np.square(x), axis=axis, keepdims=True) + epsilon)
    return rms


def standardization(x, axis, epsilon):
    mean = np.mean(x, axis=axis, keepdims=True)
    variance = np.var(x, axis=axis, keepdims=True)
    return (x - mean) / np.sqrt(variance + epsilon)
