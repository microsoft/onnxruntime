# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from itertools import product

import numpy as np


def dtype_to_bytes(dtype):
    type_map = {
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
    }[dtype]


def get_gemm_bound(dtype: str, a: np.ndarray, b: np.ndarray, c: np.ndarray, transa: bool, transb: bool):
    k = b.shape[1] if transb else b.shape[0]
    # The machine epsilon, unit roundoff, the smallest positive floating point number n such that the floating point
    # number that represents 1 + n is greater than 1.
    machine_eps = 2.0 ** -(24 if dtype == "float32" else 11)

    # The following implements error bound 5.7 in paper I. C. Ipsen and H. Zhou, “Probabilistic error analysis for
    # Inner Products,” SIAM Journal on Matrix Analysis and Applications, vol. 41, no. 4, pp. 1726–1741, 2020.
    # NOTE: the bound is not tight for float16 when k is large
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
