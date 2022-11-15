# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import numpy as np


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
