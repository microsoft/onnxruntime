# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import sys
from dataclasses import dataclass
from itertools import product

import kernel_explorer as ke
import numpy as np
import pytest
from utils import (
    dtype_to_suffix,
    fast_gelu,
    get_gemm_basic_sizes,
    get_gemm_bert_sizes,
    get_gemm_bound,
    matmul,
    transab_to_suffix,
)


# TODO The test method needs update.
def _test_gemmfastgelu(my_func, dtype: str, m: int, n: int, k: int, transa=False, transb=False):
    assert dtype in ["float16", "float32"]

    a_shape = (k, m) if transa else (m, k)
    b_shape = (n, k) if transb else (k, n)

    np.random.seed(0)
    a = (np.random.rand(*a_shape)).astype(dtype).astype("float64")
    b = (np.random.rand(*b_shape)).astype(dtype).astype("float64")
    bias = (np.random.rand(n)).astype(dtype)
    temp_c = matmul(a, b, transa, transb)

    bound = get_gemm_bound(dtype, a, b, temp_c, transa, transb, a_b_positive=True)

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
    my_op = my_func(opa, opb, m, n, k, alpha, dev_a, lda, dev_b, ldb, dev_bias, beta, dev_c, n)

    print(f"dtype={dtype} {transab_to_suffix((transa, transb))} m={m:<5} n={n:<5} k={k:<5} bound: {max(bound, 1e-2)}")

    for impl in my_op.ListOps():
        if not my_op.SelectOp(impl):
            continue

        my_op.Run()
        dev_c.UpdateHostNumpyArray()

        np.testing.assert_allclose(my_c, ref_c, rtol=max(bound, 1e-2))


dtypes = ["float16", "float32"]
all_transabs = list(product([True, False], repeat=2))


@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("size", get_gemm_basic_sizes(full=False) + get_gemm_bert_sizes(full=False))
@pytest.mark.parametrize("transab", all_transabs)
def test_gemmfastgelu_unfused_bert_cases(dtype, size, transab):
    _test_gemmfastgelu(getattr(ke, "GemmFastGeluUnfused_" + dtype_to_suffix(dtype)), dtype, *size, *transab)


@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("size", get_gemm_basic_sizes(full=False) + get_gemm_bert_sizes(full=False))
@pytest.mark.parametrize("transab", all_transabs)
def test_gemmfastgelu_tunable_bert_cases(dtype, size, transab):
    wrapper_name = f"GemmFastGeluTunable_{dtype_to_suffix(dtype)}_{transab_to_suffix(transab)}"
    _test_gemmfastgelu(getattr(ke, wrapper_name), dtype, *size, *transab)


@pytest.mark.skipif(not ke.is_composable_kernel_available(), reason="ck is not enabled")
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("size", get_gemm_basic_sizes(full=False) + get_gemm_bert_sizes(full=False))
@pytest.mark.parametrize("transab", all_transabs)
def test_gemmfastgelu_ck_bert_cases(dtype, size, transab):
    wrapper_name = f"CKGemmFastGelu_{dtype_to_suffix(dtype)}_{transab_to_suffix(transab)}"
    _test_gemmfastgelu(getattr(ke, wrapper_name), dtype, *size, *transab)


@pytest.mark.skipif(not ke.is_hipblaslt_available(), reason="hipblaslt is not available")
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("size", get_gemm_basic_sizes(full=False) + get_gemm_bert_sizes(full=False))
@pytest.mark.parametrize("transab", all_transabs)
def test_gemmfastgelu_hipblaslt_bert_cases(dtype, size, transab):
    _test_gemmfastgelu(getattr(ke, "GemmFastGeluHipBlasLt_" + dtype_to_suffix(dtype)), dtype, *size, *transab)


@dataclass
class GemmFastGeluMetric(ke.ComputeMetric):
    transa: bool
    transb: bool
    m: int
    n: int
    k: int

    def report(self):
        transab = transab_to_suffix((self.transa, self.transb))
        common = f"{self.dtype} m={self.m:<4} n={self.n:<4} k={self.k:<4} {transab}, {self.name}"
        if self.duration <= 0:
            return "not supported          " + common
        return f"{self.duration:>6.2f} us {self.tflops:>5.2f} tflops " + common


def profile_gemmfastgelu_func(my_func, dtype: str, m: int, n: int, k: int, transa: bool, transb: bool):
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
    my_op = my_func(opa, opb, m, n, k, alpha, dev_a, lda, dev_b, ldb, dev_bias, beta, dev_c, n)

    for impl in my_op.ListOps():
        duration_ms = -1
        if my_op.SelectOp(impl):
            duration_ms = my_op.Profile()
        # only counts gemm tflops because fastgelu is low order term (7 * n).
        floating_point_operations = m * k * n * 2

        ke.report(GemmFastGeluMetric(impl, dtype, duration_ms, floating_point_operations, transa, transb, m, n, k))


def profile_with_args(transa, transb, dtype, m, n, k, sort):
    dtype_suffix = "_" + dtype_to_suffix(dtype)
    transab_suffix = "_" + transab_to_suffix((transa, transb))
    with ke.benchmark(sort):
        profile_gemmfastgelu_func(getattr(ke, "GemmFastGeluUnfused" + dtype_suffix), dtype, m, n, k, transa, transb)
        profile_gemmfastgelu_func(
            getattr(ke, "CKGemmFastGelu" + dtype_suffix + transab_suffix), dtype, m, n, k, transa, transb
        )
        profile_gemmfastgelu_func(
            getattr(ke, "GemmFastGeluTunable" + dtype_suffix + transab_suffix), dtype, m, n, k, transa, transb
        )
        if ke.is_hipblaslt_available():
            profile_gemmfastgelu_func(
                getattr(ke, "GemmFastGeluHipBlasLt" + dtype_suffix + transab_suffix), dtype, m, n, k, transa, transb
            )


def profile():
    for dtype in dtypes:
        for m, n, k in get_gemm_bert_sizes(full=True):
            profile_with_args(False, False, dtype, m, n, k, True)
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
    group.add_argument("--sort", action="store_true")
    if len(sys.argv) == 1:
        profile()
    else:
        args = parser.parse_args()
        profile_with_args(args.transa == "T", args.transb == "T", args.dtype, args.m, args.n, args.k, args.sort)
