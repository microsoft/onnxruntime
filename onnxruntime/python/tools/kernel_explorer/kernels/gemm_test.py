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
from utils import dtype_to_suffix, get_gemm_basic_sizes, get_gemm_bert_sizes, get_gemm_bound, matmul, transab_to_suffix


def _test_gemm(func, dtype: str, transa: bool, transb: bool, m: int, n: int, k: int, alpha=1.0, beta=0.0):
    assert dtype in ["float32", "float16"]

    a_shape = (k, m) if transa else (m, k)
    b_shape = (n, k) if transb else (k, n)

    np.random.seed(0)
    a = (np.random.rand(*a_shape) + 0.5).astype(dtype).astype("float64")
    b = (np.random.rand(*b_shape) + 0.5).astype(dtype).astype("float64")
    intermediate_c = matmul(a, b, transa, transb)
    if alpha == 1.0 and beta == 0.0:  # fast path
        ref_c = intermediate_c
    else:
        ref_c = alpha * intermediate_c + beta * np.ones_like(intermediate_c)

    bound = get_gemm_bound(dtype, a, b, ref_c, transa, transb, a_b_positive=True)

    a = a.astype(dtype)
    b = b.astype(dtype)

    my_c = np.ones((m, n), dtype=dtype)
    dev_a = ke.DeviceArray(a)
    dev_b = ke.DeviceArray(b)
    dev_c = ke.DeviceArray(my_c)

    opa = ke.blas_op.T if transa else ke.blas_op.N
    opb = ke.blas_op.T if transb else ke.blas_op.N
    lda = a_shape[1]
    ldb = b_shape[1]
    my_gemm = func(opa, opb, m, n, k, alpha, dev_a, lda, dev_b, ldb, beta, dev_c, n)

    failures = {}
    print(f"dtype={dtype} {transab_to_suffix((transa, transb))} m={m:<5} n={n:<5} k={k:<5} bound: {bound}")

    for impl in my_gemm.ListOps():
        if not my_gemm.SelectOp(impl):
            continue
        # Restore C Array
        my_c.fill(1.0)
        dev_c.UpdateDeviceArray()
        my_gemm.Run()
        dev_c.UpdateHostNumpyArray()

        try:
            np.testing.assert_allclose(my_c, ref_c, rtol=bound)
        except Exception as err:  # noqa: PERF203
            header = "*" * 30 + impl + "*" * 30
            print(header)
            print(err)
            print("*" * len(header))
            failures[impl] = str(err)

    if failures:
        raise Exception(failures)


dtypes = ["float32", "float16"]
all_transabs = list(product([True, False], repeat=2))


@pytest.mark.parametrize("m, n, k", get_gemm_basic_sizes(full=True) + get_gemm_bert_sizes(full=False))
@pytest.mark.parametrize("transa, transb", all_transabs)
@pytest.mark.parametrize("dtype", dtypes)
def test_rocblas_gemm_all_cases(dtype, transa, transb, m, n, k):
    _test_gemm(getattr(ke, "RocBlasGemm_" + dtype_to_suffix(dtype)), dtype, transa, transb, m, n, k)


@pytest.mark.skipif(not ke.is_composable_kernel_available(), reason="ck is not enabled")
@pytest.mark.parametrize("m, n, k", get_gemm_basic_sizes(full=False) + get_gemm_bert_sizes(full=False))
@pytest.mark.parametrize("transa, transb", all_transabs)
@pytest.mark.parametrize("dtype", dtypes)
def test_ck_gemm_bert_cases(dtype, transa, transb, m, n, k):
    wrapper_name = f"CKGemm_{dtype_to_suffix(dtype)}_{transab_to_suffix((transa, transb))}"
    _test_gemm(getattr(ke, wrapper_name), dtype, transa, transb, m, n, k)


# Tunable is basically wrapped around of rocblas and ck gemm, so no need for full tests
@pytest.mark.parametrize("m, n, k", get_gemm_basic_sizes(full=False) + get_gemm_bert_sizes(full=False))
@pytest.mark.parametrize("transa, transb", all_transabs)
@pytest.mark.parametrize("dtype", dtypes)
def test_gemm_tunable_bert_cases(dtype, transa, transb, m, n, k):
    wrapper_name = f"GemmTunable_{dtype_to_suffix(dtype)}_{transab_to_suffix((transa, transb))}"
    _test_gemm(getattr(ke, wrapper_name), dtype, transa, transb, m, n, k)


@pytest.mark.parametrize("alpha, beta", [(0.5, 0.5)])
@pytest.mark.parametrize("transa, transb", all_transabs)
@pytest.mark.parametrize("dtype", dtypes)
def test_rocblas_gemm_alpha_beta(dtype, transa, transb, alpha, beta):
    wrapper_name = "RocBlasGemm_" + dtype_to_suffix(dtype)
    _test_gemm(getattr(ke, wrapper_name), dtype, transa, transb, 128, 256, 768, alpha=alpha, beta=beta)


@pytest.mark.skipif(not ke.is_composable_kernel_available(), reason="ck is not enabled")
@pytest.mark.parametrize("alpha, beta", [(0.5, 0.5)])
@pytest.mark.parametrize("transa, transb", all_transabs)
@pytest.mark.parametrize("dtype", dtypes)
def test_ck_gemm_alpha_beta(dtype, transa, transb, alpha, beta):
    wrapper_name = f"CKGemm_{dtype_to_suffix(dtype)}_{transab_to_suffix((transa, transb))}"
    _test_gemm(getattr(ke, wrapper_name), dtype, transa, transb, 256, 128, 384, alpha=alpha, beta=beta)


@pytest.mark.parametrize("alpha, beta", [(0.5, 0.5)])
@pytest.mark.parametrize("transa, transb", all_transabs)
@pytest.mark.parametrize("dtype", dtypes)
def test_gemm_tunable_alpha_beta(dtype, transa, transb, alpha, beta):
    wrapper_name = f"GemmTunable_{dtype_to_suffix(dtype)}_{transab_to_suffix((transa, transb))}"
    _test_gemm(getattr(ke, wrapper_name), dtype, transa, transb, 128, 512, 384, alpha=alpha, beta=beta)


@dataclass
class GemmMetric(ke.ComputeMetric):
    transa: bool
    transb: bool
    m: int
    n: int
    k: int

    def report(self):
        prefix = (
            f"{self.name:<50} {self.dtype} {transab_to_suffix((self.transa, self.transb))} "
            f"m={self.m:<4} n={self.n:<4} k={self.k:<4} "
        )
        if self.duration <= 0:
            return prefix + "not supported"

        return prefix + f"{self.duration:>8.4f} us {self.tflops:>5.2f} tflops"


def profile_gemm_func(f, dtype: str, transa: bool, transb: bool, m: int, n: int, k: int):
    a_shape = (k, m) if transa else (m, k)
    b_shape = (n, k) if transb else (k, n)

    np.random.seed(0)
    a = (np.random.rand(*a_shape) * 2 - 1).astype(dtype)
    b = (np.random.rand(*b_shape) * 2 - 1).astype(dtype)
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
    my_gemm = f(opa, opb, m, n, k, alpha, dev_a, lda, dev_b, ldb, beta, dev_c, n)

    for impl in my_gemm.ListOps():
        duration_ms = -1
        if my_gemm.SelectOp(impl):
            duration_ms = my_gemm.Profile()
        FLOPs = m * k * n * 2  # noqa: N806

        ke.report(GemmMetric(impl, dtype, duration_ms, FLOPs, transa, transb, m, n, k))


def profile_with_args(dtype, transa, transb, m, n, k, sort):
    dtype_suffix = "_" + dtype_to_suffix(dtype)
    transab_suffix = "_" + transab_to_suffix((transa, transb))
    with ke.benchmark(sort):
        profile_gemm_func(getattr(ke, "RocBlasGemm" + dtype_suffix), dtype, transa, transb, m, n, k)
        profile_gemm_func(getattr(ke, "CKGemm" + dtype_suffix + transab_suffix), dtype, transa, transb, m, n, k)
        profile_gemm_func(getattr(ke, "GemmTunable" + dtype_suffix + transab_suffix), dtype, transa, transb, m, n, k)
        if ke.is_hipblaslt_available():
            profile_gemm_func(
                getattr(ke, "GemmHipBlasLt" + dtype_suffix + transab_suffix), dtype, transa, transb, m, n, k
            )
    print()


def profile():
    for dtype in dtypes:
        for m, n, k in get_gemm_bert_sizes(full=True):
            profile_with_args(dtype, False, False, m, n, k, True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    group = parser.add_argument_group("profile with args")
    group.add_argument("dtype", choices=dtypes)
    group.add_argument("transa", choices="NT")
    group.add_argument("transb", choices="NT")
    group.add_argument("m", type=int)
    group.add_argument("n", type=int)
    group.add_argument("k", type=int)
    group.add_argument("--sort", action="store_true")

    if len(sys.argv) == 1:
        profile()
    else:
        args = parser.parse_args()
        profile_with_args(args.dtype, args.transa == "T", args.transb == "T", args.m, args.n, args.k, args.sort)
