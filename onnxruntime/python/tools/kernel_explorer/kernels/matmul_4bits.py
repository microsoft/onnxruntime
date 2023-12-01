# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import sys
from dataclasses import dataclass

import kernel_explorer as ke
import numpy as np
from utils import dtype_to_bytes


def dtype_to_funcs(dtype):
    type_map = {
        "float16": list(filter(lambda x: "MatrixFloatInt4_half" in x, dir(ke))),
        "float32": list(filter(lambda x: "MatrixFloatInt4_float" in x, dir(ke))),
    }
    return type_map[dtype]


def dtype_to_funcs_cublas(dtype):
    type_map = {
        "float16": list(filter(lambda x: "GemmBenchmark_half" in x, dir(ke))),
        "float32": list(filter(lambda x: "GemmBenchmark_float" in x, dir(ke))),
    }
    return type_map[dtype]


dtypes = ["float16", "float32"]


@dataclass
class MatrixMulMetric(ke.BandwidthMetric):
    m: int
    n: int
    k: int

    def report(self):
        return (
            f"{self.duration:6.2f} us {self.gbps:5.2f} GB/s {self.dtype} m={self.m} n={self.n} k={self.k} {self.name}"
        )


@dataclass
class MatrixFpInt4Metric(MatrixMulMetric):
    is_symmetric: bool

    def report(self):
        return f"{self.duration:6.2f} us {self.gbps:5.2f} GB/s {self.dtype} m={self.m} n={self.n} k={self.k} is_symmetric={self.is_symmetric} {self.name}"


@ke.dispatchable(pattern_arg=4)
def profile_matmul_fp_int4_func(m, n, k, dtype, func, is_symmetric):
    np.random.seed(0)
    output = np.random.rand(m, n).astype(dtype)
    a = np.random.rand(m, k).astype(dtype)
    b = np.random.randint(low=0, high=127, size=(n, (k + 31) // 32, 16)).astype("uint8")
    scales = np.random.rand(n * ((k + 31) // 32)).astype(dtype)
    zeropoints = np.random.rand(n * (((k + 31) // 32 + 1) // 2)).astype(dtype)

    output_d = ke.DeviceArray(output)
    a_d = ke.DeviceArray(a)
    b_d = ke.DeviceArray(b)
    scales_d = ke.DeviceArray(scales)
    zeropoints_d = ke.DeviceArray(zeropoints)
    f = getattr(ke, func)

    my_op = (
        f(output_d, a_d, b_d, scales_d, m, n, k)
        if is_symmetric
        else f(output_d, a_d, b_d, scales_d, zeropoints_d, m, n, k)
    )
    duration_ms = my_op.Profile()
    total_bytes = (m * k + n * k + m * n) * (dtype_to_bytes(dtype))

    ke.report(MatrixFpInt4Metric(func, dtype, duration_ms, total_bytes, m, n, k, is_symmetric))


@ke.dispatchable(pattern_arg=4)
def profile_gemm_func(m, n, k, dtype, func):
    np.random.seed(0)
    output = np.random.rand(m, n).astype(dtype)
    a = np.random.rand(m, k).astype(dtype)
    b = np.random.rand(k, n).astype(dtype)

    output_d = ke.DeviceArray(output)
    a_d = ke.DeviceArray(a)
    b_d = ke.DeviceArray(b)
    f = getattr(ke, func)
    my_op = f(output_d, a_d, b_d, m, n, k)
    duration_ms = my_op.Profile()
    total_bytes = (m * k + n * k + m * n) * (dtype_to_bytes(dtype))

    ke.report(MatrixMulMetric(func, dtype, duration_ms, total_bytes, m, n, k))


@ke.dispatchable
def profile_with_args(m, n, k, dtype):
    with ke.benchmark():
        for func in dtype_to_funcs(dtype):
            profile_matmul_fp_int4_func(m, n, k, dtype, func, True)

        for func in dtype_to_funcs(dtype):
            profile_matmul_fp_int4_func(m, n, k, dtype, func, False)

        for func in dtype_to_funcs_cublas(dtype):
            profile_gemm_func(m, n, k, dtype, func)


def profile():
    dims_m = [1]
    for dt in dtypes:
        for m in dims_m:
            for n, k in (
                (4096, 4096),
                (4096, 12288),
                (12288, 4096),
                (4096, 11008),
                (11008, 4096),
                (2 * 11008, 4096),
            ):
                profile_with_args(m, n, k, dt)
                print()


if __name__ == "__main__":
    parser = ke.get_argument_parser()
    group = parser.add_argument_group()
    group.add_argument("m", type=int)
    group.add_argument("n", type=int)
    group.add_argument("k", type=int)
    group.add_argument("dtype", choices=dtypes)

    if not ke.has_args():
        profile()
    else:
        args = parser.parse_args()
        args.dispatch(args.m, args.n, args.k, args.dtype)
