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
        "float16": list(filter(lambda x: "DequantizeInt4_half" in x, dir(ke))),
        "float32": list(filter(lambda x: "DequantizeInt4_float" in x, dir(ke))),
    }
    return type_map[dtype]


dtypes = ["float16", "float32"]


@dataclass
class DequantizeInt4Metric(ke.BandwidthMetric):
    n: int
    k: int

    def report(self):
        return f"{self.duration:6.2f} us {self.gbps:5.2f} GB/s {self.dtype} n={self.n} k={self.k} {self.name}"


def profile_dequantize_int4_func(n, k, dtype, func):
    np.random.seed(0)
    output = np.random.rand(n, k).astype(dtype)
    quant = np.random.randint(low=0, high=127, size=(n, (k + 31) // 32, 16)).astype("uint8")
    scales = np.random.rand(n, (k + 31) // 32).astype(dtype)

    output_d = ke.DeviceArray(output)
    quant_d = ke.DeviceArray(quant)
    scales_d = ke.DeviceArray(scales)
    f = getattr(ke, func)
    my_op = f(output_d, quant_d, scales_d, n, k)
    duration_ms = my_op.Profile()
    total_bytes = (n * k) / 2 + (n * k + n * k / 32) * dtype_to_bytes(dtype)

    ke.report(DequantizeInt4Metric(func, dtype, duration_ms, total_bytes, n, k))


def profile_with_args(n, k, dtype, sort):
    with ke.benchmark(sort):
        for func in dtype_to_funcs(dtype):
            profile_dequantize_int4_func(n, k, dtype, func)


def profile():
    for dt in dtypes:
        for n, k in ((4096, 4096), (4096, 12288), (12288, 4096)):
            profile_with_args(n, k, dt, True)
            print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    group = parser.add_argument_group("profile with args")
    group.add_argument("n", type=int)
    group.add_argument("k", type=int)
    group.add_argument("dtype", choices=dtypes)
    group.add_argument("--sort", action="store_true")

    if len(sys.argv) == 1:
        profile()
    else:
        args = parser.parse_args()
        profile_with_args(args.n, args.k, args.dtype, args.sort)
