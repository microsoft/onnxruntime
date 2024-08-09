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
        "float16": list(filter(lambda x: "DequantizeBnb4_half" in x, dir(ke))),
        "float32": list(filter(lambda x: "DequantizeBnb4_float" in x, dir(ke))),
    }
    return type_map[dtype]


quant_enums = {"FP4": 0, "NF4": 1}


dtypes = ["float16", "float32"]
quant_types = ["FP4", "NF4"]


@dataclass
class DequantizeBnb4Metric(ke.BandwidthMetric):
    quant_type: str
    n: int
    k: int

    def report(self):
        return (
            f"{self.duration:6.2f} us {self.gbps:5.2f} GB/s"
            f" {self.quant_type} {self.dtype} n={self.n} k={self.k} {self.name}"
        )


def profile_dequantize_int4_func(qt, n, k, dtype, func):
    np.random.seed(0)
    block_size = 64
    numel = n * k
    output = np.random.rand(n, k).astype(dtype)
    quant = np.random.randint(low=0, high=255, size=(numel + 1) // 2).astype("uint8")
    absmax = np.random.rand((numel + block_size - 1) // block_size).astype(dtype)
    quant_map_buffer = np.zeros(16).astype(dtype)

    output_d = ke.DeviceArray(output)
    quant_d = ke.DeviceArray(quant)
    absmax_d = ke.DeviceArray(absmax)
    quant_map_buffer_d = ke.DeviceArray(quant_map_buffer)
    f = getattr(ke, func)
    my_op = f(quant_enums[qt], output_d, quant_d, absmax_d, quant_map_buffer_d, n, k)
    duration_ms = my_op.Profile()
    total_bytes = numel / 2 + (numel + numel / block_size) * dtype_to_bytes(dtype)

    ke.report(DequantizeBnb4Metric(func, dtype, duration_ms, total_bytes, qt, n, k))


def profile_with_args(qt, n, k, dtype, sort):
    with ke.benchmark(sort):
        for func in dtype_to_funcs(dtype):
            profile_dequantize_int4_func(qt, n, k, dtype, func)


def profile():
    for qt in quant_types:
        for dt in dtypes:
            for n, k in ((4096, 4096), (4096, 12288), (12288, 4096)):
                profile_with_args(qt, n, k, dt, True)
                print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    group = parser.add_argument_group("profile with args")
    group.add_argument("n", type=int)
    group.add_argument("k", type=int)
    group.add_argument("quant_type", choices=quant_types)
    group.add_argument("dtype", choices=dtypes)
    group.add_argument("--sort", action="store_true")

    if len(sys.argv) == 1:
        profile()
    else:
        args = parser.parse_args()
        profile_with_args(args.quant_type, args.n, args.k, args.dtype, args.sort)
