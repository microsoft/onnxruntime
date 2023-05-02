# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""This file provides wrapper for native _kernel_explorer.so library and benchmark reporter for operator"""

import ctypes
import os
import sys
from abc import abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass

build_dir = os.environ.get("KERNEL_EXPLORER_BUILD_DIR", None)
if build_dir is None:
    raise ValueError("Environment variable KERNEL_EXPLORER_BUILD_DIR is required")

if not os.path.exists(build_dir):
    raise ValueError(f"KERNEL_EXPLORER_BUILD_DIR ({build_dir}) points to nonexistent path")

# onnxruntime_pybind11_state and kernel_explorer
sys.path.insert(0, build_dir)

# pylint: disable=wrong-import-position
import onnxruntime_pybind11_state  # noqa: E402

# We need to call some functions to properly initialize so pointers in the library
available_providers = onnxruntime_pybind11_state.get_available_providers()


build_dir = os.path.realpath(build_dir)
search_paths = [build_dir]

# As Kernel Explorer makes use of utility functions in ONNXRuntime, we dlopen all relevant libraries to bring required
# symbols into global namespace, so that we don't need to worry about linking.
library_files_to_load = [
    "onnxruntime_pybind11_state.so",
    "libonnxruntime_providers_shared.so",
]
if "CUDAExecutionProvider" in available_providers:
    library_files_to_load.append("libonnxruntime_providers_cuda.so")
if "ROCMExecutionProvider" in available_providers:
    library_files_to_load.append("libonnxruntime_providers_rocm.so")

library_to_load = []

for lib in library_files_to_load:
    for prefix in search_paths:
        path = os.path.join(prefix, lib)
        if os.path.exists(path):
            library_to_load.append(path)
            continue

        raise OSError(f"cannot found {lib}")


# use RTLD_GLOBAL to bring all symbols to global name space
libraries = [ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL) for lib_path in library_to_load]

# pylint: disable=wrong-import-position, disable=unused-import
import _kernel_explorer  # noqa: E402, F401

# pylint: disable=wrong-import-position, disable=unused-import, disable=wildcard-import
from _kernel_explorer import *  # noqa: F403, E402


# Benchmark Reporter
@dataclass
class MetricBase:
    name: str
    dtype: str
    milliseconds_duration: float

    def __lt__(self, other):
        if "Tunable" in self.name or other.duration < 0:
            return True
        if "Tunable" in other.name or self.duration < 0:
            return False

        return self.duration < other.duration

    @property
    def duration(self):
        return self.milliseconds_duration * 1000

    @abstractmethod
    def report(self) -> str:
        raise NotImplementedError()


@dataclass
class ComputeMetric(MetricBase):
    FLOPs: int

    @property
    def tflops(self):
        return self.FLOPs * 1e6 / self.duration / 1e12


@dataclass
class BandwidthMetric(MetricBase):
    bytes: int

    @property
    def gbps(self):
        return self.bytes * 1e6 / self.duration / 1e9


class InstanceBenchmarkReporter:
    def __init__(self):
        self.sort = False
        self.reporters = []

    def set_sort(self, sort):
        self.sort = sort

    def make_report(self):
        self.reporters.sort()
        for item in self.reporters:
            print(item.report())
        self.reporters.clear()

    def receive(self, status):
        self.reporters.append(status)
        if not self.sort:
            self.make_report()


_reporter = InstanceBenchmarkReporter()


@contextmanager
def benchmark(sort):
    _reporter.set_sort(sort)
    try:
        yield
    finally:
        _reporter.make_report()


def report(status):
    _reporter.receive(status)
