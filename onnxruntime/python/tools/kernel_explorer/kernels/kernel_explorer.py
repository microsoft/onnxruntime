# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""This file provides wrapper for native _kernel_explorer.so library and benchmark reporter for operator"""

from __future__ import annotations
from typing import Callable

import ctypes
import os
import json
import sys
from abc import abstractmethod
from argparse import ArgumentParser, Action
from contextlib import contextmanager
from dataclasses import dataclass
from fnmatch import fnmatch
from functools import wraps

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
_is_cuda_available = False
_is_rocm_available = False
if "CUDAExecutionProvider" in available_providers:
    library_files_to_load.append("libonnxruntime_providers_cuda.so")
    _is_cuda_available = True
if "ROCMExecutionProvider" in available_providers:
    library_files_to_load.append("libonnxruntime_providers_rocm.so")
    _is_rocm_available = True

library_to_load = []

for lib in library_files_to_load:
    for prefix in search_paths:
        path = os.path.join(prefix, lib)
        if os.path.exists(path):
            library_to_load.append(path)
            continue

        raise OSError(f"cannot found {lib}")


# use RTLD_GLOBAL to bring all symbols to global name space
_libraries = [ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL) for lib_path in library_to_load]
del library_files_to_load, library_to_load

# pylint: disable=wrong-import-position, disable=unused-import
import _kernel_explorer  # noqa: E402, F401

# pylint: disable=wrong-import-position, disable=unused-import, disable=wildcard-import
from _kernel_explorer import *  # noqa: F403, E402


@dataclass
class _KeContext:
    sort: bool = False

    pattern = "*"

    # mapping the module to dispatch to
    dispatchable = {}
    instance_dispatchable = {}  # can be filter with pattern

    dispatch_depth = 0

    save_tuning_results: str | None = None
    return_tuning_results: bool = False


_ke_context = _KeContext()


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


@dataclass
class ComputeAndBandwidthMetric(ComputeMetric, BandwidthMetric):
    pass


class InstanceBenchmarkReporter:
    def __init__(self):
        self.best = float("inf")
        self.reporters = []

    def make_report(self):
        self.reporters.sort()
        for item in self.reporters:
            if not _ke_context.sort and item.milliseconds_duration > 0 and item.milliseconds_duration < self.best:
                self.best = item.milliseconds_duration
                print(item.report(), "*")
            else:
                print(item.report())
        self.reporters.clear()

    def receive(self, status):
        self.reporters.append(status)
        if not _ke_context.sort:
            self.make_report()

    def _reset_best(self):
        self.best = float("inf")


_reporter = InstanceBenchmarkReporter()


@contextmanager
def benchmark():
    _reporter._reset_best()
    try:
        yield
    finally:
        _reporter.make_report()


def report(status):
    _reporter.receive(status)


def set_ort_severity(v):
    v = int(v)
    onnxruntime_pybind11_state.set_default_logger_severity(v)
    return v


def set_ort_verbosity(v):
    v = int(v)
    onnxruntime_pybind11_state.set_default_logger_verbosity(v)
    return v


def register_common_arguments(parser: ArgumentParser):
    class SortAction(Action):
        def __init__(self, option_strings, dest, default=False, help=None):
            super().__init__(option_strings=option_strings, dest=dest, nargs=0, default=default, help=help)

        def __call__(self, parser, namespace, values, option_string=None):
            setattr(namespace, self.dest, True)
            _ke_context.sort = True

    def set_dispatch(name):
        if name in _ke_context.dispatchable:
            dispatch = _ke_context.dispatchable[name]
            _ke_context.dispatch = dispatch
            return dispatch

        if name in _ke_context.instance_dispatchable:
            msg = f"'{name}' needs an instance to dispatch, thus it is not dispatchable from commandline."
            print(msg)
            raise ValueError(msg)

        from difflib import SequenceMatcher as Matcher

        valid_names = list(_ke_context.dispatchable.keys())
        scored_names = list(reversed(sorted([(Matcher(None, name, a).ratio(), a) for a in valid_names])))
        top10 = "\n    ".join([a for _, a in scored_names[:10]])
        msg = f"'{name}' is not registered for dispatch. Top 10 matches are:\n    {top10}"
        print(msg)
        raise ValueError(msg)

    def set_pattern(pattern):
        pattern = str(pattern)
        _ke_context.pattern = pattern

    def set_save_tuning_results(path):
        _ke_context.save_tuning_results = path
        return path

    group = parser.add_argument_group("kernel explorer args", "Common arguments for kernel explorer")
    group.add_argument(
        "--sort",
        action=SortAction,
        help="control the sort of ke benchmark results based on timing",
    )
    group.add_argument(
        "--ort_default_logger_severity",
        default=2,
        choices=[0, 1, 2, 3, 4],
        type=set_ort_severity,
        help="0:Verbose, 1:Info, 2:Warning, 3:Error, 4:Fatal",
    )
    group.add_argument("--ort_default_logger_verbosity", default=0, type=set_ort_verbosity)
    group.add_argument(
        "--dispatch",
        default="profile_with_args",
        help="dispatch a registered dispatchable.",
        type=set_dispatch,
    )
    group.add_argument(
        "--pattern",
        default="*",
        help="filter the register instanced dispatchables, only matched pattern will be ran.",
        type=set_pattern,
    )
    group.add_argument(
        "--save_tuning_results",
        default=None,
        type=set_save_tuning_results,
        help="patch the dispatch function to save tuning results to the specified path.",
    )

    return parser


def get_argument_parser():
    parser = ArgumentParser()
    return register_common_arguments(parser)


def has_args():
    if "--help" in sys.argv or "-h" in sys.argv or "--func" in sys.argv:
        return True

    # parse the KE args group
    parser = get_argument_parser()
    _, remainder = parser.parse_known_args(sys.argv)
    return len(remainder) > 1  # the file path is always the remainder


def is_cuda_available():
    return _is_cuda_available


def is_rocm_available():
    return _is_rocm_available


def dispatchable(f: Callable | None = None, *, pattern_arg: int | None = None):
    def wrap_dispatch(f, *args, **kwargs):
        _ke_context.dispatch_depth += 1
        ret = f(*args, **kwargs)
        _ke_context.dispatch_depth -= 1
        if _ke_context.dispatch_depth == 0:
            if _ke_context.save_tuning_results is not None:
                try:
                    trs = _kernel_explorer.get_collected_tuning_results()
                    json.dump(trs, open(_ke_context.save_tuning_results, "x"))
                finally:
                    pass

            if _ke_context.return_tuning_results:
                if ret is not None:
                    print(
                        f"WARNING: kernel explorer wants to override the return value of {f.__name__},",
                        "but original return value is not None!",
                    )
                    return ret
                try:
                    trs = _kernel_explorer.get_collected_tuning_results()
                finally:
                    return trs

        return ret

    if f is None:  # Used with ke.dispatchable(...)
        assert pattern_arg is not None

        def decorator(f):
            _ke_context.instance_dispatchable[f.__name__] = f

            @wraps(f)
            def wrapper(*args, **kwargs):
                func_name = args[pattern_arg] if isinstance(args[pattern_arg], str) else args[pattern_arg].__name__
                if not fnmatch(func_name, _ke_context.pattern):
                    print(
                        f"Trying to run {func_name},",
                        f"does not match allowed function name pattern '{_ke_context.pattern}', skip...",
                    )
                    return
                return wrap_dispatch(f, *args, **kwargs)

            return wrapper

        return decorator

    else:  # Used with @ke.dispatchable
        _ke_context.dispatchable[f.__name__] = f

        @wraps(f)
        def wrapper(*args, **kwargs):
            return wrap_dispatch(f, *args, **kwargs)

        return wrapper


def set_dispatchable_pattern(p: str = "*"):
    _ke_context.pattern = p


def set_return_tuning_results(b: bool = True):
    _ke_context.return_tuning_results = b
