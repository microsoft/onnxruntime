# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import functools
import json
import os
import sys
from types import ModuleType
from typing import List, Tuple

import onnx
from torch._C import _from_dlpack
from torch.utils.dlpack import to_dlpack

from onnxruntime.training import ortmodule

from ._codecache import PyCodeCache
from ._codegen import codegen
from ._op_config import get_supported_ops
from ._sorted_graph import SortedGraph
from ._sympy_utils import parse_shape
from ._utils import gen_unique_name

_DEBUG_MODE = ortmodule._defined_from_envvar("ORTMODULE_TRITON_DEBUG", 0) != 0


@functools.lru_cache(None)
def _gen_module(sorted_graph: SortedGraph) -> Tuple[str, str, ModuleType]:
    func_name = gen_unique_name("func")
    src_code = codegen(func_name, sorted_graph)
    return func_name, src_code, PyCodeCache().load(src_code)


class ModuleCache:
    """
    Compiled Triton module cache by onnx_key and input tensor shapes.

    """

    cache = dict()
    clear = staticmethod(cache.clear)

    @classmethod
    def load(cls, onnx_key: int, onnx_str: bytes, shapes: List[List[int]]) -> Tuple[str, ModuleType]:
        key = hash(f"{onnx_key}|{str(shapes).replace(' ', '')}") % (10**8)
        if key not in cls.cache:
            model = onnx.load_model_from_string(onnx_str)
            sorted_graph = SortedGraph(model, [parse_shape(shape) for shape in shapes])
            if _DEBUG_MODE:
                os.makedirs(os.path.dirname("triton_debug/"), exist_ok=True)
                sorted_graph.save_onnx(f"triton_debug/{onnx_key}")
            func_name, src_code, mod = _gen_module(sorted_graph)
            if _DEBUG_MODE:
                py_file_path = f"triton_debug/{func_name}_{onnx_key}.py"
                with open(py_file_path, "w") as f:
                    f.write(src_code)
            cls.cache[key] = (func_name, mod)
        return cls.cache[key]


# Get the supported ops and other configs in JSON format to control the Triton fusion on backend side.
# All supported ops are from _op_config.py. The Triton fusion will try to fuse subgraphs with connected supported ops.
# The initializer value can be "none", "scalar", and "all".
#   "none": no initializer will be added to subgraphs.
#   "scalar": only related scalar initializers will be added to subgraphs.
#   "all": all related initializers will be added to subgraphs.
# The min_nodes is used to control the minimum number of non-no-op nodes in a subgraph.
def get_config() -> str:
    config = {"ops": get_supported_ops(), "initializer": "scalar", "min_nodes": 2}
    return json.dumps(config)


# Entry function for TritonOp in Python side. It supports two modes:
# 1. func_name is provided, onnx_key and onnx_str are None: it's expected that there are functions and kernels
#    registered manually with that func_name, try to get the Python function and execute it with the given tensors.
# 2. onnx_key and onnx_str are not None: load the ONNX model from onnx_str, generate the Triton function and kernels,
#    and execute the function with the given tensors. The generated function and kernels are cached for reuse
#    by onnx_key and tensor shapes.
def execute_triton_op(func_name: str, onnx_key: int, onnx_str: bytes, *tensors):
    # TODO: use try-except to print error message for better developer experience for now. Remove it later.
    try:
        torch_tensors = [_from_dlpack(tensor) if tensor is not None else None for tensor in tensors]
        if not onnx_str:
            assert func_name
            func = getattr(sys.modules[".".join(__name__.split(".")[:-1])], func_name)
        else:
            assert all(tensor is not None for tensor in torch_tensors)
            concrete_shapes = [list(tensor.size()) for tensor in torch_tensors]
            func_name, mod = ModuleCache.load(onnx_key, onnx_str, concrete_shapes)
            func = getattr(mod, func_name)
        output = func(*torch_tensors)
    except Exception as e:
        import traceback

        traceback.print_exc()
        raise e
    if isinstance(output, tuple):
        return tuple([to_dlpack(tensor) for tensor in output])
    return to_dlpack(output)
