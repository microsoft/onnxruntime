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
import sympy
from torch._C import _from_dlpack
from torch.utils.dlpack import to_dlpack

from ._cache import ModuleCache, PyCodeCache
from ._codegen import codegen
from ._op_config import get_supported_ops
from ._sorted_graph import SortedGraph
from ._sympy_utils import extract_shape_from_symbol, parse_shape
from ._utils import gen_unique_name, next_power_of_2

_DEBUG_MODE = "ORTMODULE_TRITON_DEBUG" in os.environ and int(os.getenv("ORTMODULE_TRITON_DEBUG")) == 1


@functools.lru_cache(None)
def _gen_module_internal(sorted_graph: SortedGraph) -> Tuple[str, str, ModuleType]:
    func_name = gen_unique_name("func")
    src_code = codegen(func_name, sorted_graph)
    return func_name, src_code, PyCodeCache().load(src_code)


class _ShapeCache:
    """
    Cache the shapes of the inputs. The inputs are the concrete shapes of inputs from each step for a given ONNX model.
    For those dimensions that the concrete shape is not changed, we use the same concrete shape.
    For those dimensions that the concrete shape is changed between different steps, we use a symbolic shape.
    """

    cache = dict()  # noqa: RUF012
    clear = staticmethod(cache.clear)

    @classmethod
    def get_shape(cls, onnx_key: int, shapes: List[List[sympy.Expr]]) -> List[List[sympy.Expr]]:
        if onnx_key not in cls.cache:
            cls.cache[onnx_key] = shapes
        else:
            changed = False
            for i, shape in enumerate(shapes):
                for j, dim in enumerate(shape):
                    if dim != cls.cache[onnx_key][i][j] and cls.cache[onnx_key][i][j].is_number:
                        max_dim = max(int(dim), int(cls.cache[onnx_key][i][j]))
                        shape[j] = sympy.Symbol(f"i{i}_dim{j}_{next_power_of_2(max_dim)}")
                        changed = True
                    elif cls.cache[onnx_key][i][j].is_symbol:
                        pre = extract_shape_from_symbol(cls.cache[onnx_key][i][j])
                        if pre >= int(dim):
                            shape[j] = cls.cache[onnx_key][i][j]
                        else:
                            shape[j] = sympy.Symbol(f"i{i}_dim{j}_{next_power_of_2(int(dim))}")
                            changed = True
            if changed:
                cls.cache[onnx_key] = shapes
        return cls.cache[onnx_key]


def _gen_key(onnx_key: int, onnx_str: bytes, shapes: List[List[sympy.Expr]]) -> int:
    # pylint: disable=unused-argument
    return hash(f"{onnx_key}|{str(shapes).replace(' ', '')}") % (10**8)


def _gen_module(onnx_key: int, onnx_str: bytes, shapes: List[List[sympy.Expr]]) -> Tuple[str, ModuleType]:
    model = onnx.load_model_from_string(onnx_str)
    sorted_graph = SortedGraph(model, shapes)
    if _DEBUG_MODE:
        os.makedirs(os.path.dirname("triton_debug/"), exist_ok=True)
        sorted_graph.save_onnx(f"triton_debug/{onnx_key}")
    func_name, src_code, mod = _gen_module_internal(sorted_graph)
    if _DEBUG_MODE:
        py_file_path = f"triton_debug/{func_name}_{onnx_key}.py"
        with open(py_file_path, "w", encoding="UTF-8") as f:
            f.write(src_code)
    return func_name, mod


def get_config() -> str:
    """
    Get the supported ops and other configs in JSON format to control the Triton fusion on backend side.
    All supported ops are from _op_config.py. The Triton fusion will try to fuse subgraphs with connected supported ops.
    The initializer value can be "none", "scalar", and "all".
        "none": no initializer will be added to subgraphs.
        "scalar": only related scalar initializers will be added to subgraphs.
        "all": all related initializers will be added to subgraphs.
    The min_nodes is used to control the minimum number of non-no-op nodes in a subgraph.
    """

    config = {"ops": get_supported_ops(), "initializer": "scalar", "min_nodes": 2}
    return json.dumps(config)


def call_triton_by_name(func_name: str, *tensors, **kwargs):
    """
    Call triton kernel by function name. It's expected that there are functions and kernels registered manually
    with that func_name (normally in .kernel sub-module), this function try to get the Python function by name
    and execute it with the given tensors.
    """

    torch_tensors = [_from_dlpack(tensor) if tensor is not None else None for tensor in tensors]
    func = getattr(sys.modules[".".join(__name__.split(".")[:-1])], func_name)
    output = func(*torch_tensors, **kwargs)
    if output is not None:
        if isinstance(output, tuple):
            return tuple([to_dlpack(tensor) for tensor in output])
        return to_dlpack(output)
    return None


def call_triton_by_onnx(onnx_key: int, onnx_str: bytes, *tensors):
    """
    Call triton kernel by ONNX model. Load the ONNX model from onnx_str, generate the Triton function and kernels,
    and execute the function with the given tensors.
    """

    assert all(tensor is not None for tensor in tensors)
    torch_tensors = [_from_dlpack(tensor) for tensor in tensors]
    concrete_shapes = [parse_shape(list(tensor.size())) for tensor in torch_tensors]
    shapes = _ShapeCache.get_shape(onnx_key, concrete_shapes)
    func_name, mod = ModuleCache.load(_gen_key, _gen_module, onnx_key, onnx_str, shapes)
    func = getattr(mod, func_name)
    output = func(*torch_tensors)
    if isinstance(output, tuple):
        return tuple([to_dlpack(tensor) for tensor in output])
    return to_dlpack(output)
