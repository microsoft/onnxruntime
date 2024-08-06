# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import functools
import json
import os
import re
import sys
from types import ModuleType
from typing import List, Tuple, Union

import onnx
from onnx import ModelProto
from torch._C import _from_dlpack
from torch.utils.dlpack import to_dlpack

from ._cache import ModuleCache, PyCodeCache
from ._codegen import codegen
from ._op_config import get_supported_ops
from ._sorted_graph import SortedGraph
from ._sympy_utils import extract_shape_from_symbol, parse_shape
from ._utils import gen_unique_name, next_power_of_2

_DEBUG_MODE = "ORTMODULE_TRITON_DEBUG" in os.environ and int(os.getenv("ORTMODULE_TRITON_DEBUG")) == 1

_CUSTOM_KERNELS = dict()


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
    symbolic_shape_hint = None
    min_symbolic_shape = 0
    clear = staticmethod(cache.clear)

    @classmethod
    def set_symbolic_shape_hint(cls, symbolic_shape_hint_config):
        for k, v in symbolic_shape_hint_config.items():
            if k == "*":
                cls.min_symbolic_shape = v
            else:
                if cls.symbolic_shape_hint is None:
                    cls.symbolic_shape_hint = dict()
                cls.symbolic_shape_hint[k] = v

    @classmethod
    def get_shape(cls, onnx_key: int, model: ModelProto, shapes: List[List[int]]) -> List[List[Union[int, str]]]:
        if onnx_key not in cls.cache:
            if cls.symbolic_shape_hint is not None:
                for i, input in enumerate(model.graph.input):
                    if input.type.tensor_type.HasField("shape"):
                        for j, dim in enumerate(input.type.tensor_type.shape.dim):
                            if dim.dim_param:
                                for k, v in cls.symbolic_shape_hint.items():
                                    if re.fullmatch(k, dim.dim_param):
                                        shapes[i][j] = f"i{i}_dim{j}_{v}"
                                        break
            cls.cache[onnx_key] = shapes
        else:
            changed = False
            for i, shape in enumerate(shapes):
                for j, dim in enumerate(shape):
                    if isinstance(cls.cache[onnx_key][i][j], int) and dim != cls.cache[onnx_key][i][j]:
                        max_dim = max(dim, cls.cache[onnx_key][i][j], cls.min_symbolic_shape)
                        shape[j] = f"i{i}_dim{j}_{next_power_of_2(max_dim)}"
                        changed = True
                    elif isinstance(cls.cache[onnx_key][i][j], str):
                        pre = extract_shape_from_symbol(cls.cache[onnx_key][i][j])
                        if pre >= dim:
                            shape[j] = cls.cache[onnx_key][i][j]
                        else:
                            shape[j] = f"i{i}_dim{j}_{next_power_of_2(dim)}"
                            changed = True
            if changed:
                cls.cache[onnx_key] = shapes
        return cls.cache[onnx_key]


def _gen_key(onnx_key: int, model: ModelProto, shapes: List[List[Union[int, str]]]) -> int:
    # pylint: disable=unused-argument
    return hash(f"{onnx_key}|{str(shapes).replace(' ', '')}")


def _gen_module(onnx_key: int, model: ModelProto, shapes: List[List[Union[int, str]]]) -> Tuple[str, ModuleType]:
    sorted_graph = SortedGraph(model, [parse_shape(shape) for shape in shapes])
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
    All supported ops are from user config specified by env ORTMODULE_TRITON_CONFIG_FILE or from _op_config.py.
    The Triton fusion will try to fuse subgraphs with connected supported ops.
    The initializer value can be "none", "scalar", and "all".
        "none": no initializer will be added to subgraphs.
        "scalar": only related scalar initializers will be added to subgraphs.
        "all": all related initializers will be added to subgraphs.
    The min_nodes is used to control the minimum number of non-no-op nodes in a subgraph.
    User can also specify symbolic_shape_hint in the config, which is a dict to control the symbolic shape hint.
    Each entry is a regex pattern to match the dim_param in ONNX model and the value is the power of 2 for the symbolic
    shape. Each dim_param will be replaced by i{input_index}_dim{dim_index}_{power_of_2} in the symbolic shape.
    """

    config = dict()
    config_file = os.getenv("ORTMODULE_TRITON_CONFIG_FILE", "")
    if config_file and os.path.exists(config_file):
        with open(config_file, encoding="UTF-8") as f:
            config = json.load(f)

    if "ops" not in config:
        config["ops"] = get_supported_ops()
    if "initializer" not in config:
        config["initializer"] = "scalar"
    if "min_nodes" not in config:
        config["min_nodes"] = 2

    if "symbolic_shape_hint" in config and len(config["symbolic_shape_hint"]) > 0:
        _ShapeCache.set_symbolic_shape_hint(config["symbolic_shape_hint"])
        del config["symbolic_shape_hint"]

    return json.dumps(config)


def call_triton_by_name(func_name: str, *tensors, **kwargs):
    """
    Call triton kernel by function name. It's expected that there are functions and kernels registered manually
    with that func_name (normally in .kernel sub-module), this function try to get the Python function by name
    and execute it with the given tensors.
    """

    torch_tensors = [_from_dlpack(tensor) if tensor is not None else None for tensor in tensors]
    func = getattr(sys.modules[".".join(__name__.split(".")[:-1])], func_name, None)
    if func is None:
        func = _CUSTOM_KERNELS.get(func_name)
    assert func is not None, f"Function {func_name} is not found in the registered kernels."
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
    concrete_shapes = [list(tensor.size()) for tensor in torch_tensors]
    model = onnx.load_model_from_string(onnx_str)
    shapes = _ShapeCache.get_shape(onnx_key, model, concrete_shapes)
    func_name, mod = ModuleCache.load(_gen_key, _gen_module, onnx_key, model, shapes)
    func = getattr(mod, func_name)
    output = func(*torch_tensors)
    if isinstance(output, tuple):
        return tuple([to_dlpack(tensor) for tensor in output])
    return to_dlpack(output)


def register_triton_kernel(fn):
    _CUSTOM_KERNELS[fn.__name__] = fn
    return fn
