# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from abc import abstractmethod
from typing import Any, Dict, List, Tuple

import sympy
from onnx import GraphProto, NodeProto, TensorProto

from ._sympy_utils import extract_shape_from_symbol
from ._utils import get_attribute, get_reduce_info, next_power_of_2


class CodegenContext:
    """
    record variable name mapping in term of IRnodes.
    """

    _special_floats = ["inf", "-inf"]

    def __init__(self, var_map: Dict[str, str]):
        self._var_map: Dict[str, str] = {**var_map}

    # Get variable name by the node arg name in ONNX graph.
    def get_variable_name(self, name: str) -> str:
        return self._var_map[name]

    # For some operators such as data load/store, we need an internal variable name inside the kernel function.
    def get_internal_variable_name(self, name: str) -> str:
        var_name = self._var_map[name]
        var_name = self._var_map[var_name] if var_name in self._var_map else var_name
        return f'float("{var_name}")' if var_name in self._special_floats else var_name


class CodeBuffer:
    def __init__(self):
        self.buffer: List[str] = []

    def __iadd__(self, other: str):
        self.buffer.append(other)
        return self

    def __str__(self):
        return "".join(self.buffer)


class NodeVisitor:
    @abstractmethod
    def codegen(self, node: Any, context: CodegenContext, code_buffer: CodeBuffer, indent: int):
        pass


class SymbolicDSU:
    """
    A 'disjoint set union' to merge symbolics so that we use less variables in the generated code.
    When handling shape inference for elementwise Ops, if two symbols are not equal and they are not 1, we merge them.
    """

    def __init__(self):
        self._dsu: Dict[sympy.Expr, sympy.Expr] = {}

    def find(self, symbolic: sympy.Expr) -> sympy.Expr:
        if symbolic not in self._dsu:
            self._dsu[symbolic] = symbolic
            return symbolic
        if symbolic == self._dsu[symbolic]:
            return symbolic
        self._dsu[symbolic] = self.find(self._dsu[symbolic])
        return self._dsu[symbolic]

    def union(self, symbolic: sympy.Expr, other_symbolic: sympy.Expr):
        root = self.find(symbolic)
        other_root = self.find(other_symbolic)
        self._dsu[other_root] = root


class TensorInfo:
    """
    Represent a input/output tensor of a node.
    """

    def __init__(self, dtype: TensorProto.DataType, shape: List[sympy.Expr]):
        self._dtype: TensorProto.DataType = dtype
        self._shape: List[sympy.Expr] = shape

    @property
    def dtype(self) -> TensorProto.DataType:
        return self._dtype

    @property
    def shape(self) -> List[sympy.Expr]:
        return self._shape

    def update_shape(self, symbolics: SymbolicDSU):
        self._shape = [symbolics.find(dim) if dim.is_symbol else dim for dim in self._shape]


def _infer_elementwise_shape(input_infos: List[TensorInfo], symbolics: SymbolicDSU) -> List[sympy.Expr]:
    max_len = max([len(input_info.shape) for input_info in input_infos])
    output_shape: List[sympy.Expr] = [sympy.Integer(1)] * max_len
    for input_info in input_infos:
        offset = max_len - len(input_info.shape)
        for idx, dim in enumerate(input_info.shape):
            if not dim.is_number or dim != 1:
                if not output_shape[idx + offset].is_number or output_shape[idx + offset] != 1:
                    symbolics.union(output_shape[idx + offset], dim)
                else:
                    output_shape[idx + offset] = dim
    return output_shape


def _infer_elementwise(
    node: NodeProto, input_infos: List[TensorInfo], graph: GraphProto, symbolics: SymbolicDSU
) -> List[TensorInfo]:
    # pylint: disable=unused-argument
    return [TensorInfo(input_infos[0].dtype, _infer_elementwise_shape(input_infos, symbolics))]


def _infer_where(
    node: NodeProto, input_infos: List[TensorInfo], graph: GraphProto, symbolics: SymbolicDSU
) -> List[TensorInfo]:
    # pylint: disable=unused-argument
    return [TensorInfo(input_infos[1].dtype, _infer_elementwise_shape(input_infos, symbolics))]


def _infer_reduction(
    node: NodeProto, input_infos: List[TensorInfo], graph: GraphProto, symbolics: SymbolicDSU
) -> List[TensorInfo]:
    # pylint: disable=unused-argument
    input_rank = len(input_infos[0].shape)
    keep_dims, axes = get_reduce_info(node, graph, input_rank)
    axes = [axis + input_rank if axis < 0 else axis for axis in axes]
    axes.sort()
    shape = [input_infos[0].shape[i] for i in range(input_rank) if i not in axes]
    if keep_dims:
        for axis in axes:
            shape.insert(axis, sympy.Integer(1))
    return [TensorInfo(input_infos[0].dtype, shape)]


def _infer_unary(
    node: NodeProto, input_infos: List[TensorInfo], graph: GraphProto, symbolics: SymbolicDSU
) -> List[TensorInfo]:
    # pylint: disable=unused-argument
    return [input_infos[0]]


def _infer_cast(
    node: NodeProto, input_infos: List[TensorInfo], graph: GraphProto, symbolics: SymbolicDSU
) -> List[TensorInfo]:
    # pylint: disable=unused-argument
    dtype = get_attribute(node, "to", TensorProto.UNDEFINED)
    assert dtype != TensorProto.UNDEFINED
    return [TensorInfo(dtype, input_infos[0].shape)]


def _infer_dropout(
    node: NodeProto, input_infos: List[TensorInfo], graph: GraphProto, symbolics: SymbolicDSU
) -> List[TensorInfo]:
    # pylint: disable=unused-argument
    return [input_infos[0], TensorInfo(TensorProto.BOOL, input_infos[0].shape)]


class TypeAndShapeInfer:
    _INFER_FUNC_MAP = {  # noqa: RUF012
        "Add": _infer_elementwise,
        "Sub": _infer_elementwise,
        "Mul": _infer_elementwise,
        "Div": _infer_elementwise,
        "Pow": _infer_elementwise,
        "Sqrt": _infer_elementwise,
        "Exp": _infer_elementwise,
        "Where": _infer_where,
        "Rsqrt": _infer_elementwise,
        "Cast": _infer_cast,
        "Dropout": _infer_dropout,
        "DropoutGrad": _infer_unary,
        "Identity": _infer_unary,
        "ReduceSum": _infer_reduction,
        "ReduceMax": _infer_reduction,
        "ReduceMin": _infer_reduction,
        "Sum": _infer_elementwise,
        "Gelu": _infer_unary,
        "QuickGelu": _infer_unary,
        "GeluGrad": _infer_elementwise,
        "QuickGeluGrad": _infer_elementwise,
    }

    @classmethod
    def infer(
        cls, node: NodeProto, input_infos: List[TensorInfo], graph: GraphProto, symbolics: SymbolicDSU
    ) -> List[TensorInfo]:
        if node.op_type not in cls._INFER_FUNC_MAP:
            raise NotImplementedError(f"Unsupported op type: {node.op_type}")
        return cls._INFER_FUNC_MAP[node.op_type](node, input_infos, graph, symbolics)


class AutotuneConfigs:
    """
    Generate all autotune configs for a kernel function by it's xnumel and rnumel.
    A config is a tuple of (xblock, rblock, num_warps).
    If it's elementwise kernel, the rnumel is 1.
    If it's reduction kernel on last contiguous dimensions, the contiguous flag is True.
    """

    def __init__(self, x_numel: sympy.Expr, r_numel: sympy.Expr, contiguous: bool):
        x_numel_int = (
            int(x_numel)
            if x_numel.is_number
            else int(
                x_numel.subs(
                    {symbol: sympy.Integer(extract_shape_from_symbol(symbol)) for symbol in x_numel.free_symbols}
                )
            )
        )
        r_numel_int = (
            int(r_numel)
            if r_numel.is_number
            else int(
                r_numel.subs(
                    {symbol: sympy.Integer(extract_shape_from_symbol(symbol)) for symbol in r_numel.free_symbols}
                )
            )
        )
        self.configs: List[Tuple[int, int, int]] = self._gen_autotune_configs(x_numel_int, r_numel_int, contiguous)
        # If there is symbolic shape, we will not tune the kernel.
        if not x_numel.is_number or not r_numel.is_number:
            self.configs = self.configs[-1:]
        self.requires_for_loop: bool = any(config[1] < r_numel_int for config in self.configs)

    def _num_warps(self, x: int, r: int) -> int:
        return min(max(x * r // 256, 2), 8)

    def _gen_config(self, xnp2: int, rnp2: int, x: int, r: int) -> Tuple[int, int, int]:
        x = min(x, xnp2)
        r = min(r, rnp2)
        return x, r, self._num_warps(x, r)

    # TODO: we need to tune more kernels to get more reasonable configs for better performance.
    def _gen_autotune_configs(self, x_numel: int, r_numel: int, contiguous: bool) -> List[Tuple[int, int, int]]:
        configs = []
        xnp2 = next_power_of_2(x_numel)
        if r_numel == 1:
            configs.append(self._gen_config(xnp2, 1, 1024, 1))
            if xnp2 > 1024:
                configs.append(self._gen_config(xnp2, 1, 2048, 1))
            return configs
        rnp2 = next_power_of_2(r_numel)
        if contiguous:
            configs.append(self._gen_config(xnp2, rnp2, 1, 2048))
            if rnp2 > 2048:
                configs.append(self._gen_config(xnp2, rnp2, 1, 4096))
            elif rnp2 <= 256:
                x = min(xnp2, 256 // rnp2 * 2)
                configs.append(self._gen_config(xnp2, rnp2, x, rnp2))
        else:
            config_set = {
                self._gen_config(xnp2, rnp2, 1, 2048),
                self._gen_config(xnp2, rnp2, 4, 512),
                self._gen_config(xnp2, rnp2, 8, 512),
                self._gen_config(xnp2, rnp2, 32, 128),
            }
            configs = list(config_set)
        return configs
