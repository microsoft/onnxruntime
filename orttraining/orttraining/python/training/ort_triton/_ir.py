# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from abc import abstractmethod
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import sympy

from ._common import AutotuneConfigs, CodeBuffer, CodegenContext, NodeVisitor, TensorInfo
from ._sympy_utils import parse_shape
from ._utils import gen_unique_name, gen_variable_name, sort_reduce_axes, to_numpy_type


class TensorArg:
    """
    A TensorArg represents a tensor argument in the kernel function.
    It contains a name (from ONNX graph), the data type, the shape.
    If it's constant (initializer or constant node), it also contains the data in numpy array.
    """

    def __init__(self, name: str, tensor_info: Optional[TensorInfo] = None, data: Optional[np.ndarray] = None):
        self._name: str = name
        self._data: Optional[np.ndarray] = data
        if data is not None:
            self._dtype: np.dtype = data.dtype
            self._shape: List[sympy.Expr] = parse_shape(list(data.shape))
        else:
            assert tensor_info is not None
            self._dtype: np.dtype = to_numpy_type(tensor_info.dtype)
            self._shape: List[sympy.Expr] = tensor_info.shape
        self.cross_kernels: bool = False

    @property
    def name(self) -> str:
        return self._name

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    @property
    def shape(self) -> List[sympy.Expr]:
        return self._shape

    @property
    def data(self) -> Optional[np.ndarray]:
        return self._data


class OffsetCalculator:
    """
    OffsetCalculator maps tensor arguments to the target shape of a kernel.
    It' used to generate the offset code for data load/store for a tensor argument in a kernel with
    specific target shape.
    If the reduce_axes is not empty, it means the kernel is a reduction kernel, otherwise it's an element-wise kernel.
    It requires the axes in reduce_axes are contiguous.
    If a reduce node has non-contiguous axes, need to decompose it into multiple reduce nodes before code-gen.
    """

    def __init__(self, target_shape: List[sympy.Expr], reduce_axes: List[int]):
        self.target_shape: List[sympy.Expr] = target_shape
        self.is_reduction: bool = len(reduce_axes) > 0
        self.rank = len(target_shape)
        self.reduce_axes = sort_reduce_axes(reduce_axes, self.rank)
        self.x_dims: List[sympy.Expr] = [target_shape[dim] for dim in range(self.rank) if dim not in self.reduce_axes]
        self.x_rank: int = len(self.x_dims)
        self.x_numel: sympy.Expr = sympy.prod(self.x_dims) if self.x_rank > 0 else sympy.Integer(1)
        self.r_dims: List[sympy.Expr] = [target_shape[dim] for dim in self.reduce_axes]
        self.r_rank: int = len(self.r_dims)
        self.r_numel: sympy.Expr = sympy.prod(self.r_dims) if self.r_rank > 0 else sympy.Integer(1)
        self.x_strides: List[sympy.Expr] = []
        if self.x_rank > 0:
            self.x_strides.append(sympy.Integer(1))
            for i in range(self.x_rank - 2, -1, -1):
                self.x_strides.insert(0, self.x_strides[0] * self.x_dims[i + 1])
        # To avoid generating useless code for offset calculation, we use x_compute_dims and r_compute_dims to
        # track the dimensions that need to be computed in the offset calculation. These 2 sets will be set in
        # register_tensor_arg function below.
        self.x_compute_dims: Set[int] = set()
        self.r_strides: List[sympy.Expr] = []
        if self.r_rank > 0:
            self.r_strides.append(sympy.Integer(1))
            for i in range(self.r_rank - 2, -1, -1):
                self.r_strides.insert(0, self.r_strides[0] * self.r_dims[i + 1])
        self.r_compute_dims: Set[int] = set()
        self.input_strides: Dict[str, List[sympy.Expr]] = dict()
        self.autotune_configs: AutotuneConfigs = AutotuneConfigs(
            self.x_numel, self.r_numel, not self.is_reduction or self.reduce_axes[-1] == self.rank - 1
        )
        simplified_x_numel = self.x_numel.subs({symbol: sympy.Integer(1) for symbol in self.x_numel.free_symbols})
        self.requires_x_mask: bool = any(
            simplified_x_numel % sympy.Integer(config[0]) != 0 for config in self.autotune_configs.configs
        )
        simplified_r_numel = self.r_numel.subs({symbol: sympy.Integer(1) for symbol in self.r_numel.free_symbols})
        self.requires_r_mask: bool = any(
            simplified_r_numel % sympy.Integer(config[1]) != 0 for config in self.autotune_configs.configs
        )
        self.reduced_args: Set[str] = set()
        self.symbolic_shape_variables: Set[str] = set()

    def get_input_strides(self, name: str) -> List[sympy.Expr]:
        assert name in self.input_strides
        return self.input_strides[name]

    def get_x_input_strides(self, name: str) -> List[sympy.Expr]:
        return [dim for idx, dim in enumerate(self.get_input_strides(name)) if idx not in self.reduce_axes]

    def get_r_input_strides(self, name: str) -> List[sympy.Expr]:
        return [dim for idx, dim in enumerate(self.get_input_strides(name)) if idx in self.reduce_axes]

    # Whether the x shape of the tensor argument is contiguous and is same as the target shape.
    def is_same_x_shape(self, name: str) -> bool:
        if (
            self.is_reduction
            and self.reduce_axes[0] != 0
            and self.reduce_axes[-1] != self.rank - 1
            and not self.is_r_reduced(name)
        ):
            return False
        return all(dim != sympy.Integer(0) for dim in self.get_x_input_strides(name))

    # Whether the r shape of the tensor argument is same as the target shape (r shape dimensions are always contiguous).
    def is_same_r_shape(self, name: str) -> bool:
        return all(dim != sympy.Integer(0) for dim in self.get_r_input_strides(name))

    def register_tensor_arg(self, tensor_arg: TensorArg):
        if tensor_arg.name in self.input_strides:
            return
        strides = []
        input_shape = tensor_arg.shape
        if tensor_arg.name in self.reduced_args:
            assert self.is_reduction
            reduced_rank = len(self.target_shape) - len(self.reduce_axes)
            if len(input_shape) < reduced_rank:
                input_shape = [sympy.Integer(1)] * (reduced_rank - len(input_shape)) + input_shape
            input_shape = (
                input_shape[: self.reduce_axes[0]]
                + ([sympy.Integer(1)] * len(self.reduce_axes))
                + input_shape[self.reduce_axes[0] :]
            )
        elif len(input_shape) < len(self.target_shape):
            input_shape = [sympy.Integer(1)] * (len(self.target_shape) - len(input_shape)) + input_shape
        running_stride = sympy.Integer(1)
        for i in range(len(self.target_shape) - 1, -1, -1):
            if self.target_shape[i] == input_shape[i] and not (
                tensor_arg.name in self.reduced_args and i in self.reduce_axes
            ):
                strides.insert(0, running_stride)
                running_stride = running_stride * input_shape[i]
            else:
                strides.insert(0, sympy.Integer(0))
        self.input_strides[tensor_arg.name] = strides
        x_input_strides = self.get_x_input_strides(tensor_arg.name)
        if not self.is_same_x_shape(tensor_arg.name):
            for idx, dim in enumerate(x_input_strides):
                if dim != sympy.Integer(0):
                    self.x_compute_dims.add(idx)
                    if idx != self.x_rank - 1:
                        self.symbolic_shape_variables.update(
                            [symbol.name for symbol in self.x_strides[idx].free_symbols]
                        )
                    if idx != 0:
                        self.symbolic_shape_variables.update([symbol.name for symbol in self.x_dims[idx].free_symbols])
        elif len(x_input_strides) > 0 and x_input_strides[-1] != sympy.Integer(1):
            self.symbolic_shape_variables.update([symbol.name for symbol in x_input_strides[-1].free_symbols])
        r_input_strides = self.get_r_input_strides(tensor_arg.name)
        if not self.is_same_r_shape(tensor_arg.name):
            for idx, dim in enumerate(r_input_strides):
                if dim != sympy.Integer(0):
                    self.r_compute_dims.add(idx)
                    if idx != self.r_rank - 1:
                        self.symbolic_shape_variables.update(
                            [symbol.name for symbol in self.r_strides[idx].free_symbols]
                        )
                    if idx != 0:
                        self.symbolic_shape_variables.update([symbol.name for symbol in self.r_dims[idx].free_symbols])
        elif len(r_input_strides) > 0 and r_input_strides[-1] != sympy.Integer(1):
            self.symbolic_shape_variables.update([symbol.name for symbol in r_input_strides[-1].free_symbols])

    def is_x_reduced(self, name: str) -> bool:
        strides = self.get_input_strides(name)
        return all(dim == sympy.Integer(0) for idx, dim in enumerate(strides) if idx not in self.reduce_axes)

    def is_r_reduced(self, name: str) -> bool:
        strides = self.get_input_strides(name)
        return all(dim == sympy.Integer(0) for idx, dim in enumerate(strides) if idx in self.reduce_axes)


class IRNode:
    """
    The base class for all IR nodes.
    """

    def __init__(self, inputs: List[TensorArg], outputs: List[TensorArg]):
        self.inputs: List[TensorArg] = inputs
        self.outputs: List[TensorArg] = outputs

    @abstractmethod
    def codegen(self, visitor: NodeVisitor, context: CodegenContext, code_buffer: CodeBuffer, indent: int = 0):
        visitor.codegen(self, context, code_buffer, indent)


class ComputeNode(IRNode):
    """
    Each operator is represented as a ComputeNode.
    """

    def __init__(
        self,
        op_type: str,
        inputs: List[TensorArg],
        outputs: List[TensorArg],
        attributes: Dict[str, Any] = {},  # noqa: B006
    ):
        super().__init__(inputs, outputs)
        self._op_type: str = op_type
        self._attributes: Dict[str, Any] = attributes

    @property
    def op_type(self):
        return self._op_type

    @property
    def attributes(self):
        return self._attributes


class ReduceNode(ComputeNode):
    def __init__(self, op_type: str, inputs: List[TensorArg], outputs: List[TensorArg], offset_calc: OffsetCalculator):
        super().__init__(op_type, inputs, outputs)
        assert op_type == "ReduceSum" or op_type == "ReduceMax" or op_type == "ReduceMin"
        self.default_value: str = (
            "0.0" if op_type == "ReduceSum" else ('float("-inf")' if op_type == "ReduceMax" else 'float("inf")')
        )
        self.triton_func: str = (
            "tl.sum" if op_type == "ReduceSum" else ("tl.max" if op_type == "ReduceMax" else "tl.min")
        )
        self.offset_calc: OffsetCalculator = offset_calc


class ReduceForLoopStart(ComputeNode):
    """
    For reduce kernels that need for loop to compute, ReduceForLoopStart and ReduceForLoopEnd are used to
    represent the start and end of the for loop.

    shared-memory declaration
    """

    def __init__(self, reduce_nodes: List[ReduceNode], offset_calc: OffsetCalculator):
        super().__init__("", [], [])
        self.reduce_nodes: List[ReduceNode] = reduce_nodes
        self.offset_calc: OffsetCalculator = offset_calc


class ReduceForLoopEnd(ComputeNode):
    """
    shared-memory reduction
    """

    def __init__(self, reduce_nodes: List[ReduceNode], offset_calc: OffsetCalculator):
        super().__init__("", [], [])
        self.reduce_nodes: List[ReduceNode] = reduce_nodes
        self.offset_calc: OffsetCalculator = offset_calc


class DropoutNode(ComputeNode):
    """
    DropoutNode is used to represent the dropout operator. It is special because we need to track the global offset
    if there are more than one dropout operators in the subgraph.
    """

    def __init__(self, inputs: List[TensorArg], outputs: List[TensorArg], offset_calc: OffsetCalculator):
        super().__init__("Dropout", inputs, outputs)
        self.offset_calc: OffsetCalculator = offset_calc
        self.offset_calc.register_tensor_arg(inputs[0])
        self.global_offset: sympy.Expr = sympy.Integer(0)


class IONode(IRNode):
    """
    The IONode is used to represent the input and output of the subgraph,
    which is used to generate data load/store code.

    """

    def __init__(self, tensor_arg: TensorArg, offset_calc: OffsetCalculator, is_load: bool):
        super().__init__([], [])
        self.tensor_arg: TensorArg = tensor_arg
        self.is_load: bool = is_load
        self.offset_calc: OffsetCalculator = offset_calc
        self.offset_calc.register_tensor_arg(tensor_arg)


class KernelNode(IRNode):
    """
    The KernelNode is used to represent a single kernel. Each kernel has a unique target shape.

    """

    def __init__(self, inputs: List[TensorArg], outputs: List[TensorArg], target_shape: List, reduce_axes: List[int]):
        super().__init__(inputs, outputs)
        self.name: str = gen_unique_name("triton")
        self.internal_args: Set[str] = set()
        self.constants: Dict[str, TensorArg] = dict()
        self.target_shape: List[sympy.Expr] = target_shape
        self.sub_nodes: List[IRNode] = []
        self.var_map: Dict[str, str] = dict()
        self.has_dropout: bool = False
        self.offset_calc: OffsetCalculator = OffsetCalculator(target_shape, reduce_axes)

    def gen_variable_names(self):
        existing_names = set()
        for input in self.inputs:
            name = gen_variable_name(input.name, "in", existing_names)
            self.var_map[input.name] = name
            self.var_map[name] = "t_" + name
        for output in self.outputs:
            name = gen_variable_name(output.name, "out", existing_names)
            self.var_map[output.name] = name
            self.var_map[name] = "t_" + name
        for name in self.internal_args:
            self.var_map[name] = gen_variable_name(name, "t", existing_names)
        for name, tensor_arg in self.constants.items():
            self.var_map[name] = gen_variable_name(name, "c", existing_names)
            if tensor_arg.data is not None:
                value = tensor_arg.data
                if value is not None:
                    assert value.size == 1, f"unsupported constant array {value}"
                    variable_name = self.var_map[name]
                    assert variable_name not in self.var_map
                    self.var_map[variable_name] = str(np.array(value.item(), value.dtype))


class ElementwiseKernelNode(KernelNode):
    def __init__(self, inputs: List[TensorArg], outputs: List[TensorArg], target_shape: List[sympy.Expr]):
        super().__init__(inputs, outputs, target_shape, [])


class ReduceKernelNode(KernelNode):
    def __init__(
        self,
        inputs: List[TensorArg],
        outputs: List[TensorArg],
        target_shape: List[sympy.Expr],
        reduce_axes: List[int],
        reduced_args: Set[str],
    ):
        super().__init__(inputs, outputs, target_shape, reduce_axes)
        self.offset_calc.reduced_args.update(reduced_args)


class ModuleNode(IRNode):
    """
    The ModuleNode is used to represent the whole subgraph. It may contain multiple kernels.

    """

    def __init__(
        self,
        func_name: str,
        inputs: List[TensorArg],
        outputs: List[TensorArg],
        constants: List[TensorArg],
        cross_kernel_args: List[Tuple[TensorArg, int]],
        kernels: List[KernelNode],
    ):
        super().__init__(inputs, outputs)
        self.func_name: str = func_name
        # Currently need inputs and outputs only. May need intermediate vars and constants later.
        self.constants: List[TensorArg] = constants
        self.kernels: List[KernelNode] = kernels
        self.var_map: Dict[str, str] = dict()
        existing_names = set()
        for input in self.inputs:
            name = gen_variable_name(input.name, "in", existing_names)
            self.var_map[input.name] = name
        for output in self.outputs:
            name = gen_variable_name(output.name, "out", existing_names)
            self.var_map[output.name] = name
        self.cross_kernel_args_to_delete: Dict[int, Set[str]] = defaultdict(set)
        for pair in cross_kernel_args:
            name = gen_variable_name(pair[0].name, "buf", existing_names)
            self.cross_kernel_args_to_delete[pair[1]].add(name)
            self.var_map[pair[0].name] = name
        running_offset = sympy.Integer(0)
        self.has_dropout: bool = False
        for kernel in self.kernels:
            for ir_node in kernel.sub_nodes:
                if isinstance(ir_node, DropoutNode):
                    ir_node.global_offset = running_offset
                    running_offset = running_offset + sympy.prod(ir_node.outputs[0].shape)
                    self.has_dropout = True
