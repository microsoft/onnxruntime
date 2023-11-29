# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import itertools
import warnings
from collections import defaultdict
from typing import Any, Dict, List, Set, Tuple

import sympy
from onnx import NodeProto, helper

from ._common import AutotuneConfigs, TensorInfo
from ._ir import (
    ComputeNode,
    DropoutNode,
    ElementwiseKernelNode,
    IONode,
    KernelNode,
    ModuleNode,
    OffsetCalculator,
    ReduceForLoopEnd,
    ReduceForLoopStart,
    ReduceKernelNode,
    ReduceNode,
    TensorArg,
)
from ._op_config import is_reduction_node
from ._sorted_graph import SortedGraph
from ._utils import get_reduce_info, sort_reduce_axes, to_numpy_array


class NodeGroup:
    """
    A NodeGroup contains nodes that can be lowered to a single Triton kernel node.

    """

    def __init__(self, node: NodeProto, reduce_axes: List[int], keep_dims: int, node_arg_infos: Dict[str, TensorInfo]):
        self._node_arg_infos = node_arg_infos
        self.nodes_groups: List[Any] = [node]
        self.target_shape: List[sympy.Expr] = self._get_target_shape(node)
        rank = len(self.target_shape)
        self.reduce_axes: List[int] = sort_reduce_axes(reduce_axes, rank)
        x_dims = [self.target_shape[dim] for dim in range(rank) if dim not in self.reduce_axes]
        # x_numel is meant to hint how many rows of tensor will be processed by each kernel.
        # x is same as CUDA block in X direction.
        x_numel: sympy.Expr = sympy.prod(x_dims) if len(x_dims) > 0 else sympy.Integer(1)
        r_dims: List[sympy.Expr] = [self.target_shape[dim] for dim in self.reduce_axes]
        # r_numel is meant to hint how many elements in a row of tensor will be processed by each kernel.
        # r is a abbreviation of reduction, so, it's only used for reduction nodes.
        r_numel: sympy.Expr = sympy.prod(r_dims) if len(r_dims) > 0 else sympy.Integer(1)
        self.autotune_configs: AutotuneConfigs = AutotuneConfigs(
            x_numel, r_numel, len(self.reduce_axes) == 0 or self.reduce_axes[-1] == rank - 1
        )
        self.reduced_args: Set[str] = set()
        if keep_dims != 1:
            self.reduced_args.add(node.output[0])

    # Check if shape can be broadcasted to target_shape.
    # For example, [1, 3, 1, 1] can be broadcasted to [1, 3, 5, 7].
    # and we support `keepdims = false``, so [1, 3, 5, 7] is compatible with [1, 3, 5].
    def _compatible_shape(self, shape: List[sympy.Expr], split_if_different: bool) -> bool:
        if split_if_different:
            return shape == self.target_shape
        if len(shape) > len(self.target_shape):
            return False
        shape = [sympy.Integer(1)] * (len(self.target_shape) - len(shape)) + shape
        for axis, dim in enumerate(shape):
            if dim != self.target_shape[axis] and (not dim.is_number or dim != sympy.Integer(1)):
                return False
        return True

    # Only we consider reduction or elementwise nodes.
    # target shape does effect how we block the tensor in a triton kernel
    # for reduction, it's possible to set keepdims=False
    # for element-wise, output shape is always the target shape.
    def _get_target_shape(self, node):
        name = node.input[0] if is_reduction_node(node) else node.output[0]
        return self._node_arg_infos[name].shape

    # Check if a node can be added to this group.
    # a group represents a single kernel.
    # Theoretically, we should fuse as more nodes as possible to benefit most from memory access pattern.
    # But we have to consider the following factors:
    #     1. We have to keep the order of nodes, so that we can't fuse nodes that are not adjacent.
    #     2. The target shape of a group is determined by the first node in the group.
    #       we call it dominators, and it determinate the partition strategy of X_numel/R_numel.
    #       A group can't have multiple dominators.
    def compatible(self, node: NodeProto, reduce_axes: List[int], keep_dims: int, split_if_different: bool) -> bool:
        target_shape = self._get_target_shape(node)
        if is_reduction_node(node):
            # If the following nodes are all elementwise nodes on reduce output shape.
            if len(self.reduce_axes) == 0 and self.target_shape == self._node_arg_infos[node.output[0]].shape:
                return True
            if keep_dims != 1:
                return False
            if (
                len(self.reduce_axes) > 0 and self.reduce_axes != sort_reduce_axes(reduce_axes, len(target_shape))
            ) or self.target_shape != target_shape:
                return False
            return True
        return self._compatible_shape(target_shape, split_if_different)

    # 1. Create a new group with the reduction node.
    # 2. Add this node to the current group.
    def add_node(self, node: NodeProto, reduce_axes: List[int], keep_dims: int):
        if is_reduction_node(node):
            group = NodeGroup(node, reduce_axes, keep_dims, self._node_arg_infos)
            self.nodes_groups.append(group)
            if len(self.reduce_axes) == 0:
                self.target_shape = group.target_shape
                self.reduce_axes = group.reduce_axes
                self.autotune_configs = group.autotune_configs
                if keep_dims != 1:
                    for idx in range(len(self.nodes_groups) - 1):
                        self.reduced_args.update(self.nodes_groups[idx].input)
                        self.reduced_args.update(self.nodes_groups[idx].output)
            return group
        self.nodes_groups.append(node)
        return self

    def has_reduced_elementwise_nodes(self) -> bool:
        return not is_reduction_node(self.nodes_groups[0]) and len(self.reduced_args) > 0

    def dependent_nodes(self, keep_reduce_node: bool):
        node_map = {}
        reduce_nodes = []
        if not keep_reduce_node and self.has_reduced_elementwise_nodes():
            for item in self.nodes_groups:
                if not isinstance(item, NodeGroup):
                    node_map[item.name] = item
            return node_map, reduce_nodes
        for item in self.nodes_groups:
            if isinstance(item, NodeGroup):
                node_map.update(item.dependent_nodes(keep_reduce_node)[0])
            elif keep_reduce_node or not is_reduction_node(item):
                node_map[item.name] = item
            else:
                reduce_nodes.append(item)
        return node_map, reduce_nodes

    # finalize the group, and return the flatten nodes
    def flatten(self, sorted_nodes: List[NodeProto]) -> Tuple[List[NodeProto], List[List[int]]]:
        if self.autotune_configs.requires_for_loop:
            layers = []
            group_layer = [self]
            while len(group_layer) > 0:
                node_map = {}
                reduce_node_map = {}
                next_layer = []
                for group in group_layer:
                    sub_node_map, reduce_nodes = group.dependent_nodes(False)
                    node_map.update(sub_node_map)
                    for node in reduce_nodes:
                        reduce_node_map[node.name] = node
                    next_layer.extend([item for item in group.nodes_groups if isinstance(item, NodeGroup)])
                layers.append((node_map, reduce_node_map.values()))
                group_layer = next_layer
            nodes = []
            layer_indices = []
            for i in range(len(layers) - 1, -1, -1):
                sub_nodes = list(layers[i][0].values())
                sub_nodes.sort(key=sorted_nodes.index)
                nodes.extend(sub_nodes)
                sub_layer_indices = []
                for node in layers[i][1]:
                    nodes.append(node)
                    sub_layer_indices.append(len(nodes) - 1)
                layer_indices.append(sub_layer_indices)
            return nodes, layer_indices
        node_map, _ = self.dependent_nodes(True)
        nodes = list(node_map.values())
        nodes.sort(key=sorted_nodes.index)
        return nodes, []

    def try_merge(self, other) -> bool:
        if (
            self.target_shape != other.target_shape
            or self.reduce_axes != other.reduce_axes
            or self.has_reduced_elementwise_nodes() != other.has_reduced_elementwise_nodes()
        ):
            return False
        self.nodes_groups.extend(other.nodes_groups)
        self.reduced_args.update(other.reduced_args)
        return True


class KernelIO:
    """
    Used to represent the inputs and outputs of a kernel(triton kernel).
    """

    def __init__(self):
        self.module_inputs: List[str] = []
        self.cross_kernel_inputs: List[str] = []
        self.constants: List[str] = []
        self.module_outputs: List[str] = []
        self.cross_kernel_outputs: List[str] = []
        self.internal_args: List[str] = []


class GraphLowering:
    """
    GraphLowering does manager all steps of lowering onnx graph to triton irnode.
    1. partition the graph into kernels (one or more kernels).
        Manager to allocate inputs, outputs and buffers reuse between kernels.
        we call it Module
    2. convert kernel to irnodes.
    3. analyze the IR relationship and buffer/Tensor inside a kernel.
    4. Generate the Auto-Tune configs for each kernel, Tunning speed/Running faster depends.

    we will end up getting a tree-liked IR structure with explicit input/output and intermediate buffer.

    """

    def __init__(self, sorted_graph: SortedGraph):
        self._sorted_graph: SortedGraph = sorted_graph
        self._node_arg_infos: Dict[str, TensorInfo] = sorted_graph.node_arg_infos
        self._module_inputs: List[TensorArg] = []
        self._module_outputs: List[TensorArg] = []
        self._module_constants: List[TensorArg] = []
        self._module_input_names: Set[str] = set()
        self._module_output_names: Set[str] = set()
        self._module_constant_names: Set[str] = set()
        self._tensor_args: Dict[str, TensorArg] = {}
        # Extract module inputs, outputs and constants.
        self._extract_module_io()

        # Group nodes into NodeGroups, each NodeGroup represents a kernel.
        self._groups: List[NodeGroup] = []
        self._group_nodes()

        # Convert NodeGroups to KernelNodes.
        self._kernel_nodes: List[KernelNode] = []
        self._kernel_io_list: List[KernelIO] = []
        self._lower()

    # A module is map to a real onnx graph.
    def _extract_module_io(self):
        graph = self._sorted_graph.original_graph
        self._module_inputs = [TensorArg(input.name, self._node_arg_infos[input.name]) for input in graph.input]
        self._module_input_names = set(arg.name for arg in self._module_inputs)
        self._module_outputs = [TensorArg(output.name, self._node_arg_infos[output.name]) for output in graph.output]
        self._module_output_names = set(arg.name for arg in self._module_outputs)
        for initializer in graph.initializer:
            data = to_numpy_array(initializer)
            self._module_constants.append(TensorArg(initializer.name, data=data))
        for const_node in self._sorted_graph.const_nodes:
            data = to_numpy_array(const_node)
            self._module_constants.append(TensorArg(const_node.output[0], data=data))
        self._module_constant_names = set(arg.name for arg in self._module_constants)
        self._tensor_args = dict(
            (arg.name, arg)
            for arg in itertools.chain(self._module_inputs, self._module_outputs, self._module_constants)
        )

    def _get_reduce_info(self, node) -> Tuple[int, List[int]]:
        assert is_reduction_node(node)
        input_rank = len(self._node_arg_infos[node.input[0]].shape)
        return get_reduce_info(node, self._sorted_graph.original_graph, input_rank)

    def _process_node(self, node: NodeProto, precessors: Dict[str, List[NodeProto]], group: NodeGroup):
        dependent_nodes = set()
        dependent_nodes.add(node.name)
        for precessor in precessors[node.name]:
            if precessor.name in dependent_nodes:
                continue
            keep_dims = 1
            reduce_axes = []
            if is_reduction_node(precessor):
                keep_dims, reduce_axes = self._get_reduce_info(precessor)
            split_if_different = any(
                output in self._sorted_graph.elementwise_graph_outputs for output in precessor.output
            )
            if group.compatible(precessor, reduce_axes, keep_dims, split_if_different):
                next_group = group.add_node(precessor, reduce_axes, keep_dims)
                dependent_nodes.update(self._process_node(precessor, precessors, next_group))
        return dependent_nodes

    def _group_nodes(self):
        producers = {}
        precessors = defaultdict(list)
        processed = set()
        groups = []
        sorted_nodes = self._sorted_graph.sorted_nodes
        for node in sorted_nodes:
            for output in node.output:
                producers[output] = node
            for input in node.input:
                if input in producers:
                    precessors[node.name].append(producers[input])
        for value in precessors.values():
            value.sort(key=sorted_nodes.index, reverse=True)
        for idx in range(len(sorted_nodes) - 1, -1, -1):
            node = sorted_nodes[idx]
            if node.name not in processed:
                reduce_axes = []
                keep_dims = 1
                if is_reduction_node(node):
                    keep_dims, reduce_axes = self._get_reduce_info(node)
                groups.append(NodeGroup(node, reduce_axes, keep_dims, self._node_arg_infos))
                processed.update(self._process_node(node, precessors, groups[-1]))

        # Merge groups with same target shape and reduce axes without dependency.
        group_dependencies = defaultdict(set)
        for i in range(len(groups) - 1):
            group_inputs = set()
            for node in groups[i].dependent_nodes(True)[0].values():
                group_inputs.update(node.input)
            for j in range(i + 1, len(groups)):
                if any(output in group_inputs for output in groups[j].nodes_groups[0].output):
                    group_dependencies[i].add(j)
                    for k in range(0, i):
                        if i in group_dependencies[k]:
                            group_dependencies[k].add(j)

        flag = set()
        for i, group_i in enumerate(groups):
            if i in flag:
                continue
            for j, group_j in enumerate(groups):
                if j <= i:
                    continue
                if j not in flag and j not in group_dependencies[i] and group_i.try_merge(group_j):
                    flag.add(j)
            self._groups.append(group_i)
            flag.add(i)

    def _get_node_io(self, node: NodeProto) -> Tuple[List[TensorArg], List[TensorArg]]:
        input_args = []
        for input in node.input:
            if input in self._tensor_args:
                input_args.append(self._tensor_args[input])
            else:
                input_args.append(TensorArg(input, self._node_arg_infos[input]))
                self._tensor_args[input] = input_args[-1]
        output_args = []
        for output in node.output:
            if output in self._tensor_args:
                output_args.append(self._tensor_args[output])
            else:
                output_args.append(TensorArg(output, self._node_arg_infos[output]))
                self._tensor_args[output] = output_args[-1]
        return input_args, output_args

    def _extract_kernel_io(self, nodes: List[NodeProto]) -> KernelIO:
        kernel_io = KernelIO()
        input_set = set()
        output_set = set()
        for node in nodes:
            for input in node.input:
                if input in input_set:
                    continue
                elif input in self._module_constant_names:
                    kernel_io.constants.append(input)
                elif input in self._module_input_names:
                    kernel_io.module_inputs.append(input)
                elif input not in output_set:
                    kernel_io.cross_kernel_inputs.append(input)
                input_set.add(input)
            for output in node.output:
                if output in output_set:
                    continue
                if output in self._module_output_names:
                    kernel_io.module_outputs.append(output)
                else:
                    kernel_io.internal_args.append(output)
                output_set.add(output)
        return kernel_io

    def _to_compute_node(self, node: NodeProto, offset_calc: OffsetCalculator):
        inputs, outputs = self._get_node_io(node)
        op_type = node.op_type
        if op_type == "Dropout":
            return DropoutNode(inputs, outputs, offset_calc)
        if is_reduction_node(node):
            return ReduceNode(op_type, inputs, outputs, offset_calc)
        attributes = {}
        for attr in node.attribute:
            attributes[attr.name] = helper.get_attribute_value(attr)
        return ComputeNode(op_type, inputs, outputs, attributes)

    def _analyze_kernel_io_list(self):
        cross_kernel_inputs = set()
        for kernel_io in self._kernel_io_list:
            cross_kernel_inputs.update(kernel_io.cross_kernel_inputs)
        for kernel_io in self._kernel_io_list:
            kernel_io.cross_kernel_outputs = [arg for arg in kernel_io.internal_args if arg in cross_kernel_inputs]
            kernel_io.internal_args = [
                arg for arg in kernel_io.internal_args if arg not in kernel_io.cross_kernel_outputs
            ]

    def _insert_load_and_store(self, kernel_node: KernelNode):
        input_names = [input.name for input in kernel_node.inputs]
        output_name_map = {}
        for output in kernel_node.outputs:
            output_name_map[output.name] = 0
        for node in kernel_node.sub_nodes:
            for output in node.outputs:
                if output.name in output_name_map:
                    output_name_map[output.name] += 1
        sub_nodes = kernel_node.sub_nodes
        new_sub_nodes = []
        cur = 0
        nxt = 0
        reduce_store_nodes = []
        while True:
            while nxt < len(sub_nodes) and not isinstance(sub_nodes[nxt], ReduceForLoopEnd):
                nxt += 1
            load_cache = set()
            load_nodes = []
            store_nodes = []
            for idx in range(cur, nxt):
                for input in sub_nodes[idx].inputs:
                    if input.name in kernel_node.constants or input.name in input_names:
                        if (input.data is not None and input.data.size == 1) or input.name in load_cache:
                            continue
                        load_nodes.append(IONode(input, kernel_node.offset_calc, True))
                        load_cache.add(input.name)
                for output in sub_nodes[idx].outputs:
                    if output.name in output_name_map:
                        output_name_map[output.name] -= 1
                        if output_name_map[output.name] == 0:
                            store_nodes.append(IONode(output, kernel_node.offset_calc, False))
            if isinstance(sub_nodes[cur], ReduceForLoopStart):
                new_sub_nodes.append(sub_nodes[cur])
                cur += 1
            if nxt < len(sub_nodes):
                assert isinstance(sub_nodes[nxt], ReduceForLoopEnd)
                for reduce_node in sub_nodes[nxt].reduce_nodes:
                    input = reduce_node.inputs[0]
                    if input.name in kernel_node.constants or input.name in input_names:
                        if (input.data is not None and input.data.size == 1) or input.name in load_cache:
                            continue
                        load_nodes.append(IONode(input, kernel_node.offset_calc, True))
                        load_cache.add(input.name)
            new_sub_nodes.extend(load_nodes)
            new_sub_nodes.extend(sub_nodes[cur:nxt])
            new_sub_nodes.extend(store_nodes)
            if nxt < len(sub_nodes):
                assert isinstance(sub_nodes[nxt], ReduceForLoopEnd)
                for reduce_node in sub_nodes[nxt].reduce_nodes:
                    if reduce_node.outputs[0].name in output_name_map:
                        reduce_store_nodes.append(IONode(reduce_node.outputs[0], kernel_node.offset_calc, False))
                new_sub_nodes.append(sub_nodes[nxt])
                nxt += 1
            cur = nxt
            if cur >= len(sub_nodes):
                break
        new_sub_nodes.extend(reduce_store_nodes)
        kernel_node.sub_nodes = new_sub_nodes

    def _lower(self):
        for group in self._groups:
            is_reduction_kernel = len(group.reduce_axes) > 0
            target_shape = group.target_shape
            # The inputs and outputs will be initialized later.
            kernel_node = (
                ReduceKernelNode([], [], target_shape, group.reduce_axes, group.reduced_args)
                if is_reduction_kernel
                else ElementwiseKernelNode([], [], target_shape)
            )
            self._kernel_nodes.append(kernel_node)
            sub_nodes = []
            nodes, layer_indices = group.flatten(self._sorted_graph.sorted_nodes)
            self._kernel_io_list.append(self._extract_kernel_io(nodes))
            if group.autotune_configs.requires_for_loop:
                start = 0
                for layer_idx, indices in enumerate(layer_indices):
                    need_for_loop = True
                    if layer_idx == len(layer_indices) - 1 and group.has_reduced_elementwise_nodes():
                        assert len(indices) == 0
                        need_for_loop = False
                    reduce_nodes = [self._to_compute_node(nodes[idx], kernel_node.offset_calc) for idx in indices]
                    assert all(isinstance(node, ReduceNode) for node in reduce_nodes)
                    if need_for_loop:
                        sub_nodes.append(ReduceForLoopStart(reduce_nodes, kernel_node.offset_calc))
                    end = indices[0] if len(indices) > 0 else len(nodes)
                    for idx in range(start, end):
                        node = nodes[idx]
                        assert not is_reduction_node(node)
                        sub_nodes.append(self._to_compute_node(node, kernel_node.offset_calc))
                        if node.op_type == "Dropout":
                            self._kernel_nodes[-1].has_dropout = True
                    if len(indices) > 0:
                        sub_nodes.append(ReduceForLoopEnd(reduce_nodes, kernel_node.offset_calc))
                    start = indices[len(indices) - 1] + 1 if len(indices) > 0 else len(nodes)
            else:
                for node in nodes:
                    sub_nodes.append(self._to_compute_node(node, kernel_node.offset_calc))
                    if node.op_type == "Dropout":
                        self._kernel_nodes[-1].has_dropout = True
            self._kernel_nodes[-1].sub_nodes = sub_nodes

        if any(kernel_node.has_dropout for kernel_node in self._kernel_nodes):
            warnings.warn("Use triton's random for Dropout, ignore the random seed from ORT.", UserWarning)

        self._analyze_kernel_io_list()
        cross_kernel_arg_map = {}
        for idx, kernel_io in enumerate(self._kernel_io_list):
            for output in itertools.chain(kernel_io.cross_kernel_outputs, kernel_io.module_outputs):
                cross_kernel_arg_map[output] = idx
        dependency = defaultdict(set)
        for idx, kernel_io in enumerate(self._kernel_io_list):
            for input in kernel_io.cross_kernel_inputs:
                dependency[cross_kernel_arg_map[input]].add(idx)
        visited = set()
        sorted_indices = []

        def _topological_sort_internal(idx):
            visited.add(idx)
            for next_idx in dependency[idx]:
                if next_idx not in visited:
                    _topological_sort_internal(next_idx)
            sorted_indices.insert(0, idx)

        for idx in range(len(self._kernel_io_list)):
            if idx not in visited:
                _topological_sort_internal(idx)

        self._kernel_nodes = [self._kernel_nodes[idx] for idx in sorted_indices]
        self._kernel_io_list = [self._kernel_io_list[idx] for idx in sorted_indices]
        cross_kernel_arg_map.clear()
        for idx, kernel_io in enumerate(self._kernel_io_list):
            for arg in kernel_io.cross_kernel_inputs:
                if arg not in self._module_output_names:
                    cross_kernel_arg_map[arg] = idx

        self._cross_kernel_args = [(self._tensor_args[key], value) for key, value in cross_kernel_arg_map.items()]

        for idx, kernel_node in enumerate(self._kernel_nodes):
            kernel_io = self._kernel_io_list[idx]
            kernel_node.internal_args.update(kernel_io.internal_args)
            kernel_node.inputs = [
                self._tensor_args[name]
                for name in itertools.chain(kernel_io.module_inputs, kernel_io.cross_kernel_inputs)
            ]
            kernel_node.outputs = [
                self._tensor_args[name]
                for name in itertools.chain(kernel_io.module_outputs, kernel_io.cross_kernel_outputs)
            ]
            for name in kernel_io.constants:
                kernel_node.constants[name] = self._tensor_args[name]
            self._insert_load_and_store(kernel_node)
            kernel_node.gen_variable_names()

    def module_node(self, func_name: str):
        return ModuleNode(
            func_name,
            self._module_inputs,
            self._module_outputs,
            self._module_constants,
            self._cross_kernel_args,
            self._kernel_nodes,
        )


def lower(func_name: str, sorted_graph: SortedGraph) -> ModuleNode:
    return GraphLowering(sorted_graph).module_node(func_name)
