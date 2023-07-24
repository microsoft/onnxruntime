# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""
Decompose a complicated op into a series of simple ops.
"simple ops" can be executed in one pass
"""

from typing import List

import numpy as np
import sympy
from onnx import GraphProto, NodeProto, TensorProto, helper

from ._utils import get_attribute, get_reduce_info, to_numpy_type


def _is_half_dtype(dtype: int):
    return dtype in [TensorProto.FLOAT16, TensorProto.BFLOAT16]


class DecomposeDispatch:
    """
    A node does only responsible for a single computation or a type of triton ops.
    For those compound Onnx nodes, like softmax/layernorm/groupnorm, etc., we need to decompose them into a series of
    simple ops.
    """

    def __init__(self):
        self.count = 0

    def __call__(self, node: NodeProto, graph: GraphProto, **kwargs) -> List[NodeProto]:
        op_type = node.op_type
        if not hasattr(self, op_type):
            raise NotImplementedError(f"Not implemented for op type: {op_type}")
        return getattr(self, op_type)(node, graph, **kwargs)

    def __contains__(self, node: NodeProto) -> bool:
        return hasattr(self, node.op_type)

    def _get_unique_var_name(self, prefix):
        self.count += 1
        return prefix + str(self.count)

    def _new_node(self, node_name, op_type, inputs, outputs=None, **kwargs):
        name = self._get_unique_var_name(f"{node_name}_{op_type}_")
        if outputs is None:
            outputs = [f"{name}_out"]
        for idx, output in enumerate(outputs):
            if output is None:
                outputs[idx] = f"{name}_out{idx}"
        return helper.make_node(op_type, inputs, outputs, name, **kwargs), *outputs

    def _get_dtype_and_shape(self, arg_name: str, **kwargs):
        node_arg_infos = kwargs["node_arg_infos"]
        arg_info = node_arg_infos[arg_name]
        return arg_info.dtype, arg_info.shape

    def _decompose_elementwise_precision(self, node: NodeProto, graph: GraphProto, **kwargs):
        x = node.input[0]
        dtype, _ = self._get_dtype_and_shape(x, **kwargs)
        if not _is_half_dtype(dtype):
            return [node]
        node_name = node.name
        y = node.output[0]
        op_type = node.op_type
        inputs = [input for input in node.input]
        cast_nodes = []
        for idx, input in enumerate(inputs):
            dtype, _ = self._get_dtype_and_shape(input, **kwargs)
            if _is_half_dtype(dtype):
                cast_node, cast_out = self._new_node(node_name, "Cast", [input], to=TensorProto.FLOAT)
                cast_nodes.append(cast_node)
                inputs[idx] = cast_out
        op_node, op_out = self._new_node(node_name, op_type, inputs)
        cast_node1, _ = self._new_node(node_name, "Cast", [op_out], outputs=[y], to=dtype)
        return [*cast_nodes, op_node, cast_node1]

    def Exp(self, node: NodeProto, graph: GraphProto, **kwargs):  # noqa: N802
        return self._decompose_elementwise_precision(node, graph, **kwargs)

    def Pow(self, node: NodeProto, graph: GraphProto, **kwargs):  # noqa: N802
        return self._decompose_elementwise_precision(node, graph, **kwargs)

    def Sqrt(self, node: NodeProto, graph: GraphProto, **kwargs):  # noqa: N802
        return self._decompose_elementwise_precision(node, graph, **kwargs)

    def LayerNormalization(self, node: NodeProto, graph: GraphProto, **kwargs):  # noqa: N802
        node_name = node.name
        x = node.input[0]
        w = node.input[1]
        b = node.input[2]
        y = node.output[0]
        mean = node.output[1] if len(node.output) > 1 and node.output[1] else None
        inv_std_dev = node.output[2] if len(node.output) > 2 and node.output[2] else None
        axis = get_attribute(node, "axis", -1)
        epsilon = get_attribute(node, "epsilon", 1e-05)
        xdtype, shape = self._get_dtype_and_shape(x, **kwargs)
        wdtype, _ = self._get_dtype_and_shape(w, **kwargs)
        is_x_half = _is_half_dtype(xdtype)
        is_w_half = _is_half_dtype(wdtype)
        if is_x_half or is_w_half:
            decomposed_nodes = []
            if is_x_half:
                cast_node, x = self._new_node(node_name, "Cast", [x], to=TensorProto.FLOAT)
                decomposed_nodes.append(cast_node)
            if is_w_half:
                cast_node1, w = self._new_node(node_name, "Cast", [w], to=TensorProto.FLOAT)
                cast_node2, b = self._new_node(node_name, "Cast", [b], to=TensorProto.FLOAT)
                decomposed_nodes.append(cast_node1)
                decomposed_nodes.append(cast_node2)
            outputs = [None] if is_w_half else [y]
            if mean is not None:
                outputs.append(mean)
            if inv_std_dev is not None:
                outputs.append(inv_std_dev)
            layer_norm_node_outputs = self._new_node(
                node_name, "LayerNormalization", [x, w, b], outputs=outputs, axis=axis, epsilon=epsilon
            )
            decomposed_nodes.append(layer_norm_node_outputs[0])
            if is_w_half:
                cast_node3, _ = self._new_node(node_name, "Cast", [layer_norm_node_outputs[1]], outputs=[y], to=wdtype)
                decomposed_nodes.append(cast_node3)
            return decomposed_nodes
        rank = len(shape)
        if axis < 0:
            axis += rank
        axes = list(range(axis, rank))
        epsilon_tensor = helper.make_tensor(name="epsilon_const", data_type=xdtype, dims=(1,), vals=np.array([epsilon]))
        const_node, const_out = self._new_node(node_name, "Constant", [], value=epsilon_tensor)
        reducemean_node, reducemean_out = self._new_node(node_name, "ReduceMean", [x], outputs=[mean], axes=axes)
        sub_node, sub_out = self._new_node(node_name, "Sub", [x, reducemean_out])
        mul_node, mul_out = self._new_node(node_name, "Mul", [sub_out, sub_out])
        reducemean_node1, reducemean_out1 = self._new_node(node_name, "ReduceMean", [mul_out], axes=axes)
        add_node, add_out = self._new_node(node_name, "Add", [reducemean_out1, const_out])
        rsqrt_node, rsqrt_out = self._new_node(node_name, "Rsqrt", [add_out], outputs=[inv_std_dev])
        mul_node1, mul_out1 = self._new_node(node_name, "Mul", [sub_out, rsqrt_out])
        mul_node2, mul_out2 = self._new_node(node_name, "Mul", [w, mul_out1])
        add_node1, _ = self._new_node(node_name, "Add", [b, mul_out2], outputs=[y])
        return [
            const_node,
            reducemean_node,
            sub_node,
            mul_node,
            reducemean_node1,
            add_node,
            rsqrt_node,
            mul_node1,
            mul_node2,
            add_node1,
        ]

    def LayerNormalizationGrad(self, node: NodeProto, graph: GraphProto, **kwargs):  # noqa: N802
        node_name = node.name
        dy = node.input[0]
        x = node.input[1]
        w = node.input[2]
        mean = node.input[3]
        inv_std_dev = node.input[4]
        dx = node.output[0]
        dw = node.output[1] if len(node.output) > 1 and node.output[1] else None
        db = node.output[2] if len(node.output) > 2 and node.output[2] else None
        axis = get_attribute(node, "axis", -1)
        xdtype, shape = self._get_dtype_and_shape(x, **kwargs)
        wdtype, _ = self._get_dtype_and_shape(w, **kwargs)
        is_x_half = _is_half_dtype(xdtype)
        is_w_half = _is_half_dtype(wdtype)
        if is_x_half or is_w_half:
            decomposed_nodes = []
            if is_x_half:
                cast_node, x = self._new_node(node_name, "Cast", [x], to=TensorProto.FLOAT)
                decomposed_nodes.append(cast_node)
            if is_w_half:
                cast_node1, dy = self._new_node(node_name, "Cast", [dy], to=TensorProto.FLOAT)
                cast_node2, w = self._new_node(node_name, "Cast", [w], to=TensorProto.FLOAT)
                decomposed_nodes.append(cast_node1)
                decomposed_nodes.append(cast_node2)
            outputs = [None] if is_x_half else [dx]
            if dw is not None:
                outputs.append(None if is_w_half else dw)
            if db is not None:
                outputs.append(None if is_w_half else db)
            layer_norm_grad_node_outputs = self._new_node(
                node_name, "LayerNormalizationGrad", [dy, x, w, mean, inv_std_dev], outputs=outputs, axis=axis
            )
            decomposed_nodes.append(layer_norm_grad_node_outputs[0])
            if is_x_half:
                cast_node3, _ = self._new_node(
                    node_name, "Cast", [layer_norm_grad_node_outputs[1]], outputs=[dx], to=xdtype
                )
                decomposed_nodes.append(cast_node3)
            if dw is not None and is_w_half:
                cast_node4, _ = self._new_node(
                    node_name, "Cast", [layer_norm_grad_node_outputs[2]], outputs=[dw], to=wdtype
                )
                decomposed_nodes.append(cast_node4)
            if db is not None and is_w_half:
                cast_node5, _ = self._new_node(
                    node_name, "Cast", [layer_norm_grad_node_outputs[3]], outputs=[db], to=wdtype
                )
                decomposed_nodes.append(cast_node5)
            return decomposed_nodes
        rank = len(shape)
        if axis < 0:
            axis += rank
        axes = list(range(axis, rank))
        sub_node, sub_out = self._new_node(node_name, "Sub", [x, mean])
        mul_node, mul_out = self._new_node(node_name, "Mul", [sub_out, inv_std_dev])
        mul_node1, mul_out1 = self._new_node(node_name, "Mul", [w, dy])
        mul_node2, mul_out2 = self._new_node(node_name, "Mul", [mul_out, mul_out1])
        reducemean_node, reducemean_out = self._new_node(node_name, "ReduceMean", [mul_out2], axes=axes)
        reducemean_node1, reducemean_out1 = self._new_node(node_name, "ReduceMean", [mul_out1], axes=axes)
        mul_node3, mul_out3 = self._new_node(node_name, "Mul", [reducemean_out, mul_out])
        add_node, add_out = self._new_node(node_name, "Add", [mul_out3, reducemean_out1])
        sub_node1, sub_out1 = self._new_node(node_name, "Sub", [mul_out1, add_out])
        mul_node4, _ = self._new_node(node_name, "Mul", [sub_out1, inv_std_dev], outputs=[dx])
        decomposed_nodes = [
            sub_node,
            mul_node,
            mul_node1,
            mul_node2,
            reducemean_node,
            reducemean_node1,
            mul_node3,
            add_node,
            sub_node1,
            mul_node4,
        ]
        dw_axes = list(range(axis))
        if dw is not None:
            mul_node5, mul_out5 = self._new_node(node_name, "Mul", [dy, mul_out])
            reducesum_node, _ = self._new_node(
                node_name, "ReduceSum", [mul_out5], outputs=[dw], axes=dw_axes, keepdims=0
            )
            decomposed_nodes.extend([mul_node5, reducesum_node])
        if db is not None:
            reducesum_node1, _ = self._new_node(node_name, "ReduceSum", [dy], outputs=[db], axes=dw_axes, keepdims=0)
            decomposed_nodes.append(reducesum_node1)
        return decomposed_nodes

    def Softmax(self, node: NodeProto, graph: GraphProto, **kwargs):  # noqa: N802
        node_name = node.name
        x = node.input[0]
        y = node.output[0]
        axis = get_attribute(node, "axis", -1)
        dtype, _ = self._get_dtype_and_shape(x, **kwargs)
        if _is_half_dtype(dtype):
            cast_node, x = self._new_node(node_name, "Cast", [x], to=TensorProto.FLOAT)
            softmax_node, softmax_out = self._new_node(node_name, "Softmax", [x], axis=axis)
            cast_node1, _ = self._new_node(node_name, "Cast", [softmax_out], outputs=[y], to=dtype)
            return [cast_node, softmax_node, cast_node1]
        max_node, max_out = self._new_node(node_name, "ReduceMax", [x], axes=[axis])
        sub_node, sub_out = self._new_node(node_name, "Sub", [x, max_out])
        exp_node, exp_out = self._new_node(node_name, "Exp", [sub_out])
        sum_node, sum_out = self._new_node(node_name, "ReduceSum", [exp_out], axes=[axis])
        div_node, _ = self._new_node(node_name, "Div", [exp_out, sum_out], outputs=[y])
        return [max_node, sub_node, exp_node, sum_node, div_node]

    def SoftmaxGrad_13(self, node: NodeProto, graph: GraphProto, **kwargs):  # noqa: N802
        node_name = node.name
        dy = node.input[0]
        y = node.input[1]
        dx = node.output[0]
        axis = get_attribute(node, "axis", -1)
        dtype, _ = self._get_dtype_and_shape(dy, **kwargs)
        if _is_half_dtype(dtype):
            cast_node, dy = self._new_node(node_name, "Cast", [dy], to=TensorProto.FLOAT)
            cast_node1, y = self._new_node(node_name, "Cast", [y], to=TensorProto.FLOAT)
            softmax_grad_node, softmax_grad_out = self._new_node(node_name, "SoftmaxGrad_13", [dy, y], axis=axis)
            cast_node2, _ = self._new_node(node_name, "Cast", [softmax_grad_out], outputs=[dx], to=dtype)
            return [cast_node, cast_node1, softmax_grad_node, cast_node2]
        mul_node, mul_out = self._new_node(node_name, "Mul", [dy, y])
        sum_node, sum_out = self._new_node(node_name, "ReduceSum", [mul_out], axes=[axis])
        mul_node1, mul_out1 = self._new_node(node_name, "Mul", [y, sum_out])
        sub_node, _ = self._new_node(node_name, "Sub", [mul_out, mul_out1], outputs=[dx])
        return [mul_node, sum_node, mul_node1, sub_node]

    # We support to codegen for reduce Ops that are reducing on contiguous axes. If it's not, we need to decompose
    # it to multiple reduce Ops.
    def _decompose_reduce_axes(self, node: NodeProto, graph: GraphProto, **kwargs):
        node_name = node.name
        op_type = node.op_type
        x = node.input[0]
        y = node.output[0]
        _, shape = self._get_dtype_and_shape(x, **kwargs)
        rank = len(shape)
        keep_dims, axes = get_reduce_info(node, graph, rank)
        if len(axes) == 0:
            identity_node, _ = self._new_node(node_name, "Identity", [x], outputs=[y])
            return [identity_node]
        splited_axes = []
        end = len(axes)
        start = end - 1
        while True:
            while start > 0 and axes[start] == axes[start - 1] + 1:
                start -= 1
            splited_axes.append(axes[start:end])
            if start == 0:
                break
            end = start
            start = end - 1
        if len(splited_axes) == 1:
            if len(node.input) <= 1:
                return [node]
            reduce_node, _ = self._new_node(node_name, op_type, [x], outputs=[y], axes=axes, keepdims=keep_dims)
            return [reduce_node]
        result = []
        for idx, axes in enumerate(splited_axes):
            outputs = [y] if idx == len(splited_axes) - 1 else None
            reduce_node, x = self._new_node(node_name, op_type, [x], outputs=outputs, axes=axes, keepdims=keep_dims)
            result.append(reduce_node)
        return result

    def _decompose_reduce_precision(self, node: NodeProto, graph: GraphProto, **kwargs):
        x = node.input[0]
        dtype, shape = self._get_dtype_and_shape(x, **kwargs)
        if not _is_half_dtype(dtype):
            return [node]
        node_name = node.name
        rank = len(shape)
        keep_dims, axes = get_reduce_info(node, graph, rank)
        y = node.output[0]
        cast_node, x = self._new_node(node_name, "Cast", [x], to=TensorProto.FLOAT)
        reduce_node, reduce_out = self._new_node(node_name, node.op_type, [x], axes=axes, keepdims=keep_dims)
        cast_node1, _ = self._new_node(node_name, "Cast", [reduce_out], outputs=[y], to=dtype)
        return [cast_node, reduce_node, cast_node1]

    def ReduceMax(self, node: NodeProto, graph: GraphProto, **kwargs):  # noqa: N802
        return self._decompose_reduce_axes(node, graph, **kwargs)

    def ReduceMin(self, node: NodeProto, graph: GraphProto, **kwargs):  # noqa: N802
        return self._decompose_reduce_axes(node, graph, **kwargs)

    def ReduceSum(self, node: NodeProto, graph: GraphProto, **kwargs):  # noqa: N802
        precision_decompose_result = self._decompose_reduce_precision(node, graph, **kwargs)
        # The decompose process will be called recursively, if it's already a decomposed result, just return.
        if len(precision_decompose_result) != 1 or precision_decompose_result[0] != node:
            return precision_decompose_result
        return self._decompose_reduce_axes(node, graph, **kwargs)

    def ReduceMean(self, node: NodeProto, graph: GraphProto, **kwargs):  # noqa: N802
        precision_decompose_result = self._decompose_reduce_precision(node, graph, **kwargs)
        # The decompose process will be called recursively, if it's already a decomposed result, just return.
        if len(precision_decompose_result) != 1 or precision_decompose_result[0] != node:
            return precision_decompose_result
        axes_decompose_result = self._decompose_reduce_axes(node, graph, **kwargs)
        # The decompose process will be called recursively, if it's already a decomposed result, just return.
        if len(axes_decompose_result) != 1 or axes_decompose_result[0] != node:
            return axes_decompose_result
        node_name = node.name
        x = node.input[0]
        y = node.output[0]
        dtype, shape = self._get_dtype_and_shape(x, **kwargs)
        rank = len(shape)
        keep_dims, axes = get_reduce_info(node, graph, rank)
        sum_node, sum_out = self._new_node(node_name, "ReduceSum", [x], axes=axes, keepdims=keep_dims)
        # If it's not concrete shape, we need add more Ops such as Shape, Gather to get the dim value,
        # which is not supported yet.
        assert all(shape[axis].is_number for axis in axes)
        denominator = int(sympy.prod([shape[axis] for axis in axes]))
        denominator_tensor = helper.make_tensor(
            name=f"{node_name}_denominator",
            dims=(),
            data_type=dtype,
            vals=np.array([denominator], dtype=to_numpy_type(dtype)),
        )
        denominator_node, denominator_out = self._new_node(node_name, "Constant", [], value=denominator_tensor)
        div_node, _ = self._new_node(node_name, "Div", [sum_out, denominator_out], outputs=[y])
        return [sum_node, denominator_node, div_node]
