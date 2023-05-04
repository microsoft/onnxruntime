# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

# Decompose a complicated op into a series of simple ops.

from typing import List

import numpy as np
import sympy
from onnx import GraphProto, NodeProto, TensorProto, helper

from ._utils import get_attribute, get_reduce_info


class DecomposeDispatch(object):
    """
    A node does only responsible for a single computation or a type of triton ops.
    For those compound Onnx nodes, like softmax/layernorm/groupnorm, etc., we need to decompose them into a series of
    simple ops.
    """

    def __init__(self):
        super().__init__()
        self.count = 0

    def _get_unique_var_name(self, prefix):
        self.count += 1
        return prefix + str(self.count)

    def _new_node(self, node_name, op_type, inputs, outputs=None, **kwargs):
        name = self._get_unique_var_name(f"{node_name}_{op_type}_")
        if outputs is None:
            outputs = [f"{name}_out"]
        return outputs[0], helper.make_node(op_type, inputs, outputs, name, **kwargs)

    def __call__(self, node: NodeProto, graph: GraphProto, **kwargs) -> List[NodeProto]:
        op_type = node.op_type
        if not hasattr(self, op_type):
            raise NotImplementedError("Not implemented for op type: {}".format(op_type))
        return getattr(self, op_type)(node, graph, **kwargs)

    def __contains__(self, node: NodeProto) -> bool:
        return hasattr(self, node.op_type)

    def _get_dtype_and_shape(self, arg_name: str, **kwargs):
        node_arg_infos = kwargs["node_arg_infos"]
        arg_info = node_arg_infos[arg_name]
        return arg_info.dtype, arg_info.shape

    def _filter_none_nodes(self, nodes):
        return [node for node in nodes if node is not None]

    def _elementwise_compute_on_float32(self, node: NodeProto, graph: GraphProto, **kwargs):
        x = node.input[0]
        dtype, _ = self._get_dtype_and_shape(x, **kwargs)
        is_half = dtype == TensorProto.FLOAT16 or dtype == TensorProto.BFLOAT16
        if not is_half:
            return [node]
        node_name = node.name
        y = node.output[0]
        op_type = node.op_type
        inputs = [input for input in node.input]
        cast_nodes = []
        for idx, input in enumerate(inputs):
            dtype, _ = self._get_dtype_and_shape(input, **kwargs)
            if dtype == TensorProto.FLOAT16 or dtype == TensorProto.BFLOAT16:
                cast_out, cast_node = self._new_node(node_name, "Cast", [input], to=TensorProto.FLOAT)
                inputs[idx] = cast_out
                cast_nodes.append(cast_node)
        op_out, op_node = self._new_node(node_name, op_type, inputs)
        _, cast_node1 = self._new_node(node_name, "Cast", [op_out], outputs=[y], to=dtype)
        return self._filter_none_nodes(cast_nodes + [op_node, cast_node1])

    def Exp(self, node: NodeProto, graph: GraphProto, **kwargs):
        return self._elementwise_compute_on_float32(node, graph, **kwargs)

    def Pow(self, node: NodeProto, graph: GraphProto, **kwargs):
        return self._elementwise_compute_on_float32(node, graph, **kwargs)

    def LayerNormalization(self, node: NodeProto, graph: GraphProto, **kwargs):
        node_name = node.name
        x = node.input[0]
        w = node.input[1]
        b = node.input[2]
        y = node.output[0]
        dtype, shape = self._get_dtype_and_shape(x, **kwargs)
        rank = len(shape)
        is_half = dtype == TensorProto.FLOAT16 or dtype == TensorProto.BFLOAT16
        mean_outputs = [node.output[1]] if len(node.output) > 1 else None
        inv_std_dev_outputs = [node.output[2]] if len(node.output) > 2 else None

        axis = get_attribute(node, "axis", -1)
        if axis < 0:
            axis += rank
        axes = list(range(axis, rank))
        epsilon = get_attribute(node, "epsilon", 1e-05)

        epsilon_tensor = helper.make_tensor(
            name="epsilon_const",
            data_type=TensorProto.FLOAT if is_half else dtype,
            dims=(1,),
            vals=np.array([get_attribute(node, "epsilon", epsilon)]),
        )

        const_out, const_node = self._new_node(node_name, "Constant", [], value=epsilon_tensor)
        cast_node = None
        if is_half:
            x, cast_node = self._new_node(node_name, "Cast", [x], to=TensorProto.FLOAT)
        reducemean_out, reducemean_node = self._new_node(node_name, "ReduceMean", [x], outputs=mean_outputs, axes=axes)
        sub_out, sub_node = self._new_node(node_name, "Sub", [x, reducemean_out])
        mul_out, mul_node = self._new_node(node_name, "Mul", [sub_out, sub_out])
        reducemean_out1, reducemean_node1 = self._new_node(node_name, "ReduceMean", [mul_out], axes=axes)
        add_out, add_node = self._new_node(node_name, "Add", [reducemean_out1, const_out])
        rsqrt_out, rsqrt_node = self._new_node(node_name, "Rsqrt", [add_out], outputs=inv_std_dev_outputs)
        mul_out1, mul_node1 = self._new_node(node_name, "Mul", [sub_out, rsqrt_out])
        cast_node1 = None
        if is_half:
            mul_out1, cast_node1 = self._new_node(node_name, "Cast", [mul_out1], to=dtype)
        mul_out2, mul_node2 = self._new_node(node_name, "Mul", [w, mul_out1])
        _, add_node1 = self._new_node(node_name, "Add", [b, mul_out2], outputs=[y])

        return self._filter_none_nodes(
            [
                const_node,
                cast_node,
                reducemean_node,
                sub_node,
                mul_node,
                reducemean_node1,
                add_node,
                rsqrt_node,
                mul_node1,
                cast_node1,
                mul_node2,
                add_node1,
            ]
        )

    def LayerNormalizationGrad(self, node: NodeProto, graph: GraphProto, **kwargs):
        node_name = node.name
        dy = node.input[0]
        x = node.input[1]
        w = node.input[2]
        mean = node.input[3]
        inv_std_dev = node.input[4]
        dx = node.output[0]
        dw = node.output[1] if len(node.output) > 1 else ""
        db = node.output[2] if len(node.output) > 2 else ""
        dtype, shape = self._get_dtype_and_shape(dy, **kwargs)
        rank = len(shape)
        is_half = dtype == TensorProto.FLOAT16 or dtype == TensorProto.BFLOAT16

        axis = get_attribute(node, "axis", -1)
        if axis < 0:
            axis += rank
        axes = list(range(axis, rank))

        cast_node = None
        cast_node1 = None
        cast_node2 = None
        if is_half:
            dy, cast_node = self._new_node(node_name, "Cast", [dy], to=TensorProto.FLOAT)
            x, cast_node1 = self._new_node(node_name, "Cast", [x], to=TensorProto.FLOAT)
            w, cast_node2 = self._new_node(node_name, "Cast", [w], to=TensorProto.FLOAT)
        sub_out, sub_node = self._new_node(node_name, "Sub", [x, mean])
        mul_out, mul_node = self._new_node(node_name, "Mul", [sub_out, inv_std_dev])
        mul_out1, mul_node1 = self._new_node(node_name, "Mul", [w, dy])
        mul_out2, mul_node2 = self._new_node(node_name, "Mul", [mul_out, mul_out1])
        reducemean_out, reducemean_node = self._new_node(node_name, "ReduceMean", [mul_out2], axes=axes)
        reducemean_out1, reducemean_node1 = self._new_node(node_name, "ReduceMean", [mul_out1], axes=axes)
        mul_out3, mul_node3 = self._new_node(node_name, "Mul", [reducemean_out, mul_out])
        add_out, add_node = self._new_node(node_name, "Add", [mul_out3, reducemean_out1])
        sub_out1, sub_node1 = self._new_node(node_name, "Sub", [mul_out1, add_out])
        cast_node3 = None
        if is_half:
            mul_out4, mul_node4 = self._new_node(node_name, "Mul", [sub_out1, inv_std_dev])
            _, cast_node3 = self._new_node(node_name, "Cast", [mul_out4], outputs=[dx], to=dtype)
        else:
            _, mul_node4 = self._new_node(node_name, "Mul", [sub_out1, inv_std_dev], outputs=[dx])

        mul_node5 = None
        reducesum_node = None
        cast_node4 = None
        dw_axes = list(range(axis))
        if dw != "":
            mul_out5, mul_node5 = self._new_node(node_name, "Mul", [dy, mul_out])
            if is_half:
                mul_out5, cast_node4 = self._new_node(node_name, "Cast", [mul_out5], to=dtype)
            _, reducesum_node = self._new_node(
                node_name, "ReduceSum", [mul_out5], outputs=[dw], axes=dw_axes, keepdims=0
            )

        reducesum_node1 = None
        if db != "":
            _, reducesum_node1 = self._new_node(node_name, "ReduceSum", [dy], outputs=[db], axes=dw_axes, keepdims=0)

        return self._filter_none_nodes(
            [
                cast_node,
                cast_node1,
                cast_node2,
                sub_node,
                mul_node,
                mul_node1,
                mul_node2,
                reducemean_node,
                reducemean_node1,
                mul_node3,
                add_node,
                sub_node1,
                cast_node3,
                mul_node4,
                mul_node5,
                cast_node4,
                reducesum_node,
                reducesum_node1,
            ]
        )

    def Softmax(self, node: NodeProto, graph: GraphProto, **kwargs):
        node_name = node.name
        x = node.input[0]
        y = node.output[0]
        dtype, _ = self._get_dtype_and_shape(x, **kwargs)
        is_half = dtype == TensorProto.FLOAT16 or dtype == TensorProto.BFLOAT16
        axis = get_attribute(node, "axis", -1)

        max_out, max_node = self._new_node(node_name, "ReduceMax", [x], axes=[axis])
        sub_out, sub_node = self._new_node(node_name, "Sub", [x, max_out])
        cast_node = None
        if is_half:
            sub_out, cast_node = self._new_node(node_name, "Cast", [sub_out], to=TensorProto.FLOAT)
        exp_out, exp_node = self._new_node(node_name, "Exp", [sub_out])
        sum_out, sum_node = self._new_node(node_name, "ReduceSum", [exp_out], axes=[axis])
        cast_node1 = None
        if is_half:
            div_out, div_node = self._new_node(node_name, "Div", [exp_out, sum_out])
            _, cast_node1 = self._new_node(node_name, "Cast", [div_out], outputs=[y], to=dtype)
        else:
            _, div_node = self._new_node(node_name, "Div", [exp_out, sum_out], outputs=[y])

        return self._filter_none_nodes([max_node, sub_node, cast_node, exp_node, sum_node, div_node, cast_node1])

    def SoftmaxGrad_13(self, node: NodeProto, graph: GraphProto, **kwargs):
        node_name = node.name
        dy = node.input[0]
        y = node.input[1]
        dx = node.output[0]
        dtype, _ = self._get_dtype_and_shape(dy, **kwargs)
        is_half = dtype == TensorProto.FLOAT16 or dtype == TensorProto.BFLOAT16
        axis = get_attribute(node, "axis", -1)

        cast_node = None
        cast_node1 = None
        if is_half:
            dy, cast_node = self._new_node(node_name, "Cast", [dy], to=TensorProto.FLOAT)
            y, cast_node1 = self._new_node(node_name, "Cast", [y], to=TensorProto.FLOAT)
        mul_out, mul_node = self._new_node(node_name, "Mul", [dy, y])
        sum_out, sum_node = self._new_node(node_name, "ReduceSum", [mul_out], axes=[axis])
        mul_out1, mul_node1 = self._new_node(node_name, "Mul", [y, sum_out])
        cast_node2 = None
        if is_half:
            sub_out, sub_node = self._new_node(node_name, "Sub", [mul_out, mul_out1])
            _, cast_node2 = self._new_node(node_name, "Cast", [sub_out], outputs=[dx], to=dtype)
        else:
            _, sub_node = self._new_node(node_name, "Sub", [mul_out, mul_out1], outputs=[dx])

        return self._filter_none_nodes([cast_node, cast_node1, mul_node, sum_node, mul_node1, sub_node, cast_node2])

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
            _, identity_node = self._new_node(node_name, "Identity", [x], outputs=[y])
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
            _, reduce_node = self._new_node(node_name, op_type, [x], outputs=[y], axes=axes, keepdims=keep_dims)
            return [reduce_node]
        result = []
        for idx, axes in enumerate(splited_axes):
            outputs = [y] if idx == len(splited_axes) - 1 else None
            x, reduce_node = self._new_node(node_name, op_type, [x], outputs=outputs, axes=axes, keepdims=keep_dims)
            result.append(reduce_node)
        return result

    def ReduceMax(self, node: NodeProto, graph: GraphProto, **kwargs):
        return self._decompose_reduce_axes(node, graph, **kwargs)

    def ReduceMin(self, node: NodeProto, graph: GraphProto, **kwargs):
        return self._decompose_reduce_axes(node, graph, **kwargs)

    def ReduceSum(self, node: NodeProto, graph: GraphProto, **kwargs):
        return self._decompose_reduce_axes(node, graph, **kwargs)

    def ReduceMean(self, node: NodeProto, graph: GraphProto, **kwargs):
        axes_decompose_result = self._decompose_reduce_axes(node, graph, **kwargs)
        # The decompose process will be called recursively, if it's already a decomposed result, just return.
        if len(axes_decompose_result) != 1 or axes_decompose_result[0] != node:
            return axes_decompose_result
        node_name = node.name
        x = node.input[0]
        y = node.output[0]
        dtype, shape = self._get_dtype_and_shape(x, **kwargs)
        rank = len(shape)
        is_half = dtype == TensorProto.FLOAT16 or dtype == TensorProto.BFLOAT16
        keep_dims, axes = get_reduce_info(node, graph, rank)
        cast_node = None
        if is_half:
            x, cast_node = self._new_node(node_name, "Cast", [x], to=TensorProto.FLOAT)
        sum_out, sum_node = self._new_node(node_name, "ReduceSum", [x], axes=axes, keepdims=keep_dims)
        # If it's not concrete shape, we need add more Ops such as Shape, Gather to get the dim value,
        # which is not supported yet.
        assert all(shape[axis].is_number for axis in axes)
        denominator = int(sympy.prod([shape[axis] for axis in axes]))
        denominator_tensor = helper.make_tensor(
            name=f"{node_name}_denominator",
            dims=(),
            data_type=TensorProto.FLOAT,
            vals=np.array([denominator], dtype=np.float32),
        )
        denominator_out, denominator_node = self._new_node(node_name, "Constant", [], value=denominator_tensor)
        cast_node1 = None
        if is_half:
            div_out, div_node = self._new_node(node_name, "Div", [sum_out, denominator_out])
            _, cast_node1 = self._new_node(node_name, "Cast", [div_out], outputs=[y], to=dtype)
        else:
            _, div_node = self._new_node(node_name, "Div", [sum_out, denominator_out], outputs=[y])

        return self._filter_none_nodes([cast_node, sum_node, denominator_node, div_node, cast_node1])
