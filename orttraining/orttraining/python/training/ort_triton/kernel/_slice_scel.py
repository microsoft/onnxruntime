# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import itertools
import sys

import torch
import triton
import triton.language as tl
from onnx import TensorProto, helper

from onnxruntime.training.ortmodule import register_graph_optimizer

from .._utils import get_attribute, to_numpy_array


@triton.jit
def _triton_slice_log_softmax(log_prob, logit, d: tl.constexpr, c: tl.constexpr, RBLOCK: tl.constexpr):
    xoffset = tl.program_id(0)
    logit_xoffset = (xoffset // d * (d + 1) + xoffset % d) * c
    rbase = tl.arange(0, RBLOCK)
    logit_max_row = tl.zeros([RBLOCK], tl.float32) + float("-inf")
    for roffset in range(0, c, RBLOCK):
        rindex = rbase + roffset
        rmask = rindex < c
        logit_row = tl.load(logit + logit_xoffset + rindex, mask=rmask, other=0.0).to(tl.float32)
        logit_max_row = tl.where(rmask & (logit_max_row < logit_row), logit_row, logit_max_row)
    logit_max_reduced = tl.max(logit_max_row, axis=0)
    exp_sum_row = tl.zeros([RBLOCK], tl.float32)
    for roffset in range(0, c, RBLOCK):
        rindex = rbase + roffset
        rmask = rindex < c
        logit_row = tl.load(logit + logit_xoffset + rindex, mask=rmask, other=0.0).to(tl.float32)
        exp_sum_row = tl.where(rmask, exp_sum_row + tl.exp(logit_row - logit_max_reduced), exp_sum_row)
    reduced_log_sum = tl.log(tl.sum(exp_sum_row, axis=0)) + logit_max_reduced
    for roffset in range(0, c, RBLOCK):
        rindex = rbase + roffset
        rmask = rindex < c
        logit_row = tl.load(logit + logit_xoffset + rindex, mask=rmask, other=0.0).to(tl.float32)
        output_row = logit_row - reduced_log_sum
        tl.store(log_prob + xoffset * c + rindex, output_row, mask=rmask)


@triton.jit
def _triton_slice_scel(
    loss,
    factor,
    log_prob,
    label,
    ignore_index,
    d: tl.constexpr,
    c: tl.constexpr,
    n_cols: tl.constexpr,
    RBLOCK: tl.constexpr,
):
    rbase = tl.arange(0, RBLOCK)
    neg_sum_row = tl.zeros([RBLOCK], tl.float32)
    factor_row = tl.zeros([RBLOCK], tl.float32)
    for roffset in range(0, n_cols, RBLOCK):
        rindex = rbase + roffset
        rmask = rindex < n_cols
        label_row = tl.load(label + (rindex // d) * (d + 1) + rindex % d + 1, mask=rmask, other=0.0).to(tl.int32)
        mask = rmask & (label_row != ignore_index)
        log_prob_row = tl.load(log_prob + rindex * c + label_row, mask=mask, other=0.0)
        neg_sum_row = tl.where(mask, neg_sum_row - log_prob_row, neg_sum_row)
        factor_row = tl.where(mask, factor_row + 1.0, factor_row)
    reduced_neg_sum = tl.sum(neg_sum_row, axis=0)
    reduced_factor = tl.sum(factor_row, axis=0)
    loss_value = reduced_neg_sum / reduced_factor
    tl.store(loss, loss_value)
    tl.store(factor, reduced_factor)


def slice_scel(logit, label, ignore_index):
    ignore_index_value = ignore_index.item()
    c = logit.shape[-1]
    logit_d = logit.shape[-2]
    d = logit_d - 1
    n = logit.numel() // (logit_d * c)
    log_prob_shape = list(logit.shape)[:-2] + [d, c]
    log_prob = torch.empty(log_prob_shape, dtype=torch.float, device=logit.device)
    rblock = 4096 if c > 4096 else triton.next_power_of_2(c)
    num_warps = 16 if rblock >= 4096 else (8 if rblock >= 2048 else 4)
    _triton_slice_log_softmax[(n * d,)](log_prob, logit, d, c, num_warps=num_warps, RBLOCK=rblock)
    loss = torch.empty([], dtype=logit.dtype, device=logit.device)
    factor = torch.empty([], dtype=torch.float, device=logit.device)
    n_cols = n * d
    rblock = 1024 if n_cols > 1024 else triton.next_power_of_2(n_cols)
    _triton_slice_scel[(1,)](loss, factor, log_prob, label, ignore_index_value, d, c, n_cols, RBLOCK=rblock)
    return loss, log_prob, factor


@triton.jit
def _triton_slice_scel_backward(
    dlogit,
    dloss,
    log_prob,
    label,
    factor,
    d: tl.constexpr,
    c: tl.constexpr,
    n_elements: tl.constexpr,
    XBLOCK: tl.constexpr,
):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)
    xmask = xindex < n_elements
    nd_index = xindex // c
    dlogit_nd_index = (nd_index // d) * (d + 1) + nd_index % d
    label_nd_index = dlogit_nd_index + 1
    c_index = xindex % c
    dloss_value = tl.load(dloss).to(tl.float32)
    log_prob_row = tl.load(log_prob + xindex, mask=xmask, other=0.0)
    label_row = tl.load(label + label_nd_index, mask=xmask, other=0.0).to(tl.int32)
    factor_value = tl.load(factor)
    dlogit_row = dloss_value * (tl.exp(log_prob_row) - tl.where(c_index == label_row, 1.0, 0.0)) / factor_value
    tl.store(dlogit + dlogit_nd_index * c + c_index, dlogit_row, mask=xmask)


@triton.jit
def _triton_slice_scel_bias_backward(
    dlogit,
    dloss,
    log_prob,
    label,
    factor,
    bias,
    dlogit_d: tl.constexpr,
    c: tl.constexpr,
    n_elements: tl.constexpr,
    XBLOCK: tl.constexpr,
):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)
    xmask = xindex < n_elements
    dlogit_nd_index = xindex // c
    dlogit_n_index = dlogit_nd_index // dlogit_d
    dlogit_d_index = dlogit_nd_index % dlogit_d
    nd_index = dlogit_n_index * (dlogit_d - 1) + dlogit_d_index
    nd_mask = xmask & (dlogit_d_index != dlogit_d - 1)
    c_index = xindex % c
    dloss_value = tl.load(dloss).to(tl.float32)
    log_prob_row = tl.load(log_prob + nd_index * c + c_index, mask=nd_mask, other=0.0)
    label_row = tl.load(label + dlogit_nd_index + 1, mask=nd_mask, other=0.0).to(tl.int32)
    factor_value = tl.load(factor)
    bias_row = tl.load(bias + xindex, mask=xmask, other=0.0).to(tl.float32)
    dlogit_row = dloss_value * (tl.exp(log_prob_row) - tl.where(c_index == label_row, 1.0, 0.0)) / factor_value
    dlogit_row = tl.where(nd_mask, dlogit_row, 0.0) + bias_row
    tl.store(dlogit + xindex, dlogit_row, mask=xmask)


def slice_scel_backward(dloss, log_prob, label, factor, bias):
    c = log_prob.shape[-1]
    d = log_prob.shape[-2]
    dlogit_d = d + 1
    dlogit_shape = list(log_prob.shape)[:-2] + [dlogit_d, c]
    dlogit = (
        torch.empty(dlogit_shape, dtype=dloss.dtype, device=dloss.device)
        if bias is not None
        else torch.zeros(dlogit_shape, dtype=dloss.dtype, device=dloss.device)
    )
    n_elements = dlogit.numel() if bias is not None else log_prob.numel()
    xblock = 1024 if n_elements > 1024 else triton.next_power_of_2(n_elements)

    def grid(meta):
        return (triton.cdiv(n_elements, meta["XBLOCK"]),)

    if bias is not None:
        _triton_slice_scel_bias_backward[grid](
            dlogit, dloss, log_prob, label, factor, bias, dlogit_d, c, n_elements, XBLOCK=xblock
        )
    else:
        _triton_slice_scel_backward[grid](dlogit, dloss, log_prob, label, factor, d, c, n_elements, XBLOCK=xblock)
    return dlogit


def _get_producer(graph, arg, op_type):
    for node in graph.node:
        if node.op_type == op_type:
            for output in node.output:
                if output == arg:
                    return node
    return None


def _get_consumer(graph, arg, op_type):
    for node in graph.node:
        if node.op_type == op_type:
            for input in node.input:
                if input == arg:
                    return node
    return None


def _get_arg_info(graph, arg):
    value_info = None
    for info in itertools.chain(graph.input, graph.output, graph.value_info):
        if info.name == arg:
            value_info = info
    if value_info is None or value_info.type is None:
        return None, None
    tensor_type = value_info.type.tensor_type
    return tensor_type.elem_type, tensor_type.shape


def _get_constant(graph, arg):
    initializer = None
    for init in graph.initializer:
        if init.name == arg:
            initializer = init
    if initializer is None:
        return None
    return to_numpy_array(initializer)


def _check_slice(graph, node, start, end, axis, step):
    _, shape = _get_arg_info(graph, node.input[0])
    if shape is None:
        return False
    rank = len(shape.dim)
    if axis < 0:
        axis += rank
    for idx, value in enumerate([start, end, axis, step]):
        constant = _get_constant(graph, node.input[idx + 1])
        if constant is None or constant.size != 1:
            return False
        constant_value = constant.item()
        if idx == 2 and constant_value < 0:
            constant_value += rank
        if constant_value != value:
            return False
    return True


def _get_shape_related_nodes(graph, start_arg, sub_graph_nodes):
    args = [start_arg]
    while len(args) > 0:
        arg = args.pop(0)
        for node in graph.node:
            if arg in node.input and node not in sub_graph_nodes:
                sub_graph_nodes.append(node)
                for output in node.output:
                    if output not in args:
                        args.append(output)


@register_graph_optimizer(devices="cuda")
def optimize_graph_for_slice_scel(graph):
    remove_nodes = []
    triton_nodes = []
    value_infos = []
    idx = 0
    for node in graph.node:
        if node.op_type != "SoftmaxCrossEntropyLossInternal":
            continue
        # Weight input not supported for now, support reduction=mean only for now.
        # It's required that the output_type is same as the logit dtype because currently we cannot pass the attribute
        # value from TritonOp. We can add support of this in the future.
        if len(node.input) > 2 and node.input[2]:
            continue
        reduction_attr = get_attribute(node, "reduction", "mean")
        if isinstance(reduction_attr, bytes):
            reduction_attr = reduction_attr.decode()
        if reduction_attr != "mean":
            continue
        elem_type, _ = _get_arg_info(graph, node.input[0])
        output_type_attr = get_attribute(node, "output_type", elem_type)
        if output_type_attr != elem_type:
            continue
        reshape0 = _get_producer(graph, node.input[0], "Reshape")
        if reshape0 is None:
            continue
        _, shape0 = _get_arg_info(graph, reshape0.input[0])
        _, shape1 = _get_arg_info(graph, reshape0.output[0])
        if shape0 is None or shape1 is None or shape0.dim[-1] != shape1.dim[-1]:
            continue
        slice0 = _get_producer(graph, reshape0.input[0], "Slice")
        if slice0 is None or not _check_slice(graph, slice0, 0, -1, -2, 1):
            continue
        reshape1 = _get_producer(graph, node.input[1], "Reshape")
        if reshape1 is None:
            continue
        slice1 = _get_producer(graph, reshape1.input[0], "Slice")
        if slice1 is None or not _check_slice(graph, slice1, 1, sys.maxsize, -1, 1):
            continue
        scel_grad = _get_consumer(graph, node.output[1], "SoftmaxCrossEntropyLossInternalGrad")
        if scel_grad is None:
            continue
        reshape2 = _get_consumer(graph, scel_grad.output[0], "Reshape")
        if reshape2 is None:
            continue
        slice_grad = _get_consumer(graph, reshape2.output[0], "SliceGrad")
        if slice_grad is None:
            continue
        shape_node = _get_producer(graph, slice_grad.input[1], "Shape")
        if shape_node is None:
            continue
        bias_arg = ""
        sum_node = _get_consumer(graph, slice_grad.output[0], "Sum")
        if sum_node is not None and len(sum_node.input) == 2:
            _, shape0 = _get_arg_info(graph, sum_node.input[0])
            _, shape1 = _get_arg_info(graph, sum_node.input[1])
            if shape0 is not None and shape0 == shape1:
                bias_arg = sum_node.input[0] if sum_node.input[1] == slice_grad.output[0] else sum_node.input[1]
        sub_graph_nodes = [node, reshape0, slice0, reshape1, slice1, scel_grad, reshape2, slice_grad, shape_node]
        _get_shape_related_nodes(graph, slice0.output[0], sub_graph_nodes)
        if bias_arg:
            sub_graph_nodes.append(sum_node)
        remove_nodes.extend(sub_graph_nodes)
        forward_inputs = [slice0.input[0], slice1.input[0]]
        if len(node.input) > 3:
            forward_inputs.append(node.input[3])
        else:
            ignore_index_arg = "ignore_index_" + str(idx)
            ignore_index_node = helper.make_node(
                "Constant",
                inputs=[],
                outputs=[ignore_index_arg],
                value=helper.make_tensor(
                    name=ignore_index_arg,
                    data_type=TensorProto.INT64,
                    dims=(),
                    vals=[-100],
                ),
            )
            forward_inputs.append(ignore_index_arg)
            triton_nodes.append(ignore_index_node)
        # Use float for log_prob, which has better precision, but may consume more memory.
        log_prob_arg = helper.make_tensor_value_info("scel_log_prob_" + str(idx), TensorProto.FLOAT, None)
        factor_arg = helper.make_tensor_value_info("scel_factor_" + str(idx), TensorProto.FLOAT, None)
        value_infos.extend([log_prob_arg, factor_arg])
        triton_fw_node = helper.make_node(
            "TritonOp",
            forward_inputs,
            [node.output[0], log_prob_arg.name, factor_arg.name],
            "TritonOp_Slice_SCEL_" + str(idx),
            None,
            "com.microsoft",
            func_name="slice_scel",
        )
        triton_nodes.append(triton_fw_node)
        backward_outputs = [sum_node.output[0] if bias_arg else slice_grad.output[0]]
        triton_bw_node = helper.make_node(
            "TritonOp",
            [scel_grad.input[0], log_prob_arg.name, slice1.input[0], factor_arg.name, bias_arg],
            backward_outputs,
            "TritonOp_Slice_SCEL_Backward_" + str(idx),
            None,
            "com.microsoft",
            func_name="slice_scel_backward",
        )
        triton_nodes.append(triton_bw_node)
        idx += 1

    all_nodes = []
    for node in graph.node:
        if node not in remove_nodes:
            all_nodes.append(node)

    for node in triton_nodes:
        all_nodes.append(node)  # noqa: PERF402

    graph.ClearField("node")
    graph.node.extend(all_nodes)
    graph.value_info.extend(value_infos)
