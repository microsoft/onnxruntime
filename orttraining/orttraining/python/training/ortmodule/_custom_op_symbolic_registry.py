# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from torch.onnx import register_custom_op_symbolic
from torch.onnx.symbolic_helper import parse_args, _get_tensor_dim_size, _get_tensor_sizes
import torch.onnx.symbolic_helper as sym_help
import torch


class CustomOpSymbolicRegistry:
    _SYMBOLICS = {}

    @classmethod
    def register(cls, name, domain, fn):
        cls._SYMBOLICS[domain + '::' + name] = fn

    @classmethod
    def register_all(cls):
        for name, fn in cls._SYMBOLICS.items():
            # Symbolic name is in format: domain::name
            register_custom_op_symbolic(name, fn, 1)


def register_symbolic(name, domain=''):
    def symbolic_wrapper(fn):
        CustomOpSymbolicRegistry.register(name, domain, fn)
        return fn
    return symbolic_wrapper


@register_symbolic('cross_entropy_loss')
@parse_args('v', 'v', 'v', 'i', 'v', 'v')
def cross_entropy_loss(g, self, target, weight, reduction, ignore_index, label_smoothing=0.0):
    label_smoothing = sym_help._maybe_get_const(label_smoothing, "f")
    if label_smoothing > 0.0:
        raise RuntimeError("Unsupported: ONNX does not support label_smoothing")

    # reduction: 0->none, 1->mean, 2->sum
    reduction = sym_help._maybe_get_const(reduction, 'i')
    reduction_vals = ['none', 'mean', 'sum']
    reduction = reduction_vals[reduction]
    output, log_prob = g.op("com.microsoft::SoftmaxCrossEntropyLossInternal",
                            self, target, weight, ignore_index,
                            reduction_s=reduction, outputs=2)
    output.setType(self.type())
    log_prob.setType(self.type())
    return output


@register_symbolic('nll_loss')
@parse_args('v', 'v', 'v', 'i', 'v')
def nll_loss(g, self, target, weight, reduction, ignore_index):
    # reduction: 0->none, 1->mean, 2->sum
    reduction = sym_help._maybe_get_const(reduction, 'i')
    reduction_vals = ['none', 'mean', 'sum']
    reduction = reduction_vals[reduction]
    output = g.op("com.microsoft::NegativeLogLikelihoodLossInternal",
                    self, target, weight, ignore_index, reduction_s=reduction)
    output.setType(self.type())
    return output


@register_symbolic('embedding')
def embedding(g, weight, indices, padding_idx, scale_grad_by_freq, sparse):
    output = g.op("com.microsoft::ATenOp", weight, indices, padding_idx, scale_grad_by_freq, sparse,
                  name_s='aten::embedding')
    indices_shape = _get_tensor_sizes(indices)
    if indices_shape is not None and hasattr(weight.type(), 'with_sizes'):
        output_type = weight.type().with_sizes(
            indices_shape + [_get_tensor_dim_size(weight, 1)])
        output.setType(output_type)
    return output


@register_symbolic('diagonal')
def diagonal(g, self, offset, dim1, dim2):
    return g.op("com.microsoft::ATenOp", self, offset, dim1, dim2,
                name_s='aten::diagonal')


@register_symbolic('multinomial')
def multinomial(g, self, num_samples, replacement=False, generator=None):
    if generator is not None and not sym_help._is_none(generator):
        raise RuntimeError("Unsupported: ONNX does not support generator for multinomial")
    return g.op("com.microsoft::ATenOp", self, num_samples, replacement, generator,
                name_s='aten::multinomial')


@register_symbolic('max_pool2d')
def max_pool2d(g, self, kernel_size, stride, padding, dilation, ceil_mode):
    stride_val = sym_help._maybe_get_const(stride, 'is')
    if not stride_val:
        stride = kernel_size
    return g.op("com.microsoft::ATenOp", self, kernel_size, stride, padding, dilation, ceil_mode,
                name_s='aten::max_pool2d_with_indices', outputs=2)[0]


@register_symbolic('unfold')
def unfold(g, input, dimension, size, step):
    return g.op("com.microsoft::ATenOp", input, dimension, size, step, name_s='aten::unfold')


@register_symbolic('argmax')
def argmax(g, input, dim, keepdim):
    return g.op("com.microsoft::ATenOp", input, dim, keepdim, name_s='aten::argmax')


@register_symbolic('avg_pool2d')
def avg_pool2d(g, self, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override):
    stride_val = sym_help._maybe_get_const(stride, 'is')
    if not stride_val:
        stride = kernel_size
    return g.op("com.microsoft::ATenOp", self, kernel_size, stride, padding, ceil_mode,
                count_include_pad, divisor_override, name_s='aten::avg_pool2d')


@register_symbolic('adaptive_avg_pool2d')
def adaptive_avg_pool2d(g, self, output_size):
    return g.op("com.microsoft::ATenOp", self, output_size, name_s='aten::_adaptive_avg_pool2d')


# For torch.einsum.
def parse_equation(equation):
    pos_comma = equation.find(',')
    pos_arrow = equation.find('->')
    assert pos_comma != -1 and pos_arrow > pos_comma
    lhs_labels = [label for label in equation[:pos_comma] if label != ' ']
    rhs_labels = [label for label in equation[pos_comma + 1:pos_arrow] if label != ' ']
    result_labels = [label for label in equation[pos_arrow + 2:] if label != ' ']
    # Two operands and result are not empty, and are all alpha characters.
    assert lhs_labels and rhs_labels and result_labels
    assert all(label.isalpha() for label in lhs_labels + rhs_labels + result_labels)
    # Output has no repeated label, each label must be in at least one operand.
    assert len(result_labels) == len(set(result_labels))
    assert all(label in lhs_labels or label in rhs_labels for label in result_labels)
    return lhs_labels, rhs_labels, result_labels

def need_permute(perm):
    return any(idx != axis for idx, axis in enumerate(perm))

def map_labels_to_output(input_labels, label_perm_map):
    output_len = len(label_perm_map)
    perm = [-1] * output_len
    unsqueeze_axes = []
    idx = 0
    for label in input_labels:
        # Lookup output index for label.
        perm[label_perm_map[label]] = idx
        idx += 1

    # Add dimensions for missing labels.
    for i in range(output_len):
        if perm[i] == -1:
            unsqueeze_axes.append(idx)
            perm[i] = idx
            idx += 1

    return perm, unsqueeze_axes

def unsqueeze_and_permute_for_mul(g, tensor, unsqueeze_axes, perm):
    # If perm is sorted after removing unsqueeze axes, then permute is not needed.
    # For example, a.unsqueeze(2).permute([0, 2, 1]) is same as a.unsqueeze(1).
    if unsqueeze_axes:
        new_perm = [v for v in perm if v not in unsqueeze_axes]
        sorted = all(new_perm[i] < new_perm[i + 1] for i in range(len(new_perm) - 1))
        if sorted:
            return sym_help._unsqueeze_helper(g, tensor, [perm.index(axis) for axis in unsqueeze_axes])

    if len(unsqueeze_axes) > 0:
        tensor = sym_help._unsqueeze_helper(g, tensor, unsqueeze_axes)
    if need_permute(perm):
        tensor = g.op("Transpose", tensor, perm_i=perm)
    return tensor

def unsqueeze_and_permute_for_matmul(g, tensor, unsqueeze_axes, perm1, perm2):
    # When going here, the unsqueeze axes must be some axes at the end.
    # We can combine two permutes and remove unsqueeze axes, because we will reshape it after this.
    # For example, a.unsqueeze([2,3]).permute([2,3,1,0]).permute([0,1,3,2])
    # = a.unsqueeze([2,3]).permute([2,3,0,1]) = a.permute([0,1]) = a.
    new_perm = [perm1[axis] for axis in perm2]
    new_perm = [axis for axis in new_perm if axis not in unsqueeze_axes]
    if need_permute(new_perm):
        tensor = g.op("Transpose", tensor, perm_i=new_perm)
    return tensor

def shape_tensor_by_axes(g, input, input_shape, axes):
    if input_shape is None:
        input_shape = g.op("Shape", input)
    return g.op("Gather", input_shape, g.op("Constant", value_t=torch.tensor(axes, dtype=torch.int64)), axis_i=0), input_shape

def total_size_tensor_by_axes(g, input, input_shape, axes):
    if len(axes) == 1:
        return shape_tensor_by_axes(g, input, input_shape, axes)

    if input_shape is None:
        input_shape = g.op("Shape", input)
    size_tensor = g.op("Gather", input_shape, g.op("Constant", value_t=torch.tensor(axes[0], dtype=torch.int64)), axis_i=0)
    for i in range(1, len(axes)):
        next_size = g.op("Gather", input_shape, g.op("Constant", value_t=torch.tensor(axes[i], dtype=torch.int64)), axis_i=0)
        size_tensor = g.op("Mul", size_tensor, next_size)
    return sym_help._unsqueeze_helper(g, size_tensor, [0]), input_shape

@register_symbolic('einsum')
@parse_args('s', 'v')
def einsum(g, equation, tensor_list):
    tensors = sym_help._unpack_list(tensor_list)
    num_ops = len(tensors)
    assert num_ops > 0

    # Doesn't support implicit output is ellipsis or more than 2 oprands for now.
    # Doesn't support ellipsis ('...') for now as not easy to get sizes of oprands.
    if num_ops != 2 or equation.find('->') == -1 or '.' in equation:
        return g.op("Einsum", *tensors, equation_s=equation)

    # Take "ks,ksm->sm" as example. After prcoess inputs,
    # lhs_labels = [k,s], rhs_labels = [k,s,m], result_labels = [s, m].
    lhs_labels, rhs_labels, result_labels = parse_equation(equation)

    # Doesn't support repeated label in operand for now as it needs to take extra diagonal.
    if len(lhs_labels) != len(set(lhs_labels)) or len(rhs_labels) != len(set(rhs_labels)):
        return g.op("Einsum", *tensors, equation_s=equation)

    # Add contraction labels (labels not present in output).
    # After process contraction labels, contraction_labels = [k],
    # label_perm_map = {(s, 0), (m, 1), (k, 2)}, out_size = 2, perm_size = 3.
    out_size = len(result_labels)
    label_perm_map = dict([(label, idx) for idx, label in enumerate(result_labels)])
    perm_size = out_size
    contraction_labels = []
    for label in lhs_labels + rhs_labels:
        if label not in label_perm_map:
            # If contraction label is missing in one side, need extra sum(dim), doesn't support for now.
            if label not in lhs_labels or label not in rhs_labels:
                return g.op("Einsum", *tensors, equation_s=equation)
            label_perm_map[label] = perm_size
            contraction_labels.append(label)
            perm_size += 1

    # Need to unsqueeze and permute the inputs to order of output with contraction labels.
    # lhs_perm = [1,2,0], lhs_unsqueeze_axes = [2].
    # rhs_perm = [1,2,0], rhs_unsqueeze_axes = [].
    lhs_perm, lhs_unsqueeze_axes = map_labels_to_output(lhs_labels, label_perm_map)
    rhs_perm, rhs_unsqueeze_axes = map_labels_to_output(rhs_labels, label_perm_map)

    lhs_tensor = tensors[0]
    rhs_tensor = tensors[1]

    # If there is no contraction labels, unsqueeze and permute the inputs and Mul them to get final result.
    if not contraction_labels:
        lhs_tensor = unsqueeze_and_permute_for_mul(g, lhs_tensor, lhs_unsqueeze_axes, lhs_perm)
        rhs_tensor = unsqueeze_and_permute_for_mul(g, rhs_tensor, rhs_unsqueeze_axes, rhs_perm)
        return g.op("Mul", lhs_tensor, rhs_tensor)

    # If contraction_labels is not empty, need a BatchedMatMul.
    # Batched labels are those in all inputs and output. Below axes are based on output.
    # batched_labels = [s], batched_axes = [0]
    # Matmul output labels are those in one of inputs and output.
    # matmul_output_labels = [m], matmul_output_axes = [1]
    # contraction_labels = [k], contraction_axes = [2]
    lhs_shape_tensor = None
    rhs_shape_tensor = None
    batched_axes = []
    matmul_output_axes = []
    contraction_axes = [v for v in range(out_size, perm_size)]
    for axis in range(out_size):
        label = result_labels[axis]
        if label in lhs_labels and label in rhs_labels:
            batched_axes.append(axis)
        else:
            matmul_output_axes.append(axis)

    # Based on above unsqueeze and permute on inputs, need to permute again.
    # For lhs input, the new permute is batched_axes + matmul_output_axes + contraction_axes: [0, 1, 2],
    # i.e., a.unsqueeze([2]).permute([1,2,0]).permute([0,1,2]) = [s,1,k].
    # For rhs input, the new permute is batched_axes + contraction_axes + matmul_output_axes: [0, 2, 1].
    # i.e., b.unsqueeze([]).permute([1,2,0]).permute([0,2,1]) = [s,k,m].
    lhs_tensor = unsqueeze_and_permute_for_matmul(g, lhs_tensor, lhs_unsqueeze_axes, lhs_perm, batched_axes + matmul_output_axes + contraction_axes)
    rhs_tensor = unsqueeze_and_permute_for_matmul(g, rhs_tensor, rhs_unsqueeze_axes, rhs_perm, batched_axes + contraction_axes + matmul_output_axes)

    # Need to Reshape two input tensors before the BatchedMatMul and Reshape result to output shape.
    # Convert all axes based on inputs.
    # lhs_batched_axes = [1], lhs_contraction_axes = [0], lhs_matmul_output_axes = [], rhs_matmul_output_axes = [2].
    lhs_batched_axes = [lhs_labels.index(result_labels[axis]) for axis in batched_axes]
    lhs_contraction_axes = [lhs_labels.index(label) for label in contraction_labels]
    lhs_matmul_output_axes = [lhs_labels.index(result_labels[axis]) for axis in matmul_output_axes if result_labels[axis] in lhs_labels]
    rhs_matmul_output_axes = [rhs_labels.index(result_labels[axis]) for axis in matmul_output_axes if result_labels[axis] in rhs_labels]

    # Check if Reshape is needed.
    # Reshape lhs input to [[batched_shapes], Mul(lhs_matmul_output_shapes), Mul(contraction_shapes)].
    # Reshape rhs input to [[batched_shapes], Mul(contraction_shapes), Mul(rhs_matmul_output_shapes)]
    # Here, lhs_need_reshape = True, rhs_need_reshape = False, output_need_reshape = True.
    lhs_need_reshape = (len(lhs_matmul_output_axes) != 1 or len(lhs_contraction_axes) != 1)
    rhs_need_reshape = (len(rhs_matmul_output_axes) != 1 or len(lhs_contraction_axes) != 1)
    output_need_reshape = (len(lhs_matmul_output_axes) != 1 or len(rhs_matmul_output_axes) != 1)

    # batch_shape_tensor = tensor([size(s)])
    batch_shape_tensor = None
    if batched_axes and (lhs_need_reshape or rhs_need_reshape or output_need_reshape):
        batch_shape_tensor, lhs_shape_tensor = shape_tensor_by_axes(g, tensors[0], lhs_shape_tensor, lhs_batched_axes)

    # contraction_shape_tensor = tensor([size(k)])
    contraction_shape_tensor = None
    if lhs_need_reshape or rhs_need_reshape:
        contraction_shape_tensor, lhs_shape_tensor = total_size_tensor_by_axes(g, tensors[0], lhs_shape_tensor, lhs_contraction_axes)

    # output_shape_tensor = tensor([size(s), size(m)])
    # lhs_single_matmul_output_shape_tensor/rhs_single_matmul_output_shape_tensor is caches for later use if
    # lhs_matmul_output_axes/rhs_matmul_output_axes has only 1 dim.
    # Here lhs_single_matmul_output_shape_tensor = None, rhs_single_matmul_output_shape_tensor = tensor([size(m)]).
    output_shape_tensor = None
    lhs_single_matmul_output_shape_tensor = None
    rhs_single_matmul_output_shape_tensor = None
    if output_need_reshape:
        shape_tensors = []
        if batch_shape_tensor is not None:
            shape_tensors.append(batch_shape_tensor)
        if lhs_matmul_output_axes:
            lhs_matmul_output_shape_tensor, lhs_shape_tensor = shape_tensor_by_axes(g, tensors[0], lhs_shape_tensor, lhs_matmul_output_axes)
            shape_tensors.append(lhs_matmul_output_shape_tensor)
            if len(lhs_matmul_output_axes) == 1:
                lhs_single_matmul_output_shape_tensor = shape_tensors[-1]
        if rhs_matmul_output_axes:
            rhs_matmul_output_shape_tensor, rhs_shape_tensor = shape_tensor_by_axes(g, tensors[1], rhs_shape_tensor, rhs_matmul_output_axes)
            shape_tensors.append(rhs_matmul_output_shape_tensor)
            if len(rhs_matmul_output_axes) == 1:
                rhs_single_matmul_output_shape_tensor = shape_tensors[-1]
        output_shape_tensor = g.op("Concat", *shape_tensors, axis_i=0) if len(shape_tensors) > 1 else shape_tensors[0]

    # lhs_new_shape_tensor = tensor([size(s), 1, size(k)])
    if lhs_need_reshape:
        shape_tensors = []
        if batch_shape_tensor is not None:
            shape_tensors.append(batch_shape_tensor)
        if lhs_single_matmul_output_shape_tensor is not None:
            shape_tensors.append(lhs_single_matmul_output_shape_tensor)
        elif not lhs_matmul_output_axes:
            shape_tensors.append(g.op("Constant", value_t=torch.tensor([1], dtype=torch.int64)))
        else:
            lhs_matmul_output_total_size_tensor, lhs_shape_tensor = total_size_tensor_by_axes(g, tensors[0], lhs_shape_tensor, lhs_matmul_output_axes)
            shape_tensors.append(lhs_matmul_output_total_size_tensor)
        shape_tensors.append(contraction_shape_tensor)
        lhs_new_shape_tensor = g.op("Concat", *shape_tensors, axis_i=0) if len(shape_tensors) > 1 else shape_tensors[0]
        lhs_tensor = g.op("Reshape", lhs_tensor, lhs_new_shape_tensor)

    # rhs_new_shape_tensor should be tensor([size(s), size(k), size(m)]), but since rhs_need_reshape is False, it's None here.
    if rhs_need_reshape:
        shape_tensors = []
        if batch_shape_tensor is not None:
            shape_tensors.append(batch_shape_tensor)
        shape_tensors.append(contraction_shape_tensor)
        if rhs_single_matmul_output_shape_tensor is not None:
            shape_tensors.append(rhs_single_matmul_output_shape_tensor)
        elif not rhs_matmul_output_axes:
            shape_tensors.append(g.op("Constant", value_t=torch.tensor([1], dtype=torch.int64)))
        else:
            rhs_matmul_output_total_size_tensor, rhs_shape_tensor = total_size_tensor_by_axes(g, tensors[1], rhs_shape_tensor, rhs_matmul_output_axes)
            shape_tensors.append(rhs_matmul_output_total_size_tensor)
        rhs_new_shape_tensor = g.op("Concat", *shape_tensors, axis_i=0) if len(shape_tensors) > 1 else shape_tensors[0]
        rhs_tensor = g.op("Reshape", rhs_tensor, rhs_new_shape_tensor)

    # Perform final BatchedMatMul and Reshape to output is needed.
    result = g.op("MatMul", lhs_tensor, rhs_tensor)
    if output_need_reshape:
        result = g.op("Reshape", result, output_shape_tensor)

    # Now output axes is ordered by [batched_axes, lhs_matmul_output_axes, rhs_matmut_output_axes], if this is not same as output, need one permute.
    labels = [lhs_labels[axis] for axis in lhs_batched_axes + lhs_matmul_output_axes] + [rhs_labels[axis] for axis in rhs_matmul_output_axes]
    assert len(labels) == out_size
    output_perm = [labels.index(label) for label in result_labels]
    assert all(axis in output_perm for axis in range(out_size))
    if need_permute(output_perm):
        result = g.op("Transpose", result, perm_i=output_perm)

    return result
# End of torch.einsum.
