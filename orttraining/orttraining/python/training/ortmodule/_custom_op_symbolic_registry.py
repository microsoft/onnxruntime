# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import torch
import torch.onnx.symbolic_helper as sym_help
from torch.onnx import register_custom_op_symbolic
from torch.onnx.symbolic_helper import _get_tensor_dim_size, _get_tensor_sizes, parse_args


class CustomOpSymbolicRegistry:
    _SYMBOLICS = {}

    @classmethod
    def register(cls, name, domain, fn):
        cls._SYMBOLICS[domain + "::" + name] = fn

    @classmethod
    def register_all(cls):
        for name, fn in cls._SYMBOLICS.items():
            # Symbolic name is in format: domain::name
            register_custom_op_symbolic(name, fn, 1)


def register_symbolic(name, domain=""):
    def symbolic_wrapper(fn):
        CustomOpSymbolicRegistry.register(name, domain, fn)
        return fn

    return symbolic_wrapper


@register_symbolic("cross_entropy_loss")
@parse_args("v", "v", "v", "i", "v", "v")
def cross_entropy_loss(g, self, target, weight, reduction, ignore_index, label_smoothing=0.0):
    label_smoothing = sym_help._maybe_get_const(label_smoothing, "f")
    if label_smoothing > 0.0:
        raise RuntimeError("Unsupported: ONNX does not support label_smoothing")

    # reduction: 0->none, 1->mean, 2->sum
    reduction = sym_help._maybe_get_const(reduction, "i")
    reduction_vals = ["none", "mean", "sum"]
    reduction = reduction_vals[reduction]
    output, log_prob = g.op(
        "com.microsoft::SoftmaxCrossEntropyLossInternal",
        self,
        target,
        weight,
        ignore_index,
        reduction_s=reduction,
        outputs=2,
    )
    output.setType(self.type())
    log_prob.setType(self.type())
    return output


@register_symbolic("nll_loss")
@parse_args("v", "v", "v", "i", "v")
def nll_loss(g, self, target, weight, reduction, ignore_index):
    # reduction: 0->none, 1->mean, 2->sum
    reduction = sym_help._maybe_get_const(reduction, "i")
    reduction_vals = ["none", "mean", "sum"]
    reduction = reduction_vals[reduction]
    output = g.op(
        "com.microsoft::NegativeLogLikelihoodLossInternal", self, target, weight, ignore_index, reduction_s=reduction
    )
    output.setType(self.type())
    return output


@register_symbolic("embedding")
def embedding(g, weight, indices, padding_idx, scale_grad_by_freq, sparse):
    output = g.op(
        "org.pytorch.aten::ATen", weight, indices, padding_idx, scale_grad_by_freq, sparse, operator_s="embedding"
    )
    indices_shape = _get_tensor_sizes(indices)
    if indices_shape is not None and hasattr(weight.type(), "with_sizes"):
        output_type = weight.type().with_sizes(indices_shape + [_get_tensor_dim_size(weight, 1)])
        output.setType(output_type)
    return output


@register_symbolic("bitwise_or")
def bitwise_or(g, self, other):
    return g.op("org.pytorch.aten::ATen", self, other, operator_s="bitwise_or", overload_name_s="Tensor")


@register_symbolic("diagonal")
def diagonal(g, self, offset, dim1, dim2):
    return g.op("org.pytorch.aten::ATen", self, offset, dim1, dim2, operator_s="diagonal")


@register_symbolic("multinomial")
def multinomial(g, self, num_samples, replacement=False, generator=None):
    if generator is not None and not sym_help._is_none(generator):
        raise RuntimeError("Unsupported: ONNX does not support generator for multinomial")
    return g.op("org.pytorch.aten::ATen", self, num_samples, replacement, generator, operator_s="multinomial")


@register_symbolic("max_pool2d")
def max_pool2d(g, self, kernel_size, stride, padding, dilation, ceil_mode):
    stride_val = sym_help._maybe_get_const(stride, "is")
    if not stride_val:
        stride = kernel_size
    return g.op(
        "org.pytorch.aten::ATen",
        self,
        kernel_size,
        stride,
        padding,
        dilation,
        ceil_mode,
        operator_s="max_pool2d_with_indices",
        outputs=2,
    )[0]


@register_symbolic("max")
def max(g, self, dim_or_y=None, keepdim=None):
    # torch.max(input), returns the max value in the tensor
    if dim_or_y is None and keepdim is None:
        return g.op("org.pytorch.aten::ATen", self, operator_s="max")
    # torch.max(input, other)
    if keepdim is None:
        return g.op("Max", self, dim_or_y)
    # torch.max(input, dim, keepdim), returns (max_values, max_indices)
    return g.op("org.pytorch.aten::ATen", self, dim_or_y, keepdim, operator_s="max", overload_name_s="dim", outputs=2)


@register_symbolic("min")
def min(g, self, dim_or_y=None, keepdim=None):
    # torch.min(input), returns the min value in the tensor
    if dim_or_y is None and keepdim is None:
        return g.op("org.pytorch.aten::ATen", self, operator_s="min")
    # torch.min(input, other)
    if keepdim is None:
        return g.op("Min", self, dim_or_y)
    # torch.min(input, dim, keepdim), returns (min_values, min_indices)
    return g.op("org.pytorch.aten::ATen", self, dim_or_y, keepdim, operator_s="min", overload_name_s="dim", outputs=2)


@register_symbolic("unfold")
def unfold(g, input, dimension, size, step):
    return g.op("org.pytorch.aten::ATen", input, dimension, size, step, operator_s="unfold")


@register_symbolic("argmax")
def argmax(g, input, dim, keepdim):
    return g.op("org.pytorch.aten::ATen", input, dim, keepdim, operator_s="argmax")


@register_symbolic("avg_pool2d")
def avg_pool2d(g, self, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override):
    stride_val = sym_help._maybe_get_const(stride, "is")
    if not stride_val:
        stride = kernel_size
    return g.op(
        "org.pytorch.aten::ATen",
        self,
        kernel_size,
        stride,
        padding,
        ceil_mode,
        count_include_pad,
        divisor_override,
        operator_s="avg_pool2d",
    )


@register_symbolic("adaptive_avg_pool2d")
def adaptive_avg_pool2d(g, self, output_size):
    return g.op("org.pytorch.aten::ATen", self, output_size, operator_s="_adaptive_avg_pool2d")


@register_symbolic("binary_cross_entropy_with_logits")
def binary_cross_entropy_with_logits(g, self, target, weight, pos_weight, reduction):
    # If weight is not None, we need to check if it requires grad and add gradient graph accordingly.
    # But current custom_gradient_registry doesn't support such None checking,
    # So doesn't support non-None weight for now.
    if weight is None or sym_help._is_none(weight):
        return g.op(
            "org.pytorch.aten::ATen",
            self,
            target,
            weight,
            pos_weight,
            reduction,
            operator_s="binary_cross_entropy_with_logits",
        )
    from torch.onnx.symbolic_opset12 import binary_cross_entropy_with_logits as bce

    return bce(g, self, target, weight, pos_weight, reduction)


@register_symbolic("numpy_T")
def numpy_T(g, self):
    # Numpy-style `a.T`: returns the tensor
    # with dims reversed
    rank = sym_help._get_tensor_rank(self)
    if rank is not None:
        axes = list(reversed(range(rank)))
        return g.op("Transpose", self, perm_i=axes)
    else:
        # if we don't have dim information we cannot
        # output a permute so use ATen instead
        return g.op("org.pytorch.aten::ATen", self, operator_s="numpy_T")


@register_symbolic("squeeze")
def squeeze(g, self, dim=None):
    # Current _infer_If does not correctly infer shapes from its then- and else- branches, and will
    # cause error in shape inference of following nodes, here we choose to export it as `Squeeze.`
    from torch.onnx.symbolic_opset11 import squeeze as squeeze_with_if

    if dim is None:
        return squeeze_with_if(g, self, dim)
    squeeze_dim = sym_help._get_const(dim, "i", "dim")
    return sym_help._squeeze_helper(g, self, axes_i=[squeeze_dim])


# For torch.einsum.
def parse_equation(equation):
    pos_comma = equation.find(",")
    pos_arrow = equation.find("->")
    assert pos_comma != -1 and pos_arrow > pos_comma
    lhs_labels = [label for label in equation[:pos_comma] if label != " "]
    rhs_labels = [label for label in equation[pos_comma + 1 : pos_arrow] if label != " "]
    result_labels = [label for label in equation[pos_arrow + 2 :] if label != " "]
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


def combine_unsqueeze_and_permute_for_matmul(unsqueeze_axes, perm1, perm2):
    # When going here, the unsqueeze axes must be some axes at the end.
    # We can combine two permutes and remove unsqueeze axes, because we will reshape it after this.
    # For example, a.unsqueeze([2,3]).permute([2,3,1,0]).permute([0,1,3,2])
    # = a.unsqueeze([2,3]).permute([2,3,0,1]) = a.permute([0,1]) = a.
    new_perm = [perm1[axis] for axis in perm2]
    new_perm = [axis for axis in new_perm if axis not in unsqueeze_axes]
    return new_perm


def is_axes_contiguous(axes):
    return len(axes) < 2 or all(axes[axis] + 1 == axes[axis + 1] for axis in range(len(axes) - 1))


def get_shape_tensor_by_axes(g, input, input_shape, axes, need_numel_shape):
    if input_shape is None:
        input_shape = g.op("Shape", input)
    shape_tensor = g.op(
        "Gather", input_shape, g.op("Constant", value_t=torch.tensor(axes, dtype=torch.int64)), axis_i=0
    )
    numel_shape_tensor = None
    if need_numel_shape:
        assert len(axes) > 1
        numel_shape_tensor = g.op("ReduceProd", shape_tensor)
    return shape_tensor, numel_shape_tensor, input_shape


def reshape_tensor(g, input, shape_tensors):
    shape_tensor = g.op("Concat", *shape_tensors, axis_i=0) if len(shape_tensors) > 1 else shape_tensors[0]
    return g.op("Reshape", input, shape_tensor)


def permute_and_reshape_tensor(
    g,
    tensor,
    is_lhs,
    rank,
    perm,
    matmul_output_axes,
    contraction_axes,
    batch_length,
    matmul_output_numel_tensor,
    contraction_numel_tensor,
    shape_tensor,
):
    # If matmul_output_axes and contraction_axes are contiguous in input tensor,
    # we can move Reshape to before Transpose, so it's possible that the Transpoase is fused to MatMul.
    # Otherwise, we have to Transpose first to move those axes together and then Reshape.
    is_matmul_output_axes_contiguous = is_axes_contiguous(matmul_output_axes)
    is_contraction_axes_contiguous = is_axes_contiguous(contraction_axes)
    if is_matmul_output_axes_contiguous and is_contraction_axes_contiguous:
        # Combine contiguous axes to one axis.
        first_matmul_output_axis = matmul_output_axes[0] if len(matmul_output_axes) > 1 else -1
        first_contraction_axis = contraction_axes[0] if len(contraction_axes) > 1 else -1
        # If length of matmul_output_axes and contraction_axes are less than 2, no need to Reshape,
        # it needs an Unsqueeze and a Transpose if needed.
        if first_matmul_output_axis == -1 and first_contraction_axis == -1:
            assert not matmul_output_axes and len(contraction_axes) == 1
            if need_permute(perm):
                new_tensor = sym_help._unsqueeze_helper(g, tensor, [-1])
                pos = batch_length if is_lhs else len(perm)
                perm = perm[:pos] + [len(perm)] + perm[pos:]
                new_tensor = g.op("Transpose", new_tensor, perm_i=perm)
            else:
                new_tensor = sym_help._unsqueeze_helper(g, tensor, [batch_length if is_lhs else -1])
        else:
            axes_to_remove = contraction_axes[1:]  # contraction_axes can't be empty.
            if len(matmul_output_axes) > 1:
                axes_to_remove = axes_to_remove + matmul_output_axes[1:]
            remaining_axes = [axis for axis in range(rank) if axis not in axes_to_remove]
            # Calculate the new shape, use 0 or -1 if possible.
            shape_tensors = []
            before_contiguous_axes = True
            last_zero_dim = -1
            has_neg_one_dim = False
            for axis in remaining_axes:
                if axis == first_matmul_output_axis:
                    shape_tensors.append(matmul_output_numel_tensor)
                    before_contiguous_axes = False
                elif axis == first_contraction_axis:
                    shape_tensors.append(contraction_numel_tensor)
                    before_contiguous_axes = False
                elif before_contiguous_axes:
                    shape_tensors.append(g.op("Constant", value_t=torch.tensor([0], dtype=torch.int64)))
                    last_zero_dim = len(shape_tensors) - 1
                elif axis == remaining_axes[-1]:
                    shape_tensors.append(g.op("Constant", value_t=torch.tensor([-1], dtype=torch.int64)))
                    has_neg_one_dim = True
                else:
                    single_axis_shape_tensor, _, shape_tensor = get_shape_tensor_by_axes(
                        g, tensor, shape_tensor, [axis], False
                    )
                    shape_tensors.append(single_axis_shape_tensor)
            if not has_neg_one_dim and last_zero_dim >= 0:
                shape_tensors[last_zero_dim] = g.op("Constant", value_t=torch.tensor([-1], dtype=torch.int64))
            # Adjust the perm.
            perm = [axis for axis in perm if axis not in axes_to_remove]
            new_axis = 0
            for axis in remaining_axes:
                perm[perm.index(axis)] = new_axis
                new_axis += 1
            # If matmul_output_axes is empty, need to add a dim-1 axis.
            if not matmul_output_axes:
                shape_tensors.append(g.op("Constant", value_t=torch.tensor([1], dtype=torch.int64)))
                pos = batch_length if is_lhs else len(perm)
                perm = perm[:pos] + [new_axis] + perm[pos:]
            new_tensor = reshape_tensor(g, tensor, shape_tensors)
            if need_permute(perm):
                new_tensor = g.op("Transpose", new_tensor, perm_i=perm)
    else:
        if need_permute(perm):
            new_tensor = g.op("Transpose", tensor, perm_i=perm)
        # Calculate the new shape, use 0 or -1 if possible.
        shape_tensors = [g.op("Constant", value_t=torch.tensor([0], dtype=torch.int64))] * batch_length
        if is_lhs:
            if matmul_output_numel_tensor is None:
                matmul_output_numel_tensor = g.op(
                    "Constant", value_t=torch.tensor([1 - len(matmul_output_axes)], dtype=torch.int64)
                )
            shape_tensors.append(matmul_output_numel_tensor)
            shape_tensors.append(g.op("Constant", value_t=torch.tensor([-1], dtype=torch.int64)))
        else:
            if (
                contraction_numel_tensor is None
            ):  # contraction_axes can't be empty, None here means only one contraction axis.
                contraction_numel_tensor = g.op("Constant", value_t=torch.tensor([0], dtype=torch.int64))
            shape_tensors.append(contraction_numel_tensor)
            shape_tensors.append(g.op("Constant", value_t=torch.tensor([-1], dtype=torch.int64)))
        new_tensor = reshape_tensor(g, new_tensor, shape_tensors)
    return new_tensor, shape_tensor


@register_symbolic("einsum")
@parse_args("s", "v")
def einsum(g, equation, tensor_list):
    tensors = sym_help._unpack_list(tensor_list)
    num_ops = len(tensors)
    assert num_ops > 0

    # Doesn't support implicit output is ellipsis or more than 2 oprands for now.
    # Doesn't support ellipsis ('...') for now as not easy to get sizes of oprands.
    if num_ops != 2 or equation.find("->") == -1 or "." in equation:
        return g.op("Einsum", *tensors, equation_s=equation)

    # Take "ks,ksm->sm" as example. After prcoess inputs,
    # lhs_labels = [k,s], rhs_labels = [k,s,m], result_labels = [s,m].
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
    lhs_reduce_sum_axes = []
    rhs_reduce_sum_axes = []
    for label in lhs_labels + rhs_labels:
        if label not in label_perm_map:
            if label in lhs_labels and label in rhs_labels:
                label_perm_map[label] = perm_size
                contraction_labels.append(label)
                perm_size += 1
            elif label in lhs_labels:
                lhs_reduce_sum_axes.append(lhs_labels.index(label))
            else:
                rhs_reduce_sum_axes.append(rhs_labels.index(label))

    lhs_tensor = tensors[0]
    rhs_tensor = tensors[1]

    # If lhs_reduce_sum_axes/rhs_reduce_sum_axes is not empty, ReduceSum on that axes, update lhs_labels/rhs_labels,
    # and use the output as original_lhs_tensor/original_rhs_tensor.
    if lhs_reduce_sum_axes:
        lhs_tensor = sym_help._reducesum_helper(g, lhs_tensor, lhs_reduce_sum_axes, keepdims_i=False)
        lhs_labels = [lhs_labels[axis] for axis in range(len(lhs_labels)) if axis not in lhs_reduce_sum_axes]

    if rhs_reduce_sum_axes:
        rhs_tensor = sym_help._reducesum_helper(g, rhs_tensor, rhs_reduce_sum_axes, keepdims_i=False)
        rhs_labels = [rhs_labels[axis] for axis in range(len(rhs_labels)) if axis not in rhs_reduce_sum_axes]

    # Need to unsqueeze and permute the inputs to order of output with contraction labels.
    # lhs_perm = [1,2,0], lhs_unsqueeze_axes = [2].
    # rhs_perm = [1,2,0], rhs_unsqueeze_axes = [].
    lhs_perm, lhs_unsqueeze_axes = map_labels_to_output(lhs_labels, label_perm_map)
    rhs_perm, rhs_unsqueeze_axes = map_labels_to_output(rhs_labels, label_perm_map)

    # If there is no contraction labels, unsqueeze and permute the inputs and Mul them to get final result.
    if not contraction_labels:
        lhs_tensor = unsqueeze_and_permute_for_mul(g, lhs_tensor, lhs_unsqueeze_axes, lhs_perm)
        rhs_tensor = unsqueeze_and_permute_for_mul(g, rhs_tensor, rhs_unsqueeze_axes, rhs_perm)
        return g.op("Mul", lhs_tensor, rhs_tensor)

    # If contraction_labels is not empty, need a BatchedMatMul.
    # Batched labels are those in all inputs and output. Below axes are based on output.
    # batched_labels = [s], batched_axes = [0] for the example.
    # Matmul output labels are those in one of inputs and output.
    # matmul_output_labels = [m], matmul_output_axes = [1] for the example.
    # contraction_labels = [k], contraction_axes = [2] for the example.
    batched_axes = []
    matmul_output_axes = []
    contraction_axes = [axis for axis in range(out_size, perm_size)]
    for axis in range(out_size):
        label = result_labels[axis]
        if label in lhs_labels and label in rhs_labels:
            batched_axes.append(axis)
        else:
            matmul_output_axes.append(axis)

    # Based on above unsqueeze and permute on inputs, need to permute again.
    # For lhs input, the new permute is batched_axes + matmul_output_axes + contraction_axes: [0, 1, 2],
    # i.e., a.unsqueeze([2]).permute([1,2,0]).permute([0,1,2]) = [s,1,k] for the example.
    # For rhs input, the new permute is batched_axes + contraction_axes + matmul_output_axes: [0, 2, 1].
    # i.e., b.unsqueeze([]).permute([1,2,0]).permute([0,2,1]) = [s,k,m] for the example.
    lhs_perm = combine_unsqueeze_and_permute_for_matmul(
        lhs_unsqueeze_axes, lhs_perm, batched_axes + matmul_output_axes + contraction_axes
    )
    rhs_perm = combine_unsqueeze_and_permute_for_matmul(
        rhs_unsqueeze_axes, rhs_perm, batched_axes + contraction_axes + matmul_output_axes
    )

    # Need to Reshape two input tensors before the BatchedMatMul and Reshape result to output shape.
    # Reshape lhs input to [[batched_shapes], Mul(lhs_matmul_output_shapes), Mul(contraction_shapes)].
    # Reshape rhs input to [[batched_shapes], Mul(contraction_shapes), Mul(rhs_matmul_output_shapes)]
    # Convert all axes based on inputs.
    # lhs_contraction_axes = [0], rhs_contraction_axes = [0], lhs_matmul_output_axes = [], rhs_matmul_output_axes = [2] for the example.
    lhs_contraction_axes = [lhs_labels.index(label) for label in contraction_labels]
    rhs_contraction_axes = [rhs_labels.index(label) for label in contraction_labels]
    lhs_matmul_output_axes = [
        lhs_labels.index(result_labels[axis]) for axis in matmul_output_axes if result_labels[axis] in lhs_labels
    ]
    rhs_matmul_output_axes = [
        rhs_labels.index(result_labels[axis]) for axis in matmul_output_axes if result_labels[axis] in rhs_labels
    ]

    # Caches of input shape tensors to avoid generating duplicated graph.
    lhs_shape_tensor = None
    rhs_shape_tensor = None

    # contraction_numel_tensor should be tensor([size(k)]) for the example, but since length is 1, it's None here.
    contraction_numel_tensor = None
    if len(lhs_contraction_axes) > 1:
        _, contraction_numel_tensor, lhs_shape_tensor = get_shape_tensor_by_axes(
            g, lhs_tensor, lhs_shape_tensor, lhs_contraction_axes, True
        )

    # Prepare some shape tensors for Reshape if needed.
    # Both lhs_matmul_output_shape_tensor and lhs_matmul_output_numel_tensor is None for the example.
    lhs_matmul_output_shape_tensor = None
    lhs_matmul_output_numel_tensor = None
    if len(lhs_matmul_output_axes) > 1:
        lhs_matmul_output_shape_tensor, lhs_matmul_output_numel_tensor, lhs_shape_tensor = get_shape_tensor_by_axes(
            g, lhs_tensor, lhs_shape_tensor, lhs_matmul_output_axes, True
        )

    # Both rhs_matmul_output_shape_tensor and rhs_matmul_output_numel_tensor is None for the example.
    rhs_matmul_output_shape_tensor = None
    rhs_matmul_output_numel_tensor = None
    if len(rhs_matmul_output_axes) > 1:
        rhs_matmul_output_shape_tensor, rhs_matmul_output_numel_tensor, rhs_shape_tensor = get_shape_tensor_by_axes(
            g, rhs_tensor, rhs_shape_tensor, rhs_matmul_output_axes, True
        )

    new_lhs_tensor = lhs_tensor
    # Need to Reshape lhs_tensor if lhs_matmul_output_axes or lhs_contraction_axes is not 1, otherwise permute it directly.
    # Need to Reshape the lhs_tensor for the example, the new shape is [size(s), 1, size(k)].
    if len(lhs_matmul_output_axes) != 1 or len(lhs_contraction_axes) != 1:
        new_lhs_tensor, lhs_shape_tensor = permute_and_reshape_tensor(
            g,
            lhs_tensor,
            True,
            len(lhs_labels),
            lhs_perm,
            lhs_matmul_output_axes,
            lhs_contraction_axes,
            len(batched_axes),
            lhs_matmul_output_numel_tensor,
            contraction_numel_tensor,
            lhs_shape_tensor,
        )
    else:
        if need_permute(lhs_perm):
            new_lhs_tensor = g.op("Transpose", lhs_tensor, perm_i=lhs_perm)

    # Need to Reshape rhs_tensor if rhs_matmul_output_axes or rhs_contraction_axes is not 1, otherwise permute it directly.
    # rhs_tensor's new shape should be [size(s), size(k), size(m)], but doesn't need to Reshape for the example.
    new_rhs_tensor = rhs_tensor
    if len(rhs_matmul_output_axes) != 1 or len(rhs_contraction_axes) != 1:
        new_rhs_tensor, rhs_shape_tensor = permute_and_reshape_tensor(
            g,
            rhs_tensor,
            False,
            len(rhs_labels),
            rhs_perm,
            rhs_matmul_output_axes,
            rhs_contraction_axes,
            len(batched_axes),
            rhs_matmul_output_numel_tensor,
            contraction_numel_tensor,
            rhs_shape_tensor,
        )
    else:
        if need_permute(rhs_perm):
            new_rhs_tensor = g.op("Transpose", rhs_tensor, perm_i=rhs_perm)

    # Perform final BatchedMatMul. Output is shape [size(s), 1, size(m)] for the example.
    result = g.op("MatMul", new_lhs_tensor, new_rhs_tensor)

    # Need to Reshape the result if lhs_matmul_output_axes or rhs_matmul_output_axes is not 1.
    # Need to Reshape the result for the example, the new shape is [size(s), size(m)].
    if len(lhs_matmul_output_axes) != 1 or len(rhs_matmul_output_axes) != 1:
        shape_tensors = [g.op("Constant", value_t=torch.tensor([0], dtype=torch.int64))] * len(batched_axes)
        last_zero_dim = len(shape_tensors) - 1
        has_neg_one_dim = False
        if lhs_matmul_output_axes:
            if len(lhs_matmul_output_axes) == 1:
                shape_tensors.append(g.op("Constant", value_t=torch.tensor([0], dtype=torch.int64)))
                last_zero_dim = len(shape_tensors) - 1
            else:
                shape_tensors.append(lhs_matmul_output_shape_tensor)
        if rhs_matmul_output_axes:
            if len(rhs_matmul_output_axes) == 1:
                shape_tensors.append(g.op("Constant", value_t=torch.tensor([-1], dtype=torch.int64)))
                has_neg_one_dim = True
            else:
                shape_tensors.append(rhs_matmul_output_shape_tensor)
        if not has_neg_one_dim and last_zero_dim >= 0:
            shape_tensors[last_zero_dim] = g.op("Constant", value_t=torch.tensor([-1], dtype=torch.int64))
        result = reshape_tensor(g, result, shape_tensors)

    # Now output axes is ordered by [batched_axes, lhs_matmul_output_axes, rhs_matmut_output_axes],
    # if this is not same as output, need one permute.
    labels = (
        [result_labels[axis] for axis in batched_axes]
        + [lhs_labels[axis] for axis in lhs_matmul_output_axes]
        + [rhs_labels[axis] for axis in rhs_matmul_output_axes]
    )
    assert len(labels) == out_size
    output_perm = [labels.index(label) for label in result_labels]
    assert all(axis in output_perm for axis in range(out_size))
    if need_permute(output_perm):
        result = g.op("Transpose", result, perm_i=output_perm)

    return result


# End of torch.einsum.
