# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from torch.onnx import register_custom_op_symbolic
from torch.onnx.symbolic_helper import parse_args, _get_tensor_dim_size, _get_tensor_sizes
import torch.onnx.symbolic_helper as sym_help


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


@register_symbolic('binary_cross_entropy_with_logits')
def binary_cross_entropy_with_logits(g, self, target, weight, pos_weight, reduction):
    # If weight is not None, we need to check if it requires grad and add gradient graph accordingly.
    # But current custom_gradient_registry doesn't support such None checking,
    # So doesn't support non-None weight for now.
    if weight is None or sym_help._is_none(weight):
        return g.op("com.microsoft::ATenOp", self, target, weight, pos_weight, reduction,
                    name_s='aten::binary_cross_entropy_with_logits')
    from torch.onnx.symbolic_opset12 import binary_cross_entropy_with_logits as bce
    return bce(g, self, target, weight, pos_weight, reduction)
