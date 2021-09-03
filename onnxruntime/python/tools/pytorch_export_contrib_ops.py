# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
Support for registering ONNX Runtime's built-in contrib ops with
PyTorch-ONNX exporter (torch.onnx.export).
"""

import typing

try:
    from torch.onnx import register_custom_op_symbolic
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "This module is only useful in combination with PyTorch. "
        "To install PyTorch see https://pytorch.org/.")
import torch.onnx.symbolic_helper as sym_help
import torch.onnx.symbolic_registry as sym_registry

_OPSET_VERSION = 1
_registered_ops: typing.AbstractSet[str] = set()


def _reg(symbolic_fn: typing.Callable):
    name = "::%s" % symbolic_fn.__name__
    register_custom_op_symbolic(name, symbolic_fn, _OPSET_VERSION)
    _registered_ops.add(name)


def register():
    """Register ONNX Runtime's built-in contrib ops.

    Should be run before torch.onnx.export().
    """

    def grid_sample(g, input, grid, mode, padding_mode, align_corners):
        # mode
        #   'bilinear'      : onnx::Constant[value={0}]
        #   'nearest'       : onnx::Constant[value={1}]
        #   'bicubic'       : onnx::Constant[value={2}]
        # padding_mode
        #   'zeros'         : onnx::Constant[value={0}]
        #   'border'        : onnx::Constant[value={1}]
        #   'reflection'    : onnx::Constant[value={2}]
        mode = sym_help._maybe_get_const(mode, "i")
        padding_mode = sym_help._maybe_get_const(padding_mode, "i")
        mode_str = ['bilinear', 'nearest', 'bicubic'][mode]
        padding_mode_str = ['zeros', 'border', 'reflection'][padding_mode]
        align_corners = int(sym_help._maybe_get_const(align_corners, "b"))

        # From opset v13 onward, the output shape can be specified with
        # (N, C, H, W) (N, H_out, W_out, 2) => (N, C, H_out, W_out)
        # input_shape = input.type().sizes()
        # gird_shape = grid.type().sizes()
        # output_shape = input_shape[:2] + gird_shape[1:3]
        # g.op(...).setType(input.type().with_sizes(output_shape))

        return g.op("com.microsoft::GridSample", input, grid,
                    mode_s=mode_str,
                    padding_mode_s=padding_mode_str,
                    align_corners_i=align_corners)
    _reg(grid_sample)

    def inverse(g, self):
        return g.op("com.microsoft::Inverse", self).setType(self.type())
    _reg(inverse)

    def gelu(g, self):
        return g.op("com.microsoft::Gelu", self).setType(self.type())
    _reg(gelu)

    def triu(g, self, diagonal):
        return g.op("com.microsoft::Trilu", self, diagonal, upper_i=1).setType(self.type())
    _reg(triu)

    def tril(g, self, diagonal):
        return g.op("com.microsoft::Trilu", self, diagonal, upper_i=0).setType(self.type())
    _reg(tril)



def unregister():
    """Unregister ONNX Runtime's built-in contrib ops."""
    # TODO: replace this once PyTorch supports unregister natively.
    # https://msdata.visualstudio.com/Vienna/_workitems/edit/1342343
    for name in _registered_ops:
        ns, kind = name.split("::")
        for version in sym_help._onnx_stable_opsets:
            if (version >= _OPSET_VERSION and
                sym_registry.is_registered_op(kind, ns, version)):
                del sym_registry._registry[(ns, version)][kind]
