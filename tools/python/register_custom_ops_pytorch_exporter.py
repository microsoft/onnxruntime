# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
#
# Register pytorch symbolic for export using ONNX Runtime contrib ops

from torch.onnx import register_custom_op_symbolic
from torch.onnx.symbolic_helper import parse_args, _get_tensor_dim_size, _get_tensor_sizes


_onnx_opset_version = 1


def register_custom_op(is_ortmodule=False):
    """
    This function registers symbolic functions for
    custom ops that are implemented as part of ONNX Runtime
    """

    # Symbolic definition
    def inverse(g, self):
        return g.op("com.microsoft::Inverse", self).setType(self.type())

    def gelu(g, self):
        return g.op("com.microsoft::Gelu", self).setType(self.type())

    def triu(g, self, diagonal):
        return g.op("com.microsoft::Trilu", self, diagonal, upper_i=1).setType(self.type())

    def tril(g, self, diagonal):
        return g.op("com.microsoft::Trilu", self, diagonal, upper_i=0).setType(self.type())

    # Op Registration
    register_custom_op_symbolic('::inverse', inverse, _onnx_opset_version)
    register_custom_op_symbolic('::gelu', gelu, _onnx_opset_version)
    register_custom_op_symbolic('::triu', triu, _onnx_opset_version)
    register_custom_op_symbolic('::tril', tril, _onnx_opset_version)

    if is_ortmodule:
        @parse_args('v', 'v', 'i', 'b', 'b')
        def embedding(g, weight, indices, padding_idx, scale_grad_by_freq, sparse):
            custom_attributes_json = (
                '{'
                f'"padding_idx":{str(padding_idx)},'
                f'"scale_grad_by_freq":{str(scale_grad_by_freq).lower()},'
                f'"sparse":{str(sparse).lower()}'
                '}'
            )
            output = g.op("com.microsoft::ATenOp", weight, indices, name_s='aten::embedding',
                          custom_attributes_json_s=custom_attributes_json)
            indices_shape = _get_tensor_sizes(indices)
            if indices_shape is not None and hasattr(weight.type(), 'with_sizes'):
                output_type = weight.type().with_sizes(indices_shape + [_get_tensor_dim_size(weight, 1)])
                output.setType(output_type)
            return output

        register_custom_op_symbolic('::embedding', embedding, _onnx_opset_version)


def unregister_custom_op():
    """
    This function unregisters symbolic functions for
    custom ops that are implemented as part of ONNX Runtime
    """

    import torch.onnx.symbolic_registry as sym_registry

    # TODO: replace this once PyTorch supports unregister natively.
    def unregister(name, opset_version):
        ns, kind = name.split("::")
        from torch.onnx.symbolic_helper import _onnx_stable_opsets

        for version in _onnx_stable_opsets:
            if version >= opset_version and sym_registry.is_registered_op(kind, ns, version):
                del sym_registry._registry[(ns, version)][kind]

    unregister('::inverse', _onnx_opset_version)
    unregister('::gelu', _onnx_opset_version)
    unregister('::triu', _onnx_opset_version)
    unregister('::tril', _onnx_opset_version)
