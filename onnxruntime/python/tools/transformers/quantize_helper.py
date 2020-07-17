# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import logging
import torch
import onnx
from transformers.modeling_utils import Conv1D

logger = logging.getLogger('')


def _conv1d_to_linear(module):
    in_size, out_size = module.weight.shape
    linear = torch.nn.Linear(in_size, out_size)
    linear.weight.data = module.weight.data.T.contiguous()
    linear.bias.data = module.bias.data
    return linear


def conv1d_to_linear(model):
    '''in-place
    This is for Dynamic Quantization, as Conv1D is not recognized by PyTorch, convert it to nn.Linear
    '''
    logger.debug("replace Conv1D with Linear")
    for name in list(model._modules):
        module = model._modules[name]
        if isinstance(module, Conv1D):
            linear = _conv1d_to_linear(module)
            model._modules[name] = linear
        else:
            conv1d_to_linear(module)


class QuantizeHelper:
    @staticmethod
    def quantize_torch_model(model, dtype=torch.qint8):
        '''
        Usage: model = quantize_model(model)

        TODO: mix of in-place and return, but results are different
        '''
        conv1d_to_linear(model)
        return torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=dtype)

    @staticmethod
    def quantize_onnx_model(onnx_model_path, quantized_model_path):
        from onnxruntime.quantization import quantize, QuantizationMode
        onnx_opt_model = onnx.load(onnx_model_path)
        quantized_onnx_model = quantize(onnx_opt_model,
                                        quantization_mode=QuantizationMode.IntegerOps,
                                        symmetric_weight=True,
                                        force_fusions=True)
        onnx.save(quantized_onnx_model, quantized_model_path)
        logger.info(f"quantized model saved to:{quantized_model_path}")
