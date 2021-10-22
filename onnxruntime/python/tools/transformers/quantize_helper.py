# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import logging
import torch
import onnx
import os
from transformers.modeling_utils import Conv1D

logger = logging.getLogger(__name__)


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


def _get_size_of_pytorch_model(model):
    torch.save(model.state_dict(), "temp.p")
    size = os.path.getsize("temp.p") / (1024 * 1024)
    os.remove('temp.p')
    return size


class QuantizeHelper:
    @staticmethod
    def quantize_torch_model(model, dtype=torch.qint8):
        '''
        Usage: model = quantize_model(model)

        TODO: mix of in-place and return, but results are different
        '''
        conv1d_to_linear(model)
        quantized_model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=dtype)
        logger.info(f'Size of full precision Torch model(MB):{_get_size_of_pytorch_model(model)}')
        logger.info(f'Size of quantized Torch model(MB):{_get_size_of_pytorch_model(quantized_model)}')
        return quantized_model

    @staticmethod
    def quantize_onnx_model(onnx_model_path, quantized_model_path, use_external_data_format=False):
        from onnxruntime.quantization import quantize, QuantizationMode
        logger.info(f'Size of full precision ONNX model(MB):{os.path.getsize(onnx_model_path)/(1024*1024)}')
        onnx_opt_model = onnx.load_model(onnx_model_path)
        quantized_onnx_model = quantize(onnx_opt_model,
                                        quantization_mode=QuantizationMode.IntegerOps,
                                        symmetric_weight=True,
                                        force_fusions=True)

        if use_external_data_format:
            from pathlib import Path
            Path(quantized_model_path).parent.mkdir(parents=True, exist_ok=True)
            onnx.external_data_helper.convert_model_to_external_data(quantized_onnx_model,
                                                                     all_tensors_to_one_file=True,
                                                                     location=Path(quantized_model_path).name + ".data")
        onnx.save_model(quantized_onnx_model, quantized_model_path)

        logger.info(f"quantized model saved to:{quantized_model_path}")
        #TODO: inlcude external data in total model size.
        logger.info(f'Size of quantized ONNX model(MB):{os.path.getsize(quantized_model_path)/(1024*1024)}')
