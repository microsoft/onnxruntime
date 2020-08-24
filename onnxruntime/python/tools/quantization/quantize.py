# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import os
import onnx
import onnx.numpy_helper
import struct
from pathlib import Path

import numpy as np

from onnx import onnx_pb as onnx_proto
from onnx import shape_inference
from onnxruntime import SessionOptions, InferenceSession, GraphOptimizationLevel

from .quant_utils import QuantizationMode,QuantizedValueType,QuantizedInitializer,QuantizedValue,quantization_modes
from .quant_utils import _find_by_name,_get_elem_index,_get_mul_node,_generate_identified_filename,_attribute_to_kwarg
from .quant_utils import QuantType, __producer__, __version__

from .registry import CreateOpQuantizer, CreateDefaultOpQuantizer

from .onnx_model import ONNXModel
from .onnx_quantizer import ONNXQuantizer


def optimize_model(model_path:Path):
        '''
            Generate model that applies graph optimization (constant folding,etc.)
            parameter model_path: path to the original onnx model
            return: optimized onnx model
        '''
        opt_model_path = _generate_identified_filename(model_path,"-opt")
        sess_option = SessionOptions()
        sess_option.optimized_model_filepath = opt_model_path.as_posix()
        sess_option.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_BASIC
        session = InferenceSession(model_path.as_posix(),sess_option)
        optimized_model = onnx.load(opt_model_path.as_posix())
        return optimized_model


def quantize(model_path,
             per_channel=False,
             nbits=8,
             quantization_mode=QuantizationMode.IntegerOps,
             static=False,
             force_fusions=False,
             symmetric_activation=False,
             symmetric_weight=False,
             quantization_params=None,
             nodes_to_quantize=None,
             nodes_to_exclude=None):
    '''
        Given an onnx model, create a quantized onnx model and save it into a file
    :param model: ModelProto to quantize
    :param per_channel: quantize weights per channel
    :param nbits: number of bits to represent quantized data. Currently only supporting 8-bit types
    :param quantization_mode: Can be one of the QuantizationMode types.
        IntegerOps:
            the function will use integer ops. Only ConvInteger and MatMulInteger ops are supported now.
        QLinearOps:
            the function will use QLinear ops. Only QLinearConv and QLinearMatMul ops are supported now.
    :param static:
        True: The inputs/activations are quantized using static scale and zero point values
              specified through quantization_params.
        False: The inputs/activations are quantized using dynamic scale and zero point values
               computed while running the model.
    :param force_fusions:
        True: Fuses nodes added for dynamic quantization
        False: No fusion is applied for nodes which are added for dynamic quantization.
        Should be only used in cases where backends want to apply special fusion routines
    :param symmetric_activation:
        True: activations are quantized into signed integers.
        False: activations are quantized into unsigned integers.
    :param symmetric_weight:
        True: weights are quantized into signed integers.
        False: weights are quantized into unsigned integers.
    :param quantization_params:
        Dictionary to specify the zero point and scale values for inputs to conv and matmul nodes.
        Should be specified when static is set to True.
        The quantization_params should be specified in the following format:
            {
                "input_name": [zero_point, scale]
            }.
        zero_point should be of type np.uint8 and scale should be of type np.float32.
        example:
            {
                'resnet_model/Relu_1:0': [np.uint8(0), np.float32(0.019539741799235344)],
                'resnet_model/Relu_2:0': [np.uint8(0), np.float32(0.011359662748873234)]
            }
    :return: ModelProto with quantization
    :param nodes_to_quantize:
        List of nodes names to quantize. When this list is not None only the nodes in this list
        are quantized.
        example:
        [
            'Conv__224',
            'Conv__252'
        ]
    :param nodes_to_exclude:
        List of nodes names to exclude. The nodes in this list will be excluded from quantization
        when it is not None.
    '''
    
    print("Warning: onnxruntime.quantization.quantize is deprecated.\n\
           Please use quantize_static for static quantization, quantize_dynamic for dynamic quantization, and quantize_qat for quantization-aware training quantization.")

    if nbits == 8:
        input_qType = onnx_proto.TensorProto.INT8 if symmetric_activation else onnx_proto.TensorProto.UINT8
        weight_qType = onnx_proto.TensorProto.INT8 if symmetric_weight else onnx_proto.TensorProto.UINT8
        mode = quantization_mode

        #optimize the original model
        optimized_model = optimize_model(Path(model_path))
        copy_model = onnx_proto.ModelProto()
        copy_model.CopyFrom(optimized_model)

        #check opset version of the original model
        fuse_dynamic_quant = check_opset_version(onnx.load(model_path), force_fusions)
        
        #apply shape inference to the ModelProto and get value informations
        inferred_model = shape_inference.infer_shapes(copy_model)
        value_infos = {vi.name: vi for vi in inferred_model.graph.value_info}
        
        #create ONNXModel and ONNXQuantizer
        onnx_model = ONNXModel(inferred_model)
        quantizer = ONNXQuantizer(onnx_model, value_infos, per_channel, mode, static, fuse_dynamic_quant, weight_qType, input_qType,
                                  quantization_params, nodes_to_quantize, nodes_to_exclude)
        quantizer.quantize_model()
        quantizer.model.producer_name = __producer__
        quantizer.model.producer_version = __version__
        return quantizer.model.model
    else:
        raise ValueError('Only 8 bit quantization is currently supported')


def quantize_static(model_path,
                    per_channel=False,
                    activation_type=QuantType.QUInt8,
                    weight_type=QuantType.QUInt8,
                    quantization_params=None,
                    nodes_to_quantize=None,
                    nodes_to_exclude=None):
    '''
        Given an onnx model, create a quantized onnx model and save it into a file
    :param model_input: file path of model to quantize
    :param model_output: file path of quantized model
    :param per_channel: quantize weights per channel
    :param nbits: number of bits to represent quantized data. Currently only supporting 8-bit types
    :param activation_type: quantization data type of activation
    :param weight_type: quantization data type of weight
    :param quantization_params:
        Dictionary to specify the zero point and scale values for inputs to conv and matmul nodes.
        It is required.
        The quantization_params should be specified in the following format:
            {
                "input_name": [zero_point, scale]
            }.
        zero_point should be of type np.uint8 and scale should be of type np.float32.
        example:
            {
                'resnet_model/Relu_1:0': [np.uint8(0), np.float32(0.019539741799235344)],
                'resnet_model/Relu_2:0': [np.uint8(0), np.float32(0.011359662748873234)]
            }
    :return: ModelProto with quantization
    :param nodes_to_quantize:
        List of nodes names to quantize. When this list is not None only the nodes in this list
        are quantized.
        example:
        [
            'Conv__224',
            'Conv__252'
        ]
    :param nodes_to_exclude:
        List of nodes names to exclude. The nodes in this list will be excluded from quantization
        when it is not None.
    '''

    input_qType = onnx_proto.TensorProto.INT8 if activation_type == QuantType.QInt8 else onnx_proto.TensorProto.UINT8
    weight_qType = onnx_proto.TensorProto.INT8 if weight_type == QuantType.QInt8 else onnx_proto.TensorProto.UINT8
    mode = QuantizationMode.QLinearOps

    #optimize the original model
    optimized_model = optimize_model(Path(model_path))
    copy_model = onnx_proto.ModelProto()
    copy_model.CopyFrom(optimized_model)

    #check opset version of the original model
    #fuse_dynamic_quant = check_opset_version(onnx.load(model_path), force_fusions)
    fuse_dynamic_quant = True

    #apply shape inference to the ModelProto and get value informations
    inferred_model = shape_inference.infer_shapes(copy_model)
    value_infos = {vi.name: vi for vi in inferred_model.graph.value_info}

    #create ONNXModel and ONNXQuantizer
    onnx_model = ONNXModel(inferred_model)
    quantizer = ONNXQuantizer(onnx_model,
                              value_infos,
                              per_channel,
                              mode,
                              True, # static
                              fuse_dynamic_quant,
                              weight_qType,
                              input_qType,
                              quantization_params,
                              nodes_to_quantize,
                              nodes_to_exclude)

    quantizer.quantize_model()
    quantizer.model.producer_name = __producer__
    quantizer.model.producer_version = __version__
    return quantizer.model.model

def quantize_dynamic(model_path,
                    per_channel=False,
                    activation_type=QuantType.QUInt8,
                    weight_type=QuantType.QUInt8,
                    nodes_to_quantize=None,
                    nodes_to_exclude=None):
    '''
        Given an onnx model, create a quantized onnx model and save it into a file
    :param model_input: file path of model to quantize
    :param model_output: file path of quantized model
    :param per_channel: quantize weights per channel
    :param nbits: number of bits to represent quantized data. Currently only supporting 8-bit types
    :param activation_type: quantization data type of activation
    :param weight_type: quantization data type of weight
    :param quantization_params:
        Dictionary to specify the zero point and scale values for inputs to conv and matmul nodes.
        It is required.
        The quantization_params should be specified in the following format:
            {
                "input_name": [zero_point, scale]
            }.
        zero_point should be of type np.uint8 and scale should be of type np.float32.
        example:
            {
                'resnet_model/Relu_1:0': [np.uint8(0), np.float32(0.019539741799235344)],
                'resnet_model/Relu_2:0': [np.uint8(0), np.float32(0.011359662748873234)]
            }
    :return: ModelProto with quantization
    :param nodes_to_quantize:
        List of nodes names to quantize. When this list is not None only the nodes in this list
        are quantized.
        example:
        [
            'Conv__224',
            'Conv__252'
        ]
    :param nodes_to_exclude:
        List of nodes names to exclude. The nodes in this list will be excluded from quantization
        when it is not None.
    '''

    input_qType = onnx_proto.TensorProto.INT8 if activation_type == QuantType.QInt8 else onnx_proto.TensorProto.UINT8
    weight_qType = onnx_proto.TensorProto.INT8 if weight_type == QuantType.QInt8 else onnx_proto.TensorProto.UINT8
    mode = QuantizationMode.IntegerOps

    #optimize the original model
    optimized_model = optimize_model(Path(model_path))
    copy_model = onnx_proto.ModelProto()
    copy_model.CopyFrom(optimized_model)

    #check opset version of the original model
    #fuse_dynamic_quant = check_opset_version(onnx.load(model_path), force_fusions)
    fuse_dynamic_quant = True

    #apply shape inference to the ModelProto and get value informations
    inferred_model = shape_inference.infer_shapes(copy_model)
    value_infos = {vi.name: vi for vi in inferred_model.graph.value_info}

    #create ONNXModel and ONNXQuantizer
    onnx_model = ONNXModel(inferred_model)
    quantizer = ONNXQuantizer(onnx_model,
                              value_infos,
                              per_channel,
                              mode,
                              False, #static
                              fuse_dynamic_quant,
                              weight_qType,
                              input_qType,
                              None,
                              nodes_to_quantize,
                              nodes_to_exclude)

    quantizer.quantize_model()
    quantizer.model.producer_name = __producer__
    quantizer.model.producer_version = __version__
    return quantizer.model.model