# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import os
import onnx
import onnx.numpy_helper
import struct
import logging
import numpy as np

from pathlib import Path

from onnx import onnx_pb as onnx_proto
from onnxruntime import SessionOptions, InferenceSession, GraphOptimizationLevel

from .quant_utils import QuantizationMode, QuantizedValueType, QuantizedInitializer, QuantizedValue
from .quant_utils import find_by_name, get_elem_index, get_mul_node, generate_identified_filename, attribute_to_kwarg
from .quant_utils import QuantType, QuantFormat

from .registry import QLinearOpsRegistry, IntegerOpsRegistry

from .onnx_model import ONNXModel
from .onnx_quantizer import ONNXQuantizer
from .qdq_quantizer import QDQQuantizer
from .calibrate import CalibrationDataReader, create_calibrator, CalibrationMethod 


def optimize_model(model_path: Path):
    '''
        Generate model that applies graph optimization (constant folding,etc.)
        parameter model_path: path to the original onnx model
        return: optimized onnx model
    '''
    opt_model_path = generate_identified_filename(model_path, "-opt")
    sess_option = SessionOptions()
    sess_option.optimized_model_filepath = opt_model_path.as_posix()
    sess_option.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_BASIC
    _ = InferenceSession(model_path.as_posix(), sess_option, providers=['CPUExecutionProvider'])
    optimized_model = onnx.load(opt_model_path.as_posix())
    return optimized_model


def load_model(model_path: Path, optimize=True):
    if optimize:
        #optimize the original model
        onnx_model = ONNXModel(optimize_model(Path(model_path)))
        # to support GEMM
        onnx_model.replace_gemm_with_matmul()
        return onnx_model.model

    return onnx.load(Path(model_path))


def quantize(model,
             per_channel=False,
             nbits=8,
             quantization_mode=QuantizationMode.IntegerOps,
             static=False,
             force_fusions=False,
             symmetric_activation=False,
             symmetric_weight=False,
             quantization_params=None,
             nodes_to_quantize=None,
             nodes_to_exclude=None,
             op_types_to_quantize=[]):
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
    :param op_types_to_quantize: specify the types of operators to quantize, like ['Conv'] to quantize Conv only. It quantizes all supported operators by default.
    :return: ModelProto with quantization
    '''
    logging.warning("onnxruntime.quantization.quantize is deprecated.\n\
         Please use quantize_static for static quantization, quantize_dynamic for dynamic quantization.")
    if nbits == 8 or nbits == 7:
        mode = quantization_mode
        copy_model = onnx_proto.ModelProto()
        copy_model.CopyFrom(model)

        if not op_types_to_quantize or len(op_types_to_quantize) == 0:
            op_types_to_quantize = list(QLinearOpsRegistry.keys()) if static else list(IntegerOpsRegistry.keys())

        quantizer = ONNXQuantizer(copy_model, per_channel, nbits == 7, mode, static, symmetric_weight,
                                  symmetric_activation, quantization_params, nodes_to_quantize, nodes_to_exclude,
                                  op_types_to_quantize)

        quantizer.quantize_model()
        return quantizer.model.model
    else:
        raise ValueError('Only 8 and 7 bit quantization is currently supported')


def quantize_static(model_input,
                    model_output,
                    calibration_data_reader: CalibrationDataReader,
                    quant_format=QuantFormat.QOperator,
                    op_types_to_quantize=[],
                    per_channel=False,
                    reduce_range=False,
                    activation_type=QuantType.QUInt8,
                    weight_type=QuantType.QUInt8,
                    nodes_to_quantize=[],
                    nodes_to_exclude=[],
                    optimize_model=True,
                    use_external_data_format=False,
                    calibrate_method=CalibrationMethod.MinMax,
                    extra_options = {}):

    '''
        Given an onnx model and calibration data reader, create a quantized onnx model and save it into a file
    :param model_input: file path of model to quantize
    :param model_output: file path of quantized model
    :param calibration_data_reader: a calibration data reader. It enumerates calibration data and generates inputs for the original model.
    :param quant_format: QuantFormat{QOperator, QDQ}.
        QOperator format quantizes the model with quantized operators directly.
        QDQ format quantize the model by inserting QuantizeLinear/DeQuantizeLinear on the tensor.
    :param op_types_to_quantize: specify the types of operators to quantize, like ['Conv'] to quantize Conv only. It quantizes all supported operators by default.
    :param op_types: operators to quantize
    :param per_channel: quantize weights per channel
    :param reduce_range: quantize weights with 7-bits. It may improve the accuracy for some models running on non-VNNI machine, especially for per-channel mode
    :param activation_type: quantization data type of activation
    :param weight_type: quantization data type of weight
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
    :param optimize_model: optimize model before quantization.
    :parma use_external_data_format: option used for large size (>2GB) model. Set to False by default. 
    :param calibrate_method: 
        Current calibration methods supported are MinMax and Entropy. 
        Please use CalibrationMethod.MinMax or CalibrationMethod.Entropy as options.
    :param extra_options:
        key value pair dictionary for various options in different case. Current used:
            extra.Sigmoid.nnapi = True  (Default is False)
    '''

    if activation_type != QuantType.QUInt8:
        raise ValueError("Static quantization only support uint8 for activation now.")

    mode = QuantizationMode.QLinearOps

    if not op_types_to_quantize or len(op_types_to_quantize) == 0:
        op_types_to_quantize = list(QLinearOpsRegistry.keys())

    model = load_model(Path(model_input), optimize_model)

    calibrator = create_calibrator(model, op_types_to_quantize, calibrate_method=calibrate_method)
    calibrator.collect_data(calibration_data_reader)
    tensors_range = calibrator.compute_range()

    if quant_format is QuantFormat.QOperator:
        quantizer = ONNXQuantizer(
            model,
            per_channel,
            reduce_range,
            mode,
            True,  # static
            weight_type,
            activation_type,
            tensors_range,
            nodes_to_quantize,
            nodes_to_exclude,
            op_types_to_quantize,
            extra_options)
    else:
        quantizer = QDQQuantizer(
            model,
            per_channel,
            reduce_range,
            mode,
            True,  # static
            weight_type,
            activation_type,
            tensors_range,
            nodes_to_quantize,
            nodes_to_exclude,
            op_types_to_quantize,
            extra_options)

    quantizer.quantize_model()
    quantizer.model.save_model_to_file(model_output, use_external_data_format)


def quantize_dynamic(model_input: Path,
                     model_output: Path,
                     op_types_to_quantize=[],
                     per_channel=False,
                     reduce_range=False,
                     activation_type=QuantType.QUInt8,
                     weight_type=QuantType.QUInt8,
                     nodes_to_quantize=[],
                     nodes_to_exclude=[],
                     optimize_model=True,
                     use_external_data_format=False,
                     extra_options = { }):
    '''
        Given an onnx model, create a quantized onnx model and save it into a file
    :param model_input: file path of model to quantize
    :param model_output: file path of quantized model
    :param op_types_to_quantize: specify the types of operators to quantize, like ['Conv'] to quantize Conv only. It quantizes all supported operators by default
    :param per_channel: quantize weights per channel
    :param reduce_range: quantize weights with 7-bits. It may improve the accuracy for some models running on non-VNNI machine, especially for per-channel mode
    :param nbits: number of bits to represent quantized data. Currently only supporting 8-bit types
    :param activation_type: quantization data type of activation
    :param weight_type: quantization data type of weight
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
    :parma use_external_data_format: option used for large size (>2GB) model. Set to False by default. 
    '''

    mode = QuantizationMode.IntegerOps

    if not op_types_to_quantize or len(op_types_to_quantize) == 0:
        op_types_to_quantize = list(IntegerOpsRegistry.keys())

    model = load_model(Path(model_input), optimize_model)
    quantizer = ONNXQuantizer(
        model,
        per_channel,
        reduce_range,
        mode,
        False,  #static
        weight_type,
        activation_type,
        None,
        nodes_to_quantize,
        nodes_to_exclude,
        op_types_to_quantize,
        extra_options)

    quantizer.quantize_model()
    quantizer.model.save_model_to_file(model_output, use_external_data_format)


def quantize_qat(model_input: Path,
                 model_output: Path,
                 op_types_to_quantize=[],
                 per_channel=False,
                 reduce_range=False,
                 activation_type=QuantType.QUInt8,
                 weight_type=QuantType.QUInt8,
                 nodes_to_quantize=[],
                 nodes_to_exclude=[],
                 use_external_data_format=False):
    '''
        Given a quantize-aware traning onnx model, create a quantized onnx model and save it into a file
    :param model_input: file path of model to quantize
    :param model_output: file path of quantized model
    :param op_types_to_quantize: specify the types of operators to quantize, like ['Conv'] to quantize Conv only. It quantizes all supported operators by default
    :param per_channel: quantize weights per channel
    :param reduce_range: quantize weights with 7-bits. It may improve the accuracy for some models running on non-VNNI machine, especially for per-channel mode
    :param activation_type: quantization data type of activation
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
    :parma use_external_data_format: option used for large size (>2GB) model. Set to False by default. 
    '''

    mode = QuantizationMode.IntegerOps

    #optimize the original model
    optimized_model = optimize_model(Path(model_input))

    if not op_types_to_quantize or len(op_types_to_quantize) == 0:
        op_types_to_quantize = list(IntegerOpsRegistry.keys())

    quantizer = ONNXQuantizer(
        optimized_model,
        per_channel,
        reduce_range,
        mode,
        False,  #static
        weight_type,
        activation_type,
        None,
        nodes_to_quantize,
        nodes_to_exclude,
        op_types_to_quantize)

    quantizer.quantize_model()
    quantizer.model.save_model_to_file(model_output, use_external_data_format)
