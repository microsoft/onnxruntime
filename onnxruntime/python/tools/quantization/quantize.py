# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import logging
from pathlib import Path

from onnx import onnx_pb as onnx_proto

from .calibrate import CalibrationDataReader, CalibrationMethod, create_calibrator
from .onnx_model import ONNXModel
from .onnx_quantizer import ONNXQuantizer
from .qdq_quantizer import QDQQuantizer
from .quant_utils import (
    QuantFormat,
    QuantizationMode,
    QuantizedInitializer,
    QuantizedValue,
    QuantizedValueType,
    QuantType,
    attribute_to_kwarg,
    find_by_name,
    generate_identified_filename,
    get_elem_index,
    get_mul_node,
    load_model,
)
from .registry import IntegerOpsRegistry, QLinearOpsRegistry


def check_static_quant_arguments(quant_format: QuantFormat, activation_type: QuantType, weight_type: QuantType):
    if activation_type == QuantType.QInt8 and weight_type == QuantType.QUInt8:
        raise ValueError(
            "ONNXRuntime quantization doesn't support data format:"
            "activation_type=QuantType.QInt8, weight_type = QuantType.QUInt8"
        )

    if activation_type == QuantType.QInt8 and weight_type == QuantType.QInt8 and quant_format != QuantFormat.QDQ:
        logging.warning(
            "Please use QuantFormat.QDQ for activation type QInt8 and weight type QInt8. "
            "Or it will lead to bad performance on x64."
        )


def quantize_static(
    model_input,
    model_output,
    calibration_data_reader: CalibrationDataReader,
    quant_format=QuantFormat.QDQ,
    op_types_to_quantize=[],
    per_channel=False,
    reduce_range=False,
    activation_type=QuantType.QInt8,
    weight_type=QuantType.QInt8,
    nodes_to_quantize=[],
    nodes_to_exclude=[],
    optimize_model=True,
    use_external_data_format=False,
    calibrate_method=CalibrationMethod.MinMax,
    extra_options={},
):

    """
        Given an onnx model and calibration data reader, create a quantized onnx model and save it into a file

    It is recommended to use QuantFormat.QDQ format from 1.11 with activation_type = QuantType.QInt8 and
                                                                    weight_type = QuantType.QInt8.
    If model is targeted to GPU/TRT, symmetric activation and weight are required.
    If model is targeted to CPU, asymmetric activation and symmetric weight are recommended for balance of performance and accuracy.

    :param model_input: file path of model to quantize
    :param model_output: file path of quantized model
    :param calibration_data_reader: a calibration data reader. It enumerates calibration data and generates inputs for the original model.
    :param quant_format: QuantFormat{QOperator, QDQ}.
        QOperator format quantizes the model with quantized operators directly.
        QDQ format quantize the model by inserting QuantizeLinear/DeQuantizeLinear on the tensor.
    :param op_types_to_quantize: specify the types of operators to quantize, like ['Conv'] to quantize Conv only. It quantizes all supported operators by default.
    :param per_channel: quantize weights per channel
    :param reduce_range: quantize weights with 7-bits. It may improve the accuracy for some models running on non-VNNI machine, especially for per-channel mode
    :param activation_type: quantization data type of activation. Please refer to https://onnxruntime.ai/docs/performance/quantization.html for more details on data type selection
    :param weight_type: quantization data type of weight. Please refer to https://onnxruntime.ai/docs/performance/quantization.html for more details on data type selection
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
    :param use_external_data_format: option used for large size (>2GB) model. Set to False by default.
    :param calibrate_method:
        Current calibration methods supported are MinMax and Entropy.
        Please use CalibrationMethod.MinMax or CalibrationMethod.Entropy as options.
    :param extra_options:
        key value pair dictionary for various options in different case. Current used:
            extra.Sigmoid.nnapi = True/False  (Default is False)
            ActivationSymmetric = True/False: symmetrize calibration data for activations (default is False).
            WeightSymmetric = True/False: symmetrize calibration data for weights (default is True).
            EnableSubgraph = True/False : Default is False. If enabled, subgraph will be quantized.
                                          Dyanmic mode currently is supported. Will support more in future.
            ForceQuantizeNoInputCheck = True/False : By default, some latent operators like maxpool, transpose, do not quantize
                                                     if their input is not quantized already. Setting to True to force such operator
                                                     always quantize input and so generate quantized output. Also the True behavior
                                                     could be disabled per node using the nodes_to_exclude.
            MatMulConstBOnly = True/False: Default is False for static mode. If enabled, only MatMul with const B will be quantized.
            AddQDQPairToWeight = True/False : Default is False which quantizes floating-point weight and feeds it to
                                              soley inserted DeQuantizeLinear node. If True, it remains floating-point weight and
                                              inserts both QuantizeLinear/DeQuantizeLinear nodes to weight.
            OpTypesToExcludeOutputQuantizatioin = list of op type : Default is []. If any op type is specified, it won't quantize
                                                                    the output of ops with this specific op types.
            DedicatedQDQPair = True/False : Default is False. When inserting QDQ pair, multiple nodes can share a single QDQ pair as their inputs.
                                            If True, it will create identical and dedicated QDQ pair for each node.
            QDQOpTypePerChannelSupportToAxis = dictionary : Default is {}. Set channel axis for specific op type, for example: {'MatMul': 1},
                                                            and it's effective only when per channel quantization is supported and per_channel is True.
                                                            If specific op type supports per channel quantization but not explicitly specified with channel axis,
                                                            default channel axis will be used.
            CalibTensorRangeSymmetric = True/False : Default is False. If enabled, the final range of tensor during calibration will be explicitly set to symmetric to central point "0".
            CalibMovingAverage = True/False : Default is False. If enabled, the moving average of the minimum and maximum values
                                              will be computed when the calibration method selected is MinMax.
            CalibMovingAverageConstant = float : Default is 0.01. Constant smoothing factor to use when computing the moving average of
                                                 the minimum and maximum values. Effective only when the calibration method selected is
                                                 MinMax and when CalibMovingAverage is set to True.
    """

    mode = QuantizationMode.QLinearOps

    if not op_types_to_quantize or len(op_types_to_quantize) == 0:
        op_types_to_quantize = list(QLinearOpsRegistry.keys())

    model = load_model(Path(model_input), optimize_model)

    calib_extra_options_keys = [
        ("CalibTensorRangeSymmetric", "symmetric"),
        ("CalibMovingAverage", "moving_average"),
        ("CalibMovingAverageConstant", "averaging_constant"),
    ]
    calib_extra_options = {
        key: extra_options.get(name) for (name, key) in calib_extra_options_keys if name in extra_options
    }
    calibrator = create_calibrator(
        model,
        op_types_to_quantize,
        calibrate_method=calibrate_method,
        use_external_data_format=use_external_data_format,
        extra_options=calib_extra_options,
    )
    calibrator.collect_data(calibration_data_reader)
    tensors_range = calibrator.compute_range()

    check_static_quant_arguments(quant_format, activation_type, weight_type)

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
            extra_options,
        )
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
            extra_options,
        )

    quantizer.quantize_model()
    quantizer.model.save_model_to_file(model_output, use_external_data_format)


def quantize_dynamic(
    model_input: Path,
    model_output: Path,
    op_types_to_quantize=[],
    per_channel=False,
    reduce_range=False,
    weight_type=QuantType.QInt8,
    nodes_to_quantize=[],
    nodes_to_exclude=[],
    optimize_model=True,
    use_external_data_format=False,
    extra_options={},
):
    """
        Given an onnx model, create a quantized onnx model and save it into a file
    :param model_input: file path of model to quantize
    :param model_output: file path of quantized model
    :param op_types_to_quantize: specify the types of operators to quantize, like ['Conv'] to quantize Conv only. It quantizes all supported operators by default
    :param per_channel: quantize weights per channel
    :param reduce_range: quantize weights with 7-bits. It may improve the accuracy for some models running on non-VNNI machine, especially for per-channel mode
    :param weight_type: quantization data type of weight. Please refer to https://onnxruntime.ai/docs/performance/quantization.html for more details on data type selection
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
    :param use_external_data_format: option used for large size (>2GB) model. Set to False by default.
    :param extra_options:
        key value pair dictionary for various options in different case. Current used:
            extra.Sigmoid.nnapi = True/False  (Default is False)
            ActivationSymmetric = True/False: symmetrize calibration data for activations (default is False).
            WeightSymmetric = True/False: symmetrize calibration data for weights (default is True).
            EnableSubgraph = True/False : Default is False. If enabled, subgraph will be quantized.
                                          Dyanmic mode currently is supported. Will support more in future.
            ForceQuantizeNoInputCheck = True/False : By default, some latent operators like maxpool, transpose, do not quantize
                                                     if their input is not quantized already. Setting to True to force such operator
                                                     always quantize input and so generate quantized output. Also the True behavior
                                                     could be disabled per node using the nodes_to_exclude.
            MatMulConstBOnly = True/False: Default is True for dynamic mode. If enabled, only MatMul with const B will be quantized.
    """

    mode = QuantizationMode.IntegerOps

    if not op_types_to_quantize or len(op_types_to_quantize) == 0:
        op_types_to_quantize = list(IntegerOpsRegistry.keys())

    model = load_model(Path(model_input), optimize_model)

    if "MatMulConstBOnly" not in extra_options:
        extra_options["MatMulConstBOnly"] = True

    quantizer = ONNXQuantizer(
        model,
        per_channel,
        reduce_range,
        mode,
        False,  # static
        weight_type,
        QuantType.QUInt8,  # dynamic activation only supports uint8
        None,
        nodes_to_quantize,
        nodes_to_exclude,
        op_types_to_quantize,
        extra_options,
    )

    quantizer.quantize_model()
    quantizer.model.save_model_to_file(model_output, use_external_data_format)
