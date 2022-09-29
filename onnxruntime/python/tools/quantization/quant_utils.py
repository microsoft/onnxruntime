import logging
import tempfile
from enum import Enum
from pathlib import Path

import numpy
import onnx
from onnx import external_data_helper
from onnx import onnx_pb as onnx_proto

from onnxruntime import GraphOptimizationLevel, InferenceSession, SessionOptions

__producer__ = "onnx.quantize"
__version__ = "0.1.0"
onnx_domain = "ai.onnx"
ms_domain = "com.microsoft"
QUANT_OP_NAME = "QuantizeLinear"
QUANT_INPUT_SUFFIX = "_QuantizeLinear_Input"
DEQUANT_OP_NAME = "DequantizeLinear"
DEQUANT_OUTPUT_SUFFIX = "_DequantizeLinear_Output"
TENSOR_NAME_QUANT_SUFFIX = "_quantized"


type_to_name = {
    1: "FLOAT",
    2: "UINT8",
    3: "INT8",
    4: "UINT16",
    5: "INT16",
    6: "INT32",
    7: "INT64",
    8: "STRING",
    9: "BOOL",
    10: "FLOAT16",
    11: "DOUBLE",
    12: "UINT32",
    13: "UINT64",
    14: "COMPLEX64",
    15: "COMPLEX128",
}

# Quantization mode
# IntegerOps: Use IntegerOps in quantized model. Only ConvInteger and MatMulInteger ops are supported now.
# QLinearOps: Use QLinearOps in quantized model. Only QLinearConv and QLinearMatMul ops are supported now.


class QuantizationMode(Enum):
    IntegerOps = 0
    QLinearOps = 1

    def __str__(self):
        return self.name

    @staticmethod
    def from_string(mode):
        try:
            return QuantizationMode[mode]
        except KeyError:
            raise ValueError()


class QuantizedValueType(Enum):
    Input = 0
    Initializer = 1

    def __str__(self):
        return self.name

    @staticmethod
    def from_string(v):
        try:
            return QuantizedValueType[v]
        except KeyError:
            raise ValueError()


class QuantType(Enum):
    QInt8 = 0
    QUInt8 = 1

    def __str__(self):
        return self.name

    @staticmethod
    def from_string(t):
        try:
            return QuantType[t]
        except KeyError:
            raise ValueError()


class QuantFormat(Enum):
    QOperator = 0
    QDQ = 1

    def __str__(self):
        return self.name

    @staticmethod
    def from_string(format):
        try:
            return QuantFormat[format]
        except KeyError:
            raise ValueError()


ONNX_TYPE_TO_NP_TYPE = {
    onnx_proto.TensorProto.INT8: numpy.dtype("int8"),
    onnx_proto.TensorProto.UINT8: numpy.dtype("uint8"),
}


def quantize_nparray(qType, arr, scale, zero_point, low=None, high=None):
    assert (
        qType in ONNX_TYPE_TO_NP_TYPE
    ), "Unexpected data type {} requested. Only INT8 and UINT8 are supported.".format(qType)
    dtype = ONNX_TYPE_TO_NP_TYPE[qType]
    cliplow = max(0 if dtype == numpy.uint8 else -127, -127 if low is None else low)
    cliphigh = min(255 if dtype == numpy.uint8 else 127, 255 if high is None else high)
    arr_fp32 = numpy.asarray((arr.astype(numpy.float32) / scale).round() + zero_point)
    numpy.clip(arr_fp32, cliplow, cliphigh, out=arr_fp32)
    return arr_fp32.astype(dtype)


def compute_scale_zp(rmin, rmax, qmin, qmax, symmetric=False):
    """Calculate the scale s and zero point z for the quantization relation
    r = s(q-z), where r are the original values and q are the corresponding
    quantized values.

    r and z are calculated such that every value within [rmin,rmax] has an
    approximate representation within [qmin,qmax]. In addition, qmin <= z <=
    qmax is enforced. If the symmetric flag is set to True, the interval
    [rmin,rmax] is symmetrized to [-absmax, +absmax], where
    absmax = max(abs(rmin), abs(rmax)).

    :parameter rmin: minimum value of r
    :parameter rmax: maximum value of r
    :parameter qmin: minimum value representable by the target quantization data type
    :parameter qmax: maximum value representable by the target quantization data type
    :return: zero and scale [z, s]

    """

    if qmin > 0 or qmax < 0:
        raise ValueError(f"qmin and qmax must meet requirement: qmin <= 0 <= qmax while qmin:{qmin}, qmmax:{qmax}")

    # Adjust rmin and rmax such that 0 is included in the range. This is
    # required to make sure zero can be represented by the quantization data
    # type (i.e. to make sure qmin <= zero_point <= qmax)
    rmin = min(rmin, 0)
    rmax = max(rmax, 0)

    if symmetric:
        absmax = max(abs(rmin), abs(rmax))
        rmin = -absmax
        rmax = +absmax

    scale = (rmax - rmin) / float(qmax - qmin)
    if scale < numpy.finfo(numpy.float32).tiny:
        scale = 1.0
        zero_point = 0
    else:
        zero_point = round(qmin - rmin / scale)

    return [zero_point, scale]


def quantize_data(data, qType, symmetric, reduce_range=False):
    """
    :param data: data to quantize
    :param qType: data type to quantize to. Supported types UINT8 and INT8
    :param symmetric: whether symmetric quantization is used or not. This is applied to INT8.
    :return: minimum, maximum, zero point, scale, and quantized weights

    To pack weights, we compute a linear transformation

    - when data `type == uint8` mode, from `[rmin, rmax]` -> :math:`[0, 2^{b-1}]` and
    - when data `type == int8`, from `[-m , m]` -> :math:`[-(2^{b-1}-1), 2^{b-1}-1]` where
        `m = max(abs(rmin), abs(rmax))`

    and add necessary intermediate nodes to trasnform quantized weight to full weight using the equation

    :math:`r = S(q-z)`, where

    - *r*: real original value
    - *q*: quantized value
    - *S*: scale
    - *z*: zero point
    """

    rmin = 0
    rmax = 0
    zero_point = 0
    scale = 1.0
    if len(data):
        rmin = min(data)
        rmax = max(data)
        qmin, qmax = get_qmin_qmax_for_qType(qType, reduce_range, symmetric=symmetric)

        zero_point, scale = compute_scale_zp(rmin, rmax, qmin, qmax, symmetric)

    quantized_data = quantize_nparray(qType, numpy.asarray(data), scale, zero_point)

    return rmin, rmax, zero_point, scale, quantized_data


def get_qmin_qmax_for_qType(qType, reduce_range=False, symmetric=False):
    """
    Return qmin and qmax, the minimum and maximum value representable by the given qType
    :parameter qType: onnx.onnx_pb.TensorProto.UINT8 or onnx.onnx_pb.TensorProto.UINT8
    :return: qmin, qmax
    """
    if qType == onnx_proto.TensorProto.UINT8:
        (qmin, qmax) = (0, 127) if reduce_range else (0, 255)
    elif qType == onnx_proto.TensorProto.INT8:
        if symmetric:
            (qmin, qmax) = (-64, 64) if reduce_range else (-127, 127)
        else:
            (qmin, qmax) = (-64, 64) if reduce_range else (-128, 127)
    else:
        raise ValueError("Unexpected data type {} requested. Only INT8 and UINT8 are supported.".format(qType))
    return qmin, qmax


def get_qrange_for_qType(qType, reduce_range=False, symmetric=False):
    """
    Helper function to get the quantization range for a type.
        parameter qType: quantization type.
        return: quantization range.
    """
    qmin, qmax = get_qmin_qmax_for_qType(qType, reduce_range, symmetric=symmetric)
    return qmax - qmin


class QuantizedInitializer:
    """
    Represents a linearly quantized weight input from ONNX operators
    """

    def __init__(
        self,
        name,
        initializer,
        rmins,
        rmaxs,
        zero_points,
        scales,
        data=[],
        quantized_data=[],
        axis=None,
    ):
        self.name = name
        self.initializer = initializer  # TensorProto initializer in ONNX graph
        self.rmins = rmins  # List of minimum range for each axis
        self.rmaxs = rmaxs  # List of maximum range for each axis
        # 1D tensor of zero points computed for each axis. scalar if axis is empty
        self.zero_points = zero_points
        self.scales = scales  # 1D tensor of scales computed for each axis. scalar if axis is empty
        self.data = data  # original data from initializer TensorProto
        self.quantized_data = quantized_data  # weight-packed data from data
        # Scalar to specify which dimension in the initializer to weight pack.
        self.axis = axis
        # If empty, single zero point and scales computed from a single rmin and rmax


class QuantizedValue:
    """
    Represents a linearly quantized value (input\output\intializer)
    """

    def __init__(
        self,
        name,
        new_quantized_name,
        scale_name,
        zero_point_name,
        quantized_value_type,
        axis=None,
    ):
        self.original_name = name
        self.q_name = new_quantized_name
        self.scale_name = scale_name
        self.zp_name = zero_point_name
        self.value_type = quantized_value_type
        self.axis = axis


class BiasToQuantize:
    """
    Represents a bias to be quantized
    """

    def __init__(self, bias_name, input_name, weight_name):
        self.bias_name = bias_name
        self.input_name = input_name
        self.weight_name = weight_name


def attribute_to_kwarg(attribute):
    """
    Convert attribute to kwarg format for use with onnx.helper.make_node.
        :parameter attribute: attribute in AttributeProto format.
        :return: attribute in {key: value} format.
    """
    if attribute.type == 0:
        raise ValueError("attribute {} does not have type specified.".format(attribute.name))

    # Based on attribute type definitions from AttributeProto
    # definition in https://github.com/onnx/onnx/blob/main/onnx/onnx.proto
    if attribute.type == 1:
        value = attribute.f
    elif attribute.type == 2:
        value = attribute.i
    elif attribute.type == 3:
        value = attribute.s
    elif attribute.type == 4:
        value = attribute.t
    elif attribute.type == 5:
        value = attribute.g
    elif attribute.type == 6:
        value = attribute.floats
    elif attribute.type == 7:
        value = attribute.ints
    elif attribute.type == 8:
        value = attribute.strings
    elif attribute.type == 9:
        value = attribute.tensors
    elif attribute.type == 10:
        value = attribute.graphs
    else:
        raise ValueError("attribute {} has unsupported type {}.".format(attribute.name, attribute.type))

    return {attribute.name: value}


def find_by_name(item_name, item_list):
    """
    Helper function to find item by name in a list.
        parameter item_name: name of the item.
        parameter item_list: list of items.
        return: item if found. None otherwise.
    """
    items = [item for item in item_list if item.name == item_name]
    return items[0] if len(items) > 0 else None


def get_elem_index(elem_name, elem_list):
    """
    Helper function to return index of an item in a node list
    """
    elem_idx = -1
    for i in range(0, len(elem_list)):
        if elem_list[i] == elem_name:
            elem_idx = i
    return elem_idx


def get_mul_node(inputs, output, name):
    """
    Helper function to create a Mul node.
        parameter inputs: list of input names.
        parameter output: output name.
        parameter name: name of the node.
        return: Mul node in NodeProto format.
    """
    return onnx.helper.make_node("Mul", inputs, [output], name)


def generate_identified_filename(filename: Path, identifier: str) -> Path:
    """
    Helper function to generate a identifiable filepath by concatenating the given identifier as a suffix.
    """
    return filename.parent.joinpath(filename.stem + identifier + filename.suffix)


def apply_plot(hist, hist_edges):
    import sys

    import matplotlib.pyplot as plt
    import numpy

    numpy.set_printoptions(threshold=sys.maxsize)
    print("Histogram:")
    print(hist)
    print("Histogram Edges:")
    print(hist_edges)
    plt.stairs(hist, hist_edges, fill=True)
    plt.xlabel("Tensor value")
    plt.ylabel("Counts")
    plt.title("Tensor value V.S. Counts")
    plt.show()


def write_calibration_table(calibration_cache):
    """
    Helper function to write calibration table to files.
    """

    import json

    import flatbuffers

    import onnxruntime.quantization.CalTableFlatBuffers.KeyValue as KeyValue
    import onnxruntime.quantization.CalTableFlatBuffers.TrtTable as TrtTable

    logging.info("calibration cache: {}".format(calibration_cache))

    with open("calibration.json", "w") as file:
        file.write(json.dumps(calibration_cache))  # use `json.loads` to do the reverse

    # Serialize data using FlatBuffers
    builder = flatbuffers.Builder(1024)
    key_value_list = []
    for key in sorted(calibration_cache.keys()):
        values = calibration_cache[key]
        value = str(max(abs(values[0]), abs(values[1])))

        flat_key = builder.CreateString(key)
        flat_value = builder.CreateString(value)

        KeyValue.KeyValueStart(builder)
        KeyValue.KeyValueAddKey(builder, flat_key)
        KeyValue.KeyValueAddValue(builder, flat_value)
        key_value = KeyValue.KeyValueEnd(builder)

        key_value_list.append(key_value)

    TrtTable.TrtTableStartDictVector(builder, len(key_value_list))
    for key_value in key_value_list:
        builder.PrependUOffsetTRelative(key_value)
    main_dict = builder.EndVector()

    TrtTable.TrtTableStart(builder)
    TrtTable.TrtTableAddDict(builder, main_dict)
    cal_table = TrtTable.TrtTableEnd(builder)

    builder.Finish(cal_table)
    buf = builder.Output()

    with open("calibration.flatbuffers", "wb") as file:
        file.write(buf)

    # Deserialize data (for validation)
    if False:
        cal_table = TrtTable.TrtTable.GetRootAsTrtTable(buf, 0)
        dict_len = cal_table.DictLength()
        for i in range(dict_len):
            key_value = cal_table.Dict(i)
            logging.info(key_value.Key())
            logging.info(key_value.Value())

    # write plain text
    with open("calibration.cache", "w") as file:
        for key in sorted(calibration_cache.keys()):
            value = calibration_cache[key]
            s = key + " " + str(max(abs(value[0]), abs(value[1])))
            file.write(s)
            file.write("\n")


def smooth_distribution(p, eps=0.0001):
    """Given a discrete distribution (may have not been normalized to 1),
    smooth it by replacing zeros with eps multiplied by a scaling factor
    and taking the corresponding amount off the non-zero values.
    Ref: http://web.engr.illinois.edu/~hanj/cs412/bk3/KL-divergence.pdf
         https://github.com//apache/incubator-mxnet/blob/master/python/mxnet/contrib/quantization.py
    """
    import numpy as np

    is_zeros = (p == 0).astype(np.float32)
    is_nonzeros = (p != 0).astype(np.float32)
    n_zeros = is_zeros.sum()
    n_nonzeros = p.size - n_zeros

    if not n_nonzeros:
        # raise ValueError('The discrete probability distribution is malformed. All entries are 0.')
        return -1
    eps1 = eps * float(n_zeros) / float(n_nonzeros)
    assert eps1 < 1.0, "n_zeros=%d, n_nonzeros=%d, eps1=%f" % (
        n_zeros,
        n_nonzeros,
        eps1,
    )

    hist = p.astype(np.float32)
    hist += eps * is_zeros + (-eps1) * is_nonzeros
    assert (hist <= 0).sum() == 0

    return hist


def model_has_external_data(model_path: Path):
    model = onnx.load(model_path.as_posix(), load_external_data=False)
    for intializer in model.graph.initializer:
        if external_data_helper.uses_external_data(intializer):
            return True
    return False


def optimize_model(model_path: Path, opt_model_path: Path):
    """
        Generate model that applies graph optimization (constant folding, etc.)
        parameter model_path: path to the original onnx model
        parameter opt_model_path: path to the optimized onnx model
    :return: optimized onnx model
    """
    sess_option = SessionOptions()
    sess_option.optimized_model_filepath = opt_model_path.as_posix()
    sess_option.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_BASIC
    _ = InferenceSession(model_path.as_posix(), sess_option, providers=["CPUExecutionProvider"])


def add_pre_process_metadata(model):
    """Tag the model that it went through quantization pre-processing"""
    metadata_props = {"onnx.quant.pre_process": "onnxruntime.quant"}
    if model.metadata_props:
        for prop in model.metadata_props:
            metadata_props.update({prop.key: prop.value})
    onnx.helper.set_model_props(model, metadata_props)


def model_has_pre_process_metadata(model):
    """Check the model whether it went through quantization pre-processing"""
    if model.metadata_props:
        for prop in model.metadata_props:
            if prop.key == "onnx.quant.pre_process" and prop.value == "onnxruntime.quant":
                return True
    return False


def add_infer_metadata(model):
    metadata_props = {"onnx.infer": "onnxruntime.quant"}
    if model.metadata_props:
        for p in model.metadata_props:
            metadata_props.update({p.key: p.value})
    onnx.helper.set_model_props(model, metadata_props)


def model_has_infer_metadata(model):
    if model.metadata_props:
        for p in model.metadata_props:
            if p.key == "onnx.infer" and p.value == "onnxruntime.quant":
                return True
    return False


def load_model_with_shape_infer(model_path: Path):
    inferred_model_path = generate_identified_filename(model_path, "-inferred")
    onnx.shape_inference.infer_shapes_path(str(model_path), str(inferred_model_path))
    model = onnx.load(inferred_model_path.as_posix())
    inferred_model_path.unlink()
    return model


def load_model(model_path: Path, need_optimize: bool):
    with tempfile.TemporaryDirectory(prefix="ort.quant.") as quant_tmp_dir:
        if need_optimize and not model_has_external_data(model_path):
            opt_model_path = Path(quant_tmp_dir).joinpath("model.onnx")
            optimize_model(model_path, opt_model_path)
            model_path = opt_model_path

        model = load_model_with_shape_infer(model_path)
        add_infer_metadata(model)
        return model


def save_and_reload_model(model):
    with tempfile.TemporaryDirectory(prefix="ort.quant.") as quant_tmp_dir:
        model_path = Path(quant_tmp_dir).joinpath("model.onnx")
        onnx.external_data_helper.convert_model_to_external_data(model, all_tensors_to_one_file=True)
        onnx.save_model(model, model_path.as_posix())
        return load_model(model_path, False)


def clone_model_with_shape_infer(model):
    if model_has_infer_metadata(model):
        cloned_model = onnx_proto.ModelProto()
        cloned_model.CopyFrom(model)
    else:
        cloned_model = save_and_reload_model(model)
    return cloned_model


def tensor_proto_to_array(initializer):
    if initializer.data_type == onnx_proto.TensorProto.FLOAT:
        return onnx.numpy_helper.to_array(initializer)

    raise ValueError(
        f"Only float type is supported. Weights {initializer.name} is {type_to_name[initializer.data_type]}"
    )


def add_quant_suffix(tensor_name):
    return tensor_name + "_QuantizeLinear"


def add_quant_input_suffix(tensor_name):
    return tensor_name + QUANT_INPUT_SUFFIX


def add_quant_output_suffix(tensor_name):
    return tensor_name + "_QuantizeLinear_Output"


def add_dequant_suffix(tensor_name):
    return tensor_name + "_DequantizeLinear"


def add_dequant_input_suffix(tensor_name):
    return tensor_name + "_DequantizeLinear_Input"


def add_dequant_output_suffix(tensor_name):
    return tensor_name + DEQUANT_OUTPUT_SUFFIX
