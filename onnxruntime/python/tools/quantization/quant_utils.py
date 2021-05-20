import logging
import numpy
import onnx

from enum import Enum
from onnx import onnx_pb as onnx_proto
from pathlib import Path

__producer__ = "onnx.quantize"
__version__ = "0.1.0"
onnx_domain = "ai.onnx"
ms_domain = "com.microsoft"

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
    onnx_proto.TensorProto.INT8: numpy.dtype('int8'),
    onnx_proto.TensorProto.UINT8:  numpy.dtype('uint8')
}

def quantize_nparray(qType, arr, scale, zero_point, low=None, high=None):
    assert qType in ONNX_TYPE_TO_NP_TYPE, \
        "Unexpected data type {} requested. Only INT8 and UINT8 are supported.".format(qType)
    dtype = ONNX_TYPE_TO_NP_TYPE[qType]
    cliplow = max(0 if dtype == numpy.uint8 else -127, -127 if low is None else low)
    cliphigh = min(255 if dtype == numpy.uint8 else 127, 255 if high is None else high)
    arr_fp32 = numpy.asarray((arr.astype(numpy.float32) / scale).round() + zero_point)
    numpy.clip(arr_fp32, cliplow, cliphigh, out=arr_fp32)
    return arr_fp32.astype(dtype)


def compute_scale_zp(rmin, rmax, qType, quantize_range, symmetric):
    if qType == onnx_proto.TensorProto.INT8:
        if symmetric:
            max_range = max(abs(rmin), abs(rmax))
            scale = (float(max_range) * 2) / quantize_range if max_range > 0 else 1.0
            zero_point = 0
        else:
            max_range = float(rmax) - float(rmin)
            scale = float(max_range) / quantize_range if max_range > 0 else 1.0
            zero_point = round((quantize_range / 2) - rmax / scale)
    elif qType == onnx_proto.TensorProto.UINT8:
        scale = (float(rmax) - rmin) / quantize_range if rmin != rmax else 1
        zero_point = round((0 - rmin) / scale)  # round to nearest integer
    else:
        raise ValueError("Unexpected data type {} requested. Only INT8 and UINT8 are supported.".format(qType))

    return [zero_point, scale]


def quantize_data(data, quantize_range, qType, symmetric=True):
    '''
        :parameter data: data to quantize
        :parameter quantize_range: list of data to weight pack.
        :parameter qType: data type to quantize to. Supported types UINT8 and INT8
        :parameter symmetric: whether symmetric quantization is used or not. This is applied to INT8.
        :return: minimum, maximum, zero point, scale, and quantized weights
        To pack weights, we compute a linear transformation
            - when data type == uint8 mode, from [rmin, rmax] -> [0, 2^{b-1}] and
            - when data type == int8, from [-m , m] -> [-(2^{b-1}-1), 2^{b-1}-1] where
                m = max(abs(rmin), abs(rmax))
        and add necessary intermediate nodes to trasnform quantized weight to full weight using the equation
        r = S(q-z), where
            r: real original value
            q: quantized value
            S: scale
            z: zero point
    '''
    rmin = min(min(data), 0.)
    rmax = max(max(data), 0.)

    zero_point, scale = compute_scale_zp(rmin, rmax, qType, quantize_range, symmetric)
    quantized_data = quantize_nparray(qType, numpy.asarray(data), scale, zero_point)

    return rmin, rmax, zero_point, scale, quantized_data


def get_qrange_for_qType(qType, reduce_range=False):
    '''
    Helper function to get the quantization range for a type.
        parameter qType: quantization type.
        return: quantization range.
    '''
    if qType == onnx_proto.TensorProto.UINT8:
        return 127 if reduce_range else 255
    elif qType == onnx_proto.TensorProto.INT8:
        return 128 if reduce_range else 254  # [-64, 64] for reduce_range, and [-127, 127] full_range.
    else:
        raise ValueError('unsupported quantization data type')


class QuantizedInitializer:
    '''
        Represents a linearly quantized weight input from ONNX operators
    '''
    def __init__(self,
                 name,
                 initializer,
                 rmins,
                 rmaxs,
                 zero_points,
                 scales,
                 data=[],
                 quantized_data=[],
                 axis=None):
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
    '''
    Represents a linearly quantized value (input\output\intializer)
    '''
    def __init__(self,
                 name,
                 new_quantized_name,
                 scale_name,
                 zero_point_name,
                 quantized_value_type,
                 axis=None):
        self.original_name = name
        self.q_name = new_quantized_name
        self.scale_name = scale_name
        self.zp_name = zero_point_name
        self.value_type = quantized_value_type
        self.axis = axis


class BiasToQuantize:
    '''
    Represents a bias to be quantized
    '''
    def __init__(self, bias_name, input_name, weight_name):
        self.bias_name = bias_name
        self.input_name = input_name
        self.weight_name = weight_name


def attribute_to_kwarg(attribute):
    '''
    Convert attribute to kwarg format for use with onnx.helper.make_node.
        :parameter attribute: attribute in AttributeProto format.
        :return: attribute in {key: value} format.
    '''
    if (attribute.type == 0):
        raise ValueError('attribute {} does not have type specified.'.format(attribute.name))

    # Based on attribute type definitions from AttributeProto
    # definition in https://github.com/onnx/onnx/blob/master/onnx/onnx.proto
    if (attribute.type == 1):
        value = attribute.f
    elif (attribute.type == 2):
        value = attribute.i
    elif (attribute.type == 3):
        value = attribute.s
    elif (attribute.type == 4):
        value = attribute.t
    elif (attribute.type == 5):
        value = attribute.g
    elif (attribute.type == 6):
        value = attribute.floats
    elif (attribute.type == 7):
        value = attribute.ints
    elif (attribute.type == 8):
        value = attribute.strings
    elif (attribute.type == 9):
        value = attribute.tensors
    elif (attribute.type == 10):
        value = attribute.graphs
    else:
        raise ValueError('attribute {} has unsupported type {}.'.format(attribute.name, attribute.type))

    return {attribute.name: value}


def find_by_name(item_name, item_list):
    '''
    Helper function to find item by name in a list.
        parameter item_name: name of the item.
        parameter item_list: list of items.
        return: item if found. None otherwise.
    '''
    items = [item for item in item_list if item.name == item_name]
    return items[0] if len(items) > 0 else None


def get_elem_index(elem_name, elem_list):
    '''
    Helper function to return index of an item in a node list
    '''
    elem_idx = -1
    for i in range(0, len(elem_list)):
        if elem_list[i] == elem_name:
            elem_idx = i
    return elem_idx


def get_mul_node(inputs, output, name):
    '''
    Helper function to create a Mul node.
        parameter inputs: list of input names.
        parameter output: output name.
        parameter name: name of the node.
        return: Mul node in NodeProto format.
    '''
    return onnx.helper.make_node("Mul", inputs, [output], name)


def generate_identified_filename(filename: Path, identifier: str) -> Path:
    '''
    Helper function to generate a identifiable filepath by concatenating the given identifier as a suffix.   
    '''
    return filename.parent.joinpath(filename.stem + identifier).with_suffix(filename.suffix)

def write_calibration_table(calibration_cache):
    '''
    Helper function to write calibration table to files.   
    '''

    import json
    import flatbuffers
    import onnxruntime.quantization.CalTableFlatBuffers.TrtTable as TrtTable
    import onnxruntime.quantization.CalTableFlatBuffers.KeyValue as KeyValue

    logging.info("calibration cache: {}".format(calibration_cache))

    with open("calibration.json", 'w') as file:
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
    main_dict = builder.EndVector(len(key_value_list))

    TrtTable.TrtTableStart(builder)
    TrtTable.TrtTableAddDict(builder, main_dict)
    cal_table = TrtTable.TrtTableEnd(builder)

    builder.Finish(cal_table)
    buf = builder.Output()

    with open("calibration.flatbuffers", 'wb') as file:
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
    with open("calibration.cache", 'w') as file:
        for key in sorted(calibration_cache.keys()):
            value = calibration_cache[key]
            s = key + ' ' + str(max(abs(value[0]), abs(value[1])))
            file.write(s)
            file.write('\n')

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
    assert eps1 < 1.0, 'n_zeros=%d, n_nonzeros=%d, eps1=%f' % (n_zeros, n_nonzeros, eps1)

    hist = p.astype(np.float32)
    hist += eps * is_zeros + (-eps1) * is_nonzeros
    assert (hist <= 0).sum() == 0

    return hist
