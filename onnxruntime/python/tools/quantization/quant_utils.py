import onnx
import numpy
from onnx import onnx_pb as onnx_proto
from enum import Enum

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


class QuantizationMode():
    IntegerOps = 0
    QLinearOps = 1


quantization_modes = [
    getattr(QuantizationMode, attr) for attr in dir(QuantizationMode)
    if not callable(getattr(QuantizationMode, attr)) and not attr.startswith("__")
]


class QuantizedValueType():
    Input = 0
    Initializer = 1


class QuantType(Enum):
    QInt8 = 1
    QUInt8 = 2


QUANT_TYPE_TO_NP_TYPE = {
    QuantType.QInt8: numpy.dtype('int8'),
    QuantType.QUInt8: numpy.dtype('uint8'),
}


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
                 axis=None,
                 qType=QuantType.QUInt8):
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
        self.qType = qType  # type of quantized data.


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
                 axis=None,
                 qType=QuantType.QUInt8):
        self.original_name = name
        self.q_name = new_quantized_name
        self.scale_name = scale_name
        self.zp_name = zero_point_name
        self.value_type = quantized_value_type
        self.axis = axis
        self.qType = qType


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

    print("calibration cache: ", calibration_cache)

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
            print(key_value.Key())
            print(key_value.Value())

    # write plain text
    with open("calibration.cache", 'w') as file:
        for key in sorted(calibration_cache.keys()):
            value = calibration_cache[key]
            s = key + ' ' + str(max(abs(value[0]), abs(value[1])))
            file.write(s)
            file.write('\n')
