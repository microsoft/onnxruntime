// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <functional>
#include <numeric>
#include <string>
#include <vector>
#include <map>

#include "core/common/common.h"
#include "core/framework/data_types.h"
#include "qnn_utils.h"
#include "core/providers/qnn/builder/qnn_def.h"

namespace onnxruntime {
namespace qnn {
namespace utils {

size_t GetElementSizeByType(const Qnn_DataType_t& data_type) {
  const static std::unordered_map<Qnn_DataType_t, size_t> data_type_to_size = {
      {QNN_DATATYPE_INT_8, 1},
      {QNN_DATATYPE_INT_16, 2},
      {QNN_DATATYPE_INT_32, 4},
      {QNN_DATATYPE_INT_64, 8},
      {QNN_DATATYPE_UINT_8, 1},
      {QNN_DATATYPE_UINT_16, 2},
      {QNN_DATATYPE_UINT_32, 4},
      {QNN_DATATYPE_UINT_64, 8},
      {QNN_DATATYPE_FLOAT_16, 2},
      {QNN_DATATYPE_FLOAT_32, 4},
      {QNN_DATATYPE_BOOL_8, 1},
      {QNN_DATATYPE_SFIXED_POINT_8, 1},
      {QNN_DATATYPE_SFIXED_POINT_16, 2},
      {QNN_DATATYPE_SFIXED_POINT_32, 4},
      {QNN_DATATYPE_UFIXED_POINT_8, 1},
      {QNN_DATATYPE_UFIXED_POINT_16, 2},
      {QNN_DATATYPE_UFIXED_POINT_32, 4},
  };

  auto pos = data_type_to_size.find(data_type);
  ORT_ENFORCE(pos != data_type_to_size.end(), "Unknown QNN data type", data_type);
  return pos->second;
}
size_t GetElementSizeByType(ONNXTensorElementDataType elem_type) {
  const static std::unordered_map<ONNXTensorElementDataType, size_t> elem_type_to_size = {
      {ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8, sizeof(int8_t)},
      {ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16, sizeof(int16_t)},
      {ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, sizeof(int32_t)},
      {ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, sizeof(int64_t)},
      {ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, sizeof(uint8_t)},
      {ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16, sizeof(uint16_t)},
      {ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32, sizeof(uint32_t)},
      {ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64, sizeof(uint64_t)},
      {ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16, 2},
      {ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, sizeof(float)},
      {ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE, sizeof(double)},
      {ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL, sizeof(bool)}};

  auto pos = elem_type_to_size.find(elem_type);
  ORT_ENFORCE(pos != elem_type_to_size.end(), "Unknown element type", elem_type);
  return pos->second;
}

std::ostream& operator<<(std::ostream& out, const Qnn_Scalar_t& scalar) {
  switch (scalar.dataType) {
    case QNN_DATATYPE_INT_8:
      out << static_cast<int32_t>(scalar.int8Value);
      break;
    case QNN_DATATYPE_INT_16:
      out << scalar.int16Value;
      break;
    case QNN_DATATYPE_INT_32:
      out << scalar.int32Value;
      break;
    case QNN_DATATYPE_INT_64:
      out << "int64_t is not supported";
      break;
    case QNN_DATATYPE_UINT_8:
      out << static_cast<int32_t>(scalar.uint8Value);
      break;
    case QNN_DATATYPE_UINT_16:
      out << scalar.uint16Value;
      break;
    case QNN_DATATYPE_UINT_32:
      out << scalar.uint32Value;
      break;
    case QNN_DATATYPE_UINT_64:
      out << "uint64_t is not supported";
      break;
    case QNN_DATATYPE_FLOAT_16:
      break;
    case QNN_DATATYPE_FLOAT_32:
      out << scalar.floatValue;
      break;
    case QNN_DATATYPE_SFIXED_POINT_8:
    case QNN_DATATYPE_SFIXED_POINT_16:
    case QNN_DATATYPE_SFIXED_POINT_32:
    case QNN_DATATYPE_UFIXED_POINT_8:
    case QNN_DATATYPE_UFIXED_POINT_16:
    case QNN_DATATYPE_UFIXED_POINT_32:
      out << "usigned fixedpoint data is not supported";
      break;
    case QNN_DATATYPE_BOOL_8:
      out << static_cast<int32_t>(scalar.bool8Value);
      break;
    default:
      ORT_THROW("Unknown Qnn Data type");
  }
  return out;
}

std::ostream& operator<<(std::ostream& out, const Qnn_DataType_t& data_type) {
  switch (data_type) {
    case QNN_DATATYPE_INT_8:
      out << "QNN_DATATYPE_INT_8";
      break;
    case QNN_DATATYPE_INT_16:
      out << "QNN_DATATYPE_INT_16";
      break;
    case QNN_DATATYPE_INT_32:
      out << "QNN_DATATYPE_INT_32";
      break;
    case QNN_DATATYPE_INT_64:
      out << "QNN_DATATYPE_INT_64";
      break;
    case QNN_DATATYPE_UINT_8:
      out << "QNN_DATATYPE_UINT_8";
      break;
    case QNN_DATATYPE_UINT_16:
      out << "QNN_DATATYPE_UINT_16";
      break;
    case QNN_DATATYPE_UINT_32:
      out << "QNN_DATATYPE_UINT_32";
      break;
    case QNN_DATATYPE_UINT_64:
      out << "QNN_DATATYPE_UINT_64";
      break;
    case QNN_DATATYPE_FLOAT_16:
      out << "QNN_DATATYPE_FLOAT_16";
      break;
    case QNN_DATATYPE_FLOAT_32:
      out << "QNN_DATATYPE_FLOAT_32";
      break;
    case QNN_DATATYPE_SFIXED_POINT_8:
      out << "QNN_DATATYPE_SFIXED_POINT_8";
      break;
    case QNN_DATATYPE_SFIXED_POINT_16:
      out << "QNN_DATATYPE_SFIXED_POINT_16";
      break;
    case QNN_DATATYPE_SFIXED_POINT_32:
      out << "QNN_DATATYPE_SFIXED_POINT_32";
      break;
    case QNN_DATATYPE_UFIXED_POINT_8:
      out << "QNN_DATATYPE_UFIXED_POINT_8";
      break;
    case QNN_DATATYPE_UFIXED_POINT_16:
      out << "QNN_DATATYPE_UFIXED_POINT_16";
      break;
    case QNN_DATATYPE_UFIXED_POINT_32:
      out << "QNN_DATATYPE_UFIXED_POINT_32";
      break;
    case QNN_DATATYPE_BOOL_8:
      out << "QNN_DATATYPE_BOOL_8";
      break;
    default:
      ORT_THROW("Unknown Qnn Data type");
  }
  return out;
}

std::ostream& operator<<(std::ostream& out, const Qnn_Definition_t& definition) {
  switch (definition) {
    case QNN_DEFINITION_IMPL_GENERATED:
      out << "QNN_DEFINITION_IMPL_GENERATED";
      break;
    case QNN_DEFINITION_DEFINED:
      out << "QNN_DEFINITION_DEFINED";
      break;
    case QNN_DEFINITION_UNDEFINED:
      out << "QNN_DEFINITION_UNDEFINED";
      break;
    default:
      out << "Undefined";
  }
  return out;
}

std::ostream& operator<<(std::ostream& out, const Qnn_QuantizationEncoding_t& encoding) {
  switch (encoding) {
    case QNN_QUANTIZATION_ENCODING_SCALE_OFFSET:
      out << "QNN_QUANTIZATION_ENCODING_SCALE_OFFSET";
      break;
    case QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET:
      out << "QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET";
      break;
    case QNN_QUANTIZATION_ENCODING_BW_SCALE_OFFSET:
      out << "QNN_QUANTIZATION_ENCODING_BW_SCALE_OFFSET";
      break;
    case QNN_QUANTIZATION_ENCODING_BW_AXIS_SCALE_OFFSET:
      out << "QNN_QUANTIZATION_ENCODING_BW_AXIS_SCALE_OFFSET";
      break;
    case QNN_QUANTIZATION_ENCODING_UNDEFINED:
      out << "QNN_QUANTIZATION_ENCODING_UNDEFINED";
      break;
    default:
      out << "Uknown quantization encoding";
  }
  return out;
}

std::ostream& operator<<(std::ostream& out, const Qnn_QuantizeParams_t& quantize_params) {
  out << " encodingDefinition=" << quantize_params.encodingDefinition;
  out << " quantizationEncoding=" << quantize_params.quantizationEncoding;
  if (quantize_params.encodingDefinition == QNN_DEFINITION_IMPL_GENERATED ||
      quantize_params.encodingDefinition == QNN_DEFINITION_DEFINED) {
    if (quantize_params.quantizationEncoding == QNN_QUANTIZATION_ENCODING_SCALE_OFFSET) {
      out << " scale=" << quantize_params.scaleOffsetEncoding.scale;
      out << " offset=" << quantize_params.scaleOffsetEncoding.offset;
    } else {
      out << " encoding not supported.";
    }
  }
  return out;
}

std::ostream& operator<<(std::ostream& out, const Qnn_TensorType_t& tensor_type) {
  switch (tensor_type) {
    case QNN_TENSOR_TYPE_APP_WRITE:
      out << "QNN_TENSOR_TYPE_APP_WRITE";
      break;
    case QNN_TENSOR_TYPE_APP_READ:
      out << "QNN_TENSOR_TYPE_APP_READ";
      break;
    case QNN_TENSOR_TYPE_APP_READWRITE:
      out << "QNN_TENSOR_TYPE_APP_READWRITE";
      break;
    case QNN_TENSOR_TYPE_NATIVE:
      out << "QNN_TENSOR_TYPE_NATIVE";
      break;
    case QNN_TENSOR_TYPE_STATIC:
      out << "QNN_TENSOR_TYPE_STATIC";
      break;
    case QNN_TENSOR_TYPE_NULL:
      out << "QNN_TENSOR_TYPE_NULL";
      break;
    default:
      out << "Unsupported type";
  }
  return out;
}

std::ostream& operator<<(std::ostream& out, const Qnn_TensorMemType_t& mem_type) {
  switch (mem_type) {
    case QNN_TENSORMEMTYPE_RAW:
      out << "QNN_TENSORMEMTYPE_RAW";
      break;
    case QNN_TENSORMEMTYPE_MEMHANDLE:
      out << "QNN_TENSORMEMTYPE_MEMHANDLE";
      break;
    default:
      out << "Unsupported mem type";
  }
  return out;
}
template <typename T>
std::ostream& operator<<(std::ostream& out, const Qnn_ClientBuffer_t& client_bufer) {
  T* data = reinterpret_cast<T*>(client_bufer.data);
  out << " dataSize=" << client_bufer.dataSize;
  uint32_t count = client_bufer.dataSize / sizeof(T);
  count = count > 100 ? 100 : count;  // limit to 100 data
  out << " clientBuf=(";
  for (uint32_t i = 0; i < count; i++) {
    if constexpr (sizeof(T) == 1) {
      out << static_cast<int32_t>(data[i]) << " ";
    } else {
      out << data[i] << " ";
    }
  }
  out << ")";
  return out;
}

std::ostream& operator<<(std::ostream& out, const Qnn_Tensor_t& tensor) {
  out << " name=" << GetQnnTensorName(tensor);
  out << " id=" << GetQnnTensorID(tensor);
  out << " version=" << tensor.version;
  out << " type=" << GetQnnTensorType(tensor);
  out << " dataFormat=" << GetQnnTensorDataFormat(tensor);
  out << " dataType=" << GetQnnTensorDataType(tensor);
  out << " rank=" << GetQnnTensorRank(tensor);
  out << " dimensions=(";
  for (uint32_t i = 0; i < GetQnnTensorRank(tensor); i++) {
    out << GetQnnTensorDims(tensor)[i] << " ";
  }
  out << ")";
  out << " memType=" << GetQnnTensorMemType(tensor);
  if (GetQnnTensorMemType(tensor) == QNN_TENSORMEMTYPE_RAW) {
    if (GetQnnTensorDataType(tensor) == QNN_DATATYPE_FLOAT_32) {
      operator<< <float>(out, GetQnnTensorClientBuf(tensor));
    } else if (GetQnnTensorDataType(tensor) == QNN_DATATYPE_UINT_32 ||
               GetQnnTensorDataType(tensor) == QNN_DATATYPE_UFIXED_POINT_32) {
      operator<< <uint32_t>(out, GetQnnTensorClientBuf(tensor));
    } else if (GetQnnTensorDataType(tensor) == QNN_DATATYPE_INT_32 ||
               GetQnnTensorDataType(tensor) == QNN_DATATYPE_SFIXED_POINT_32) {
      operator<< <int32_t>(out, GetQnnTensorClientBuf(tensor));
    } else if (GetQnnTensorDataType(tensor) == QNN_DATATYPE_UINT_8 ||
               GetQnnTensorDataType(tensor) == QNN_DATATYPE_UFIXED_POINT_8) {
      operator<< <uint8_t>(out, GetQnnTensorClientBuf(tensor));
    } else {
      operator<< <int8_t>(out, GetQnnTensorClientBuf(tensor));
    }
  }
  out << " quantizeParams:" << GetQnnTensorQParams(tensor);
  return out;
}

std::ostream& operator<<(std::ostream& out, const Qnn_ParamType_t& param_type) {
  switch (param_type) {
    case QNN_PARAMTYPE_SCALAR:
      out << "QNN_PARAMTYPE_SCALAR";
      break;
    case QNN_PARAMTYPE_TENSOR:
      out << "QNN_PARAMTYPE_TENSOR";
      break;
    default:
      out << "Unknown type";
  }
  return out;
}

std::ostream& operator<<(std::ostream& out, const Qnn_Param_t& qnn_param) {
  out << " type=" << qnn_param.paramType;
  out << " name=" << qnn_param.name;
  if (qnn_param.paramType == QNN_PARAMTYPE_TENSOR) {
    out << qnn_param.tensorParam;
  } else {
    out << " value=" << qnn_param.scalarParam;
  }
  return out;
}

std::ostream& operator<<(std::ostream& out, const QnnOpConfigWrapper& op_conf_wrapper) {
  out << "Qnn_OpConfig node name: " << op_conf_wrapper.GetOpName()
      << " package_name: " << op_conf_wrapper.GetPackageName()
      << " QNN_op_type: " << op_conf_wrapper.GetTypeName()
      << " num_of_inputs: " << op_conf_wrapper.GetInputsNum()
      << " num_of_outputs: " << op_conf_wrapper.GetOutputsNum()
      << " num_of_params: " << op_conf_wrapper.GetParamsNum();

  out << std::endl
      << " node_inputs:" << std::endl;
  for (uint32_t i = 0; i < op_conf_wrapper.GetInputsNum(); i++) {
    out << op_conf_wrapper.GetInputTensors()[i] << std::endl;
  }
  out << " node_outputs:" << std::endl;
  for (uint32_t i = 0; i < op_conf_wrapper.GetOutputsNum(); i++) {
    out << op_conf_wrapper.GetOutputTensors()[i] << std::endl;
  }
  out << " node_params:" << std::endl;
  for (uint32_t i = 0; i < op_conf_wrapper.GetParamsNum(); i++) {
    out << op_conf_wrapper.GetParams()[i] << std::endl;
  }
  return out;
}

template <typename T>
static inline nlohmann::json JSONFromSpan(gsl::span<const T> elems) {
  nlohmann::json json_array = nlohmann::json::array();

  for (auto elem : elems) {
    json_array.push_back(elem);
  }

  return json_array;
}

template <typename T>
static inline nlohmann::json JSONFromRawArray(const void* ptr, uint32_t num_bytes) {
  gsl::span<const T> elems{reinterpret_cast<const T*>(ptr), num_bytes / sizeof(T)};
  return JSONFromSpan(elems);
}

static nlohmann::json GetQnnClientBufJSON(const Qnn_ClientBuffer_t& buf, Qnn_DataType_t data_type) {
  switch (data_type) {
    case QNN_DATATYPE_BOOL_8:  // Print bool the same as int8 (0 or 1)
    case QNN_DATATYPE_INT_8:
      return JSONFromRawArray<int8_t>(buf.data, buf.dataSize);
    case QNN_DATATYPE_INT_16:
      return JSONFromRawArray<int16_t>(buf.data, buf.dataSize);
    case QNN_DATATYPE_INT_32:
      return JSONFromRawArray<int32_t>(buf.data, buf.dataSize);
    case QNN_DATATYPE_INT_64:
      return JSONFromRawArray<int64_t>(buf.data, buf.dataSize);
    case QNN_DATATYPE_UINT_8:
      return JSONFromRawArray<uint8_t>(buf.data, buf.dataSize);
    case QNN_DATATYPE_UINT_16:
      return JSONFromRawArray<uint16_t>(buf.data, buf.dataSize);
    case QNN_DATATYPE_UINT_32:
      return JSONFromRawArray<uint32_t>(buf.data, buf.dataSize);
    case QNN_DATATYPE_UINT_64:
      return JSONFromRawArray<uint64_t>(buf.data, buf.dataSize);
    case QNN_DATATYPE_FLOAT_16:  // Print float16 as float32 in JSON
    case QNN_DATATYPE_FLOAT_32:
      return JSONFromRawArray<float>(buf.data, buf.dataSize);
    default:
      return nlohmann::json::array();  // Default to empty array for unsupported types.
  }
}

static nlohmann::json GetQnnTensorJSON(const Qnn_Tensor_t& tensor, bool include_static_data = false) {
  using json = nlohmann::json;
  json tensor_json = json::object();

  if (tensor.version != QNN_TENSOR_VERSION_1) {
    ORT_THROW("QNN tensor version not supported, QNN tensor version: ", tensor.version);
  }

  tensor_json["id"] = tensor.v1.id;
  tensor_json["type"] = tensor.v1.type;
  tensor_json["dataFormat"] = tensor.v1.dataFormat;
  tensor_json["data_type"] = tensor.v1.dataType;
  tensor_json["src_axis_format"] = "NOT_YET_DEFINED";
  tensor_json["axis_format"] = "NOT_YET_DEFINED";

  const Qnn_QuantizeParams_t& quant_params = tensor.v1.quantizeParams;
  tensor_json["quant_params"] = {
      {"definition", quant_params.encodingDefinition},
      {"encoding", quant_params.quantizationEncoding},
      {"scale_offset", {{"scale", quant_params.scaleOffsetEncoding.scale}, {"offset", quant_params.scaleOffsetEncoding.offset}}}};

  gsl::span<const uint32_t> dims{tensor.v1.dimensions, tensor.v1.rank};
  tensor_json["dims"] = JSONFromSpan(dims);

  if (tensor.v1.type == Qnn_TensorType_t::QNN_TENSOR_TYPE_STATIC) {
    if (include_static_data) {
      tensor_json["data"] = GetQnnClientBufJSON(tensor.v1.clientBuf, tensor.v1.dataType);
    } else {
      std::stringstream ss;
      ss << CalcQnnTensorNumElems(tensor);
      tensor_json["params_count"] = ss.str();
    }
  }

  return tensor_json;
}

static nlohmann::json GetQnnScalarParamJSON(const Qnn_Scalar_t& param) {
  nlohmann::json param_json = nlohmann::json::object();
  std::stringstream ss;
  ss << static_cast<uint64_t>(param.dataType);

  switch (param.dataType) {
    case QNN_DATATYPE_BOOL_8:  // Print bool the same as int8 (0 or 1)
    case QNN_DATATYPE_INT_8:
      param_json[ss.str()] = param.int8Value;
      break;
    case QNN_DATATYPE_INT_16:
      param_json[ss.str()] = param.int16Value;
      break;
    case QNN_DATATYPE_INT_32:
      param_json[ss.str()] = param.int32Value;
      break;
    case QNN_DATATYPE_UINT_8:
      param_json[ss.str()] = param.uint8Value;
      break;
    case QNN_DATATYPE_UINT_16:
      param_json[ss.str()] = param.uint16Value;
      break;
    case QNN_DATATYPE_UINT_32:
      param_json[ss.str()] = param.uint32Value;
      break;
    case QNN_DATATYPE_FLOAT_32:
      param_json[ss.str()] = param.floatValue;
      break;
    default:
      // Do nothing for unsupported types.
      break;
  }

  return param_json;
}

static nlohmann::json GetQnnTensorNamesJSON(gsl::span<const Qnn_Tensor_t> tensors) {
  nlohmann::json names_json = nlohmann::json::array();

  for (const auto& tensor : tensors) {
    names_json.push_back(tensor.v1.name);
  }

  return names_json;
}

static nlohmann::json GetQnnOpJSON(const Qnn_OpConfig_t& node) {
  using json = nlohmann::json;
  json op_json = json::object();
  op_json["package"] = node.v1.packageName;
  op_json["type"] = node.v1.typeName;

  json tensor_params_json = json::object();
  json scalar_params_json = json::object();

  gsl::span<const Qnn_Param_t> params{node.v1.params, node.v1.numOfParams};
  for (const auto& param : params) {
    if (param.paramType == QNN_PARAMTYPE_SCALAR) {
      scalar_params_json[param.name] = GetQnnScalarParamJSON(param.scalarParam);
    } else if (param.paramType == QNN_PARAMTYPE_TENSOR) {
      tensor_params_json[param.name][param.tensorParam.v1.name] = GetQnnTensorJSON(param.tensorParam, true);
    }
  }

  op_json["tensor_params"] = std::move(tensor_params_json);
  op_json["scalar_params"] = std::move(scalar_params_json);
  op_json["input_names"] = GetQnnTensorNamesJSON(gsl::span<const Qnn_Tensor_t>{node.v1.inputTensors, node.v1.numOfInputs});
  op_json["output_names"] = GetQnnTensorNamesJSON(gsl::span<const Qnn_Tensor_t>{node.v1.outputTensors, node.v1.numOfOutputs});
  op_json["macs_per_inference"] = "";  // Don't know what this is.

  return op_json;
}

QnnJSONGraph::QnnJSONGraph() {
  using json = nlohmann::json;

  json_ = {
      {"model.cpp", ""},
      {"model.bin", ""},
      {"converter_command", ""},
      {"copyright_str", ""},
      {"op_types", json::array()},
      {"Total parameters", ""},
      {"Total MACs per inference", ""},
      {"graph", {{"tensors", json::object()}, {"nodes", json::object()}}}
  };
}

void QnnJSONGraph::AddOp(const QnnOpConfigWrapper& op_conf_wrapper) {
  // Serialize inputs and outputs.
  AddOpTensors({op_conf_wrapper.GetInputTensors(), op_conf_wrapper.GetInputsNum()});
  AddOpTensors({op_conf_wrapper.GetOutputTensors(), op_conf_wrapper.GetOutputsNum()});

  // Track unique op types (serialized in Finalize()).
  const std::string& op_type = op_conf_wrapper.GetTypeName();
  if (seen_op_types_.count(op_type) == 0) {
    seen_op_types_.insert(op_type);
  }

  // Serialize op
  json_["graph"]["nodes"][op_conf_wrapper.GetOpName()] = GetQnnOpJSON(op_conf_wrapper.GetQnnOpConfig());
}

void QnnJSONGraph::AddOpTensors(gsl::span<const Qnn_Tensor_t> tensors) {
  for (const auto& tensor : tensors) {
    std::string name = GetQnnTensorName(tensor);  // Copies name into std::string, which is moved into seen_tensors_.
    if (seen_tensors_.count(name) == 0) {
      json_["graph"]["tensors"][name] = GetQnnTensorJSON(tensor);
      seen_tensors_.insert(std::move(name));
    }
  }
}

const nlohmann::json& QnnJSONGraph::Finalize() {
  json_["op_types"] = seen_op_types_;
  return json_;
}

}  // namespace utils
}  // namespace qnn
}  // namespace onnxruntime
