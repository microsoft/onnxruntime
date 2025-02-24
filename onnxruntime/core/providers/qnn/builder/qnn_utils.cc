// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/qnn/builder/qnn_utils.h"

#include <algorithm>
#include <functional>
#include <limits>
#include <map>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "core/providers/qnn/ort_api.h"
#include "core/providers/qnn/builder/qnn_def.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "nlohmann/json.hpp"

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
      {ONNX_TENSOR_ELEMENT_DATA_TYPE_INT4, sizeof(Int4x2)},
      {ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT4, sizeof(UInt4x2)},
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

size_t GetElementSizeByType(ONNX_NAMESPACE::TensorProto_DataType onnx_type) {
  switch (onnx_type) {
    case ONNX_NAMESPACE::TensorProto_DataType_INT4:
      return sizeof(Int4x2);
    case ONNX_NAMESPACE::TensorProto_DataType_UINT4:
      return sizeof(UInt4x2);
    case ONNX_NAMESPACE::TensorProto_DataType_INT8:
      return sizeof(int8_t);
    case ONNX_NAMESPACE::TensorProto_DataType_UINT8:
      return sizeof(uint8_t);
    case ONNX_NAMESPACE::TensorProto_DataType_INT16:
      return sizeof(int16_t);
    case ONNX_NAMESPACE::TensorProto_DataType_UINT16:
      return sizeof(uint16_t);
    case ONNX_NAMESPACE::TensorProto_DataType_INT32:
      return sizeof(int32_t);
    case ONNX_NAMESPACE::TensorProto_DataType_UINT32:
      return sizeof(uint32_t);
    case ONNX_NAMESPACE::TensorProto_DataType_INT64:
      return sizeof(int64_t);
    case ONNX_NAMESPACE::TensorProto_DataType_UINT64:
      return sizeof(uint64_t);
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT16:
      return 2;
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:
      return sizeof(float);
    case ONNX_NAMESPACE::TensorProto_DataType_DOUBLE:
      return sizeof(double);
    case ONNX_NAMESPACE::TensorProto_DataType_BOOL:
      return sizeof(bool);
    default:
      return 0;
  }
  // Unreachable
}

size_t GetQnnTensorDataSizeInBytes(gsl::span<const uint32_t> shape, Qnn_DataType_t element_type) {
  ORT_ENFORCE(!shape.empty(), "Empty shape not allowed.");  // TODO can we just treat empty shape as a scalar?
  SafeInt<size_t> data_length = GetElementSizeByType(element_type);
  return std::accumulate(shape.begin(), shape.end(), data_length, std::multiplies<>{});
}

bool QnnTensorHasDynamicShape(const Qnn_Tensor_t& tensor) {
  const uint8_t* is_dynamic_dimensions = GetQnnTensorIsDynamicDimensions(tensor);
  if (is_dynamic_dimensions == nullptr) {
    return false;
  }

  const auto rank = GetQnnTensorRank(tensor);
  return std::any_of(is_dynamic_dimensions, is_dynamic_dimensions + rank,
                     [](uint8_t is_dynamic_dimension) { return is_dynamic_dimension != 0; });
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
    case QNN_DATATYPE_SFIXED_POINT_4:
      out << "QNN_DATATYPE_SFIXED_POINT_4";
      break;
    case QNN_DATATYPE_UFIXED_POINT_4:
      out << "QNN_DATATYPE_UFIXED_POINT_4";
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
    } else if (quantize_params.quantizationEncoding == QNN_QUANTIZATION_ENCODING_BW_SCALE_OFFSET) {
      out << " bitwidth=" << quantize_params.bwScaleOffsetEncoding.bitwidth;
      out << " scale=" << quantize_params.bwScaleOffsetEncoding.scale;
      out << " offset=" << quantize_params.bwScaleOffsetEncoding.offset;
    } else if (quantize_params.quantizationEncoding == QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET) {
      out << " axis=" << quantize_params.axisScaleOffsetEncoding.axis;
      size_t num_elems = quantize_params.axisScaleOffsetEncoding.numScaleOffsets;
      bool truncate = num_elems > 20;
      num_elems = truncate ? 20 : num_elems;
      out << " scales=(";
      for (size_t i = 0; i < num_elems; i++) {
        out << quantize_params.axisScaleOffsetEncoding.scaleOffset[i].scale << (i == num_elems - 1 ? "" : " ");
      }
      out << ") offsets=(";
      for (size_t i = 0; i < num_elems; i++) {
        out << quantize_params.axisScaleOffsetEncoding.scaleOffset[i].offset << (i == num_elems - 1 ? "" : " ");
      }
      out << (truncate ? "...)" : ")");
    } else if (quantize_params.quantizationEncoding == QNN_QUANTIZATION_ENCODING_BW_AXIS_SCALE_OFFSET) {
      out << " axis=" << quantize_params.bwAxisScaleOffsetEncoding.axis;
      out << " bw=" << quantize_params.bwAxisScaleOffsetEncoding.bitwidth;
      size_t num_elems = quantize_params.bwAxisScaleOffsetEncoding.numElements;
      bool truncate = num_elems > 20;
      num_elems = truncate ? 20 : num_elems;
      out << " scales=(";
      for (size_t i = 0; i < num_elems; i++) {
        out << quantize_params.bwAxisScaleOffsetEncoding.scales[i] << (i == num_elems - 1 ? "" : " ");
      }
      out << ") offsets=(";
      for (size_t i = 0; i < num_elems; i++) {
        out << quantize_params.bwAxisScaleOffsetEncoding.offsets[i] << (i == num_elems - 1 ? "" : " ");
      }
      out << (truncate ? "...)" : ")");
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
  const bool truncate = count > 100;

  count = truncate ? 100 : count;  // limit to 100 data
  out << " clientBuf=(";
  for (uint32_t i = 0; i < count; i++) {
    if constexpr (sizeof(T) == 1) {
      out << static_cast<int32_t>(data[i]) << " ";
    } else {
      out << data[i] << " ";
    }
  }
  out << (truncate ? "..." : "") << ")";
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
// TODO: the code below has compilation errors with the latest ABSL
#if 0
  if (GetQnnTensorMemType(tensor) == QNN_TENSORMEMTYPE_RAW) {
    if (GetQnnTensorDataType(tensor) == QNN_DATATYPE_FLOAT_32) {
      operator<< <float>(out, GetQnnTensorClientBuf(tensor));
    } else if (GetQnnTensorDataType(tensor) == QNN_DATATYPE_UINT_32 ||
               GetQnnTensorDataType(tensor) == QNN_DATATYPE_UFIXED_POINT_32) {
      operator<< <uint32_t>(out, GetQnnTensorClientBuf(tensor));
    } else if (GetQnnTensorDataType(tensor) == QNN_DATATYPE_INT_32 ||
               GetQnnTensorDataType(tensor) == QNN_DATATYPE_SFIXED_POINT_32) {
      operator<< <int32_t>(out, GetQnnTensorClientBuf(tensor));
    } else if (GetQnnTensorDataType(tensor) == QNN_DATATYPE_UINT_16 ||
               GetQnnTensorDataType(tensor) == QNN_DATATYPE_UFIXED_POINT_16) {
      operator<< <uint16_t>(out, GetQnnTensorClientBuf(tensor));
    } else if (GetQnnTensorDataType(tensor) == QNN_DATATYPE_INT_16 ||
               GetQnnTensorDataType(tensor) == QNN_DATATYPE_SFIXED_POINT_16) {
      operator<< <int16_t>(out, GetQnnTensorClientBuf(tensor));
    } else if (GetQnnTensorDataType(tensor) == QNN_DATATYPE_UINT_8 ||
               GetQnnTensorDataType(tensor) == QNN_DATATYPE_UFIXED_POINT_8) {
      operator<< <uint8_t>(out, GetQnnTensorClientBuf(tensor));
    } else {
      operator<< <int8_t>(out, GetQnnTensorClientBuf(tensor));
    }
  }
#endif
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

// Returns a JSON array from a gsl::span.
template <typename T>
static inline nlohmann::json JSONFromSpan(gsl::span<const T> elems) {
  nlohmann::json json_array = nlohmann::json::array();

  for (auto elem : elems) {
    json_array.push_back(elem);
  }

  return json_array;
}

// Fills json array with elements from the raw source buffer.
// Returns the number of bytes copied from the raw source buffer.
template <typename T>
static inline uint32_t FillJSONArrayFromRawData(nlohmann::json* json_array, const void* ptr, uint32_t num_elems) {
  gsl::span<const T> elems{reinterpret_cast<const T*>(ptr), static_cast<size_t>(num_elems)};
  for (auto elem : elems) {
    json_array->push_back(elem);
  }

  return num_elems * sizeof(T);
}

template <>
inline uint32_t FillJSONArrayFromRawData<MLFloat16>(nlohmann::json* json_array, const void* ptr, uint32_t num_elems) {
  gsl::span<const MLFloat16> elems{reinterpret_cast<const MLFloat16*>(ptr), static_cast<size_t>(num_elems)};
  for (auto elem : elems) {
    json_array->push_back(elem.ToFloat());
  }

  return num_elems * sizeof(MLFloat16);
}

// Fills json array with typed elements from the raw source buffer.
// Returns the number of bytes copied from the raw source buffer.
static uint32_t AppendQnnElemsToJSONArray(nlohmann::json* json_array, const void* data, uint32_t num_elems, Qnn_DataType_t data_type) {
  switch (data_type) {
    case QNN_DATATYPE_BOOL_8:  // Handle bool the same as int8 (0 or 1)
    case QNN_DATATYPE_INT_8:
      return FillJSONArrayFromRawData<int8_t>(json_array, data, num_elems);
    case QNN_DATATYPE_INT_16:
      return FillJSONArrayFromRawData<int16_t>(json_array, data, num_elems);
    case QNN_DATATYPE_INT_32:
      return FillJSONArrayFromRawData<int32_t>(json_array, data, num_elems);
    case QNN_DATATYPE_INT_64:
      return FillJSONArrayFromRawData<int64_t>(json_array, data, num_elems);
    case QNN_DATATYPE_UINT_8:
      return FillJSONArrayFromRawData<uint8_t>(json_array, data, num_elems);
    case QNN_DATATYPE_UINT_16:
      return FillJSONArrayFromRawData<uint16_t>(json_array, data, num_elems);
    case QNN_DATATYPE_UINT_32:
      return FillJSONArrayFromRawData<uint32_t>(json_array, data, num_elems);
    case QNN_DATATYPE_UINT_64:
      return FillJSONArrayFromRawData<uint64_t>(json_array, data, num_elems);
    case QNN_DATATYPE_FLOAT_32:
      return FillJSONArrayFromRawData<float>(json_array, data, num_elems);
    case QNN_DATATYPE_FLOAT_16:
      return FillJSONArrayFromRawData<MLFloat16>(json_array, data, num_elems);
    default:
      return 0;  // Do not append anything for unsupported types.
  }
}

// Returns a JSON array that contains static tensor data. The resulting JSON array is constructed hierarchically
// according to the provided dimensions/shape.
//
// Example:
// If buf = [0, 1, 2, 3, 4, 5] and dims = [1, 2, 3]
//   => returns JSON array [[[0, 1, 2], [3, 4, 5]]]
static nlohmann::json GetQnnClientBufJSON(const Qnn_ClientBuffer_t& buf, Qnn_DataType_t data_type,
                                          gsl::span<const uint32_t> dims) {
  using json = nlohmann::json;
  const char* data_ptr = reinterpret_cast<const char*>(buf.data);

  // Calculate number of elements.
  uint32_t num_elems = 1;
  for (auto d : dims) {
    num_elems *= d;
  }

  if (num_elems == 0) {
    return json::array();
  }

  const uint32_t last_dim = dims.back();
  const uint32_t num_dims = gsl::narrow_cast<uint32_t>(dims.size());
  std::vector<json> curr;
  curr.reserve(num_elems / last_dim);

  // Group raw data into individual JSON arrays of size `last_dim` each.
  // Store these JSON arrays in the `curr` vector.
  for (uint32_t j = num_elems; j > 0; j -= last_dim) {
    curr.push_back(json::array());
    data_ptr += AppendQnnElemsToJSONArray(&curr.back(), data_ptr, last_dim, data_type);
  }

  // Iterate through dimension values backwards (starting at second-to-last).
  // In each iteration, we collect the JSON arrays in the `curr` vector into groups (i.e., new JSON arrays) of
  // size `dim_val`. This new/smaller collection of JSON arrays becomes the input for the next iteration.
  for (uint32_t i = num_dims - 1; i-- > 0;) {
    const uint32_t dim_val = dims[i];
    std::vector<json> next;

    for (uint32_t j = 0; j < curr.size(); ++j) {
      if (j % dim_val == 0) {
        next.push_back(json::array());
      }

      next.back().emplace_back(std::move(curr[j]));
    }

    curr = std::move(next);
  }

  assert(curr.size() == 1);
  return curr[0];
}

// Returns a JSON representation of a QNN tensor.
// Example:
//
// {
//     "id" : 1652639423,
//     "type" : 3
//     "dataFormat" : 0,
//     "data_type" : 562,
//     "dims" : [ 1, 224, 224, 3 ],
//     "quant_params" : { ... },
//     "axis_format" : "NOT_YET_DEFINED",
//     "src_axis_format" : "NOT_YET_DEFINED",
// }
static nlohmann::json GetQnnTensorJSON(const Qnn_Tensor_t& tensor, bool include_static_data = false) {
  using json = nlohmann::json;
  json tensor_json = json::object();
  const Qnn_TensorType_t tensor_type = GetQnnTensorType(tensor);

  tensor_json["id"] = GetQnnTensorID(tensor);
  tensor_json["type"] = tensor_type;
  tensor_json["dataFormat"] = GetQnnTensorDataFormat(tensor);
  tensor_json["data_type"] = GetQnnTensorDataType(tensor);
  tensor_json["src_axis_format"] = "NOT_YET_DEFINED";
  tensor_json["axis_format"] = "NOT_YET_DEFINED";

  const Qnn_QuantizeParams_t& quant_params = GetQnnTensorQParams(tensor);
  tensor_json["quant_params"] = {
      {"definition", quant_params.encodingDefinition},
      {"encoding", quant_params.quantizationEncoding},
      {"scale_offset", {{"scale", quant_params.scaleOffsetEncoding.scale}, {"offset", quant_params.scaleOffsetEncoding.offset}}}};

  gsl::span<const uint32_t> dims{GetQnnTensorDims(tensor), GetQnnTensorRank(tensor)};
  tensor_json["dims"] = JSONFromSpan(dims);

  if (tensor_type == Qnn_TensorType_t::QNN_TENSOR_TYPE_STATIC) {
    if (include_static_data) {
      tensor_json["data"] = GetQnnClientBufJSON(GetQnnTensorClientBuf(tensor), GetQnnTensorDataType(tensor), dims);
    } else {
      std::stringstream ss;
      ss << CalcQnnTensorNumElems(tensor);
      tensor_json["params_count"] = ss.str();
    }
  }

  return tensor_json;
}

// Returns a JSON object representation of a QNN scalar parameter. Example: { "306": 1 }
// Note that the key is the stringified data type.
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

// Returns a JSON array initialized with the names of the provided QNN tensors.
static nlohmann::json GetQnnTensorNamesJSON(gsl::span<const Qnn_Tensor_t> tensors) {
  nlohmann::json names_json = nlohmann::json::array();

  for (const auto& tensor : tensors) {
    names_json.push_back(GetQnnTensorName(tensor));
  }

  return names_json;
}

// Returns a JSON representation of a QNN operator.
// Example:
// {
//     "package": "qti.aisw",
//     "type": "Conv2d",
//     "input_names": [ "Transpose_token_2012_out0", "weight_quantized", "beta_quantized" ],
//     "output_names": [ "resnetv17_relu0_fwd_QuantizeLinear" ],
//     "scalar_params": { "group": {...} },
//     "tensor_params": { "stride": {...} },
//     "macs_per_inference": ""
// }
static nlohmann::json GetQnnOpJSON(const QnnOpConfigWrapper& op_config) {
  using json = nlohmann::json;
  json op_json = json::object();
  op_json["package"] = op_config.GetPackageName();
  op_json["type"] = op_config.GetTypeName();

  json tensor_params_json = json::object();
  json scalar_params_json = json::object();

  gsl::span<const Qnn_Param_t> params{op_config.GetParams(), op_config.GetParamsNum()};
  for (const auto& param : params) {
    if (param.paramType == QNN_PARAMTYPE_SCALAR) {
      scalar_params_json[param.name] = GetQnnScalarParamJSON(param.scalarParam);
    } else if (param.paramType == QNN_PARAMTYPE_TENSOR) {
      tensor_params_json[param.name][GetQnnTensorName(param.tensorParam)] = GetQnnTensorJSON(param.tensorParam, true);
    }
  }

  op_json["tensor_params"] = std::move(tensor_params_json);
  op_json["scalar_params"] = std::move(scalar_params_json);
  op_json["input_names"] = GetQnnTensorNamesJSON(gsl::span<const Qnn_Tensor_t>{op_config.GetInputTensors(),
                                                                               op_config.GetInputsNum()});
  op_json["output_names"] = GetQnnTensorNamesJSON(gsl::span<const Qnn_Tensor_t>{op_config.GetOutputTensors(),
                                                                                op_config.GetOutputsNum()});
  op_json["macs_per_inference"] = "";  // Metadata set by QNN converter tools. Not needed.

  return op_json;
}

QnnJSONGraph::QnnJSONGraph() {
  using json = nlohmann::json;

  json_ = {
      // Use dummy model.cpp and model.bin files when loading JSON with QNN Netron.
      // They don't have to exist in order to visualize the graph.
      {"model.cpp", "N/A"},
      {"model.bin", "N/A"},
      {"converter_command", ""},
      {"copyright_str", "Copyright (c) Microsoft Corporation. All rights reserved."},
      {"op_types", json::array()},
      {"Total parameters", ""},
      {"Total MACs per inference", ""},
      {"graph", {{"tensors", json::object()}, {"nodes", json::object()}}}};
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
  json_["graph"]["nodes"][op_conf_wrapper.GetOpName()] = GetQnnOpJSON(op_conf_wrapper);
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

Status GetQnnDataType(const bool is_quantized_tensor, const ONNX_NAMESPACE::TypeProto* type_proto,
                      Qnn_DataType_t& tensor_data_type) {
  if (!type_proto || !type_proto->tensor_type().has_elem_type()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "The tensor doesn't have elem_type.");
  }

  int32_t onnx_data_type = type_proto->tensor_type().elem_type();
  ORT_RETURN_IF_NOT(OnnxDataTypeToQnnDataType(onnx_data_type, tensor_data_type, is_quantized_tensor),
                    "Failed to map Onnx data type to Qnn data type!");

  return Status::OK();
}

const std::string& GetNodeName(const NodeUnit& node_unit) {
  const std::string& node_name = node_unit.Name();
  if (node_name.empty()) {
    return node_unit.Outputs()[0].node_arg.Name();
  }

  return node_name;
}

bool OnnxDataTypeToQnnDataType(const int32_t onnx_data_type, Qnn_DataType_t& qnn_data_type, bool is_quantized) {
  const std::unordered_map<int32_t, Qnn_DataType_t> onnx_to_qnn_data_type = {
      {ONNX_NAMESPACE::TensorProto_DataType_INT8, QNN_DATATYPE_INT_8},
      {ONNX_NAMESPACE::TensorProto_DataType_INT16, QNN_DATATYPE_INT_16},
      {ONNX_NAMESPACE::TensorProto_DataType_INT32, QNN_DATATYPE_INT_32},
      {ONNX_NAMESPACE::TensorProto_DataType_INT64, QNN_DATATYPE_INT_64},
      {ONNX_NAMESPACE::TensorProto_DataType_UINT8, QNN_DATATYPE_UINT_8},
      {ONNX_NAMESPACE::TensorProto_DataType_UINT16, QNN_DATATYPE_UINT_16},
      {ONNX_NAMESPACE::TensorProto_DataType_UINT32, QNN_DATATYPE_UINT_32},
      {ONNX_NAMESPACE::TensorProto_DataType_UINT64, QNN_DATATYPE_UINT_64},
      {ONNX_NAMESPACE::TensorProto_DataType_FLOAT16, QNN_DATATYPE_FLOAT_16},
      {ONNX_NAMESPACE::TensorProto_DataType_FLOAT, QNN_DATATYPE_FLOAT_32},
      {ONNX_NAMESPACE::TensorProto_DataType_BOOL, QNN_DATATYPE_BOOL_8},
  };

  const std::unordered_map<int32_t, Qnn_DataType_t> onnx_to_qnn_data_type_quantized = {
      {ONNX_NAMESPACE::TensorProto_DataType_INT4, QNN_DATATYPE_SFIXED_POINT_8},
      {ONNX_NAMESPACE::TensorProto_DataType_INT8, QNN_DATATYPE_SFIXED_POINT_8},
      {ONNX_NAMESPACE::TensorProto_DataType_INT16, QNN_DATATYPE_SFIXED_POINT_16},
      {ONNX_NAMESPACE::TensorProto_DataType_INT32, QNN_DATATYPE_SFIXED_POINT_32},
      {ONNX_NAMESPACE::TensorProto_DataType_INT64, QNN_DATATYPE_INT_64},
      {ONNX_NAMESPACE::TensorProto_DataType_UINT4, QNN_DATATYPE_UFIXED_POINT_8},
      {ONNX_NAMESPACE::TensorProto_DataType_UINT8, QNN_DATATYPE_UFIXED_POINT_8},
      {ONNX_NAMESPACE::TensorProto_DataType_UINT16, QNN_DATATYPE_UFIXED_POINT_16},
      {ONNX_NAMESPACE::TensorProto_DataType_UINT32, QNN_DATATYPE_UFIXED_POINT_32},
      {ONNX_NAMESPACE::TensorProto_DataType_UINT64, QNN_DATATYPE_UINT_64},
      {ONNX_NAMESPACE::TensorProto_DataType_FLOAT16, QNN_DATATYPE_FLOAT_16},
      {ONNX_NAMESPACE::TensorProto_DataType_FLOAT, QNN_DATATYPE_FLOAT_32},
      {ONNX_NAMESPACE::TensorProto_DataType_BOOL, QNN_DATATYPE_BOOL_8},
  };

  const auto do_type_mapping = [](const std::unordered_map<int32_t, Qnn_DataType_t>& mapping_table,
                                  const int32_t onnx_data_type,
                                  Qnn_DataType_t& qnn_data_type) -> bool {
    auto pos = mapping_table.find(onnx_data_type);
    if (pos == mapping_table.end()) {
      return false;
    }
    qnn_data_type = pos->second;
    return true;
  };

  if (is_quantized) {
    return do_type_mapping(onnx_to_qnn_data_type_quantized, onnx_data_type, qnn_data_type);
  } else {
    return do_type_mapping(onnx_to_qnn_data_type, onnx_data_type, qnn_data_type);
  }
}

std::pair<float, float> CheckMinMax(float rmin, float rmax) {
  // Both QNN and ORT require the range to include 0.0f
  rmin = std::min(rmin, 0.0f);
  rmax = std::max(rmax, 0.0f);

  // Ensure a minimum range of 0.0001 (required by QNN)
  rmax = std::max(rmax, rmin + 0.0001f);

  return std::make_pair(rmin, rmax);
}

inline float RoundHalfToEven(float input) {
  if (!std::isfinite(input)) {
    return input;
  }
  // std::remainder returns x - n, where n is the integral value nearest to x. When |x - n| = 0.5, n is chosen to be even
  return input - std::remainderf(input, 1.f);
}

Status GetQuantParams(float rmin,
                      float rmax,
                      const Qnn_DataType_t qnn_data_type,
                      float& scale,
                      int32_t& zero_point,
                      bool symmetric) {
  std::tie(rmin, rmax) = CheckMinMax(rmin, rmax);
  if (symmetric) {
    float abs_max = std::max(abs(rmax), abs(rmin));
    rmax = abs_max;
    rmin = -abs_max;
  }

  double rmin_dbl = static_cast<double>(rmin);
  double rmax_dbl = static_cast<double>(rmax);
  double qmin = 0.0;
  double qmax = 0.0;
  ORT_RETURN_IF_ERROR(GetQminQmax(qnn_data_type, qmin, qmax, symmetric));

  double scale_dbl = (rmax_dbl - rmin_dbl) / (qmax - qmin);
  double initial_zero_point = 0.0;
  if (symmetric) {
    initial_zero_point = std::round(rmin_dbl + rmax_dbl) / 2;
  } else {
    initial_zero_point = qmin - (rmin_dbl / scale_dbl);
  }
  zero_point = static_cast<int32_t>(RoundHalfToEven(static_cast<float>(Saturate(qmax, qmin, initial_zero_point))));
  zero_point = -zero_point;  // Negate to match QNN quantization definition.
  scale = static_cast<float>(scale_dbl);
  return Status::OK();
}

double Dequantize(int32_t offset, float scale, const double quant_value) {
  double offset_d = static_cast<double>(offset);
  double scale_d = static_cast<double>(scale);
  return (quant_value + offset_d) * scale_d;
}

Status Quantize(const double double_value,
                const float scale,
                const int32_t zero_point,
                const Qnn_DataType_t qnn_data_type,
                int& quant_value) {
  int qmin = 0;
  int qmax = 255;
  ORT_RETURN_IF_ERROR(GetQminQmax(qnn_data_type, qmin, qmax));
  quant_value = Saturate(qmax, qmin, static_cast<int>(std::round((double_value / scale) - zero_point)));
  return Status::OK();
}

size_t ShapeSizeCalc(gsl::span<const uint32_t> shape, size_t start, size_t end) {
  size_t size = 1;
  for (size_t i = start; i < end; i++) {
    size *= shape[i];
  }
  return size;
}

Status GetDataQuantParams(gsl::span<const float> data, gsl::span<const uint32_t> shape,
                          /*out*/ gsl::span<float> scales, /*out*/ gsl::span<int32_t> offsets,
                          Qnn_DataType_t data_type, bool symmetric, std::optional<int64_t> axis) {
  const size_t num_dims = shape.size();
  const size_t num_elems = ShapeSizeCalc(shape, 0, num_dims);
  ORT_RETURN_IF_NOT(num_elems == data.size(), "Shape mismatch with data to quantize");

  size_t block_count = 1;
  size_t broadcast_dim = 1;
  size_t block_size = num_elems;

  if (axis.has_value()) {
    size_t axis_no_neg = *axis < 0 ? static_cast<size_t>(*axis) + num_dims : static_cast<size_t>(*axis);
    block_count = ShapeSizeCalc(shape, 0, axis_no_neg);
    broadcast_dim = shape[axis_no_neg];
    block_size = ShapeSizeCalc(shape, axis_no_neg + 1, num_dims);
  }

  ORT_RETURN_IF_NOT(scales.size() == broadcast_dim, "Unexpected size of scales output buffer");
  ORT_RETURN_IF_NOT(offsets.size() == broadcast_dim, "Unexpected size of offsets output buffer");

  size_t i = 0;
  for (size_t n = 0; n < block_count; n++) {
    for (size_t bd = 0; bd < broadcast_dim; bd++) {
      float rmin = std::numeric_limits<float>::max();
      float rmax = std::numeric_limits<float>::lowest();
      for (size_t j = 0; j < block_size; j++) {
        rmin = std::min(rmin, data[i]);
        rmax = std::max(rmax, data[i]);
        i++;
      }

      scales[bd] = 1.0f;
      offsets[bd] = 0;
      ORT_RETURN_IF_ERROR(GetQuantParams(rmin, rmax, data_type, scales[bd], offsets[bd], symmetric));
    }
  }

  assert(i == data.size());
  return Status::OK();
}

Status QuantizeData(gsl::span<const float> data, gsl::span<const uint32_t> shape,
                    gsl::span<const float> scales, gsl::span<const int32_t> offsets,
                    /*out*/ gsl::span<uint8_t> quant_bytes, Qnn_DataType_t data_type,
                    std::optional<int64_t> axis) {
  const size_t num_dims = shape.size();
  const size_t num_elems = ShapeSizeCalc(shape, 0, num_dims);
  ORT_RETURN_IF_NOT(num_elems == data.size(), "Shape mismatch with data to quantize");
  size_t expected_num_quant_bytes = GetElementSizeByType(data_type) * data.size();
  ORT_RETURN_IF_NOT(quant_bytes.size() == expected_num_quant_bytes,
                    "Cannot quantize data because output buffer is not the correct size");

  size_t block_count = 1;
  size_t broadcast_dim = 1;
  size_t block_size = num_elems;

  if (axis.has_value()) {
    size_t axis_no_neg = *axis < 0 ? static_cast<size_t>(*axis) + num_dims : static_cast<size_t>(*axis);
    block_count = ShapeSizeCalc(shape, 0, axis_no_neg);
    broadcast_dim = shape[axis_no_neg];
    block_size = ShapeSizeCalc(shape, axis_no_neg + 1, num_dims);
  }

  ORT_RETURN_IF_NOT(scales.size() == broadcast_dim, "Unexpected size of scales output buffer");
  ORT_RETURN_IF_NOT(offsets.size() == broadcast_dim, "Unexpected size of offsets output buffer");

  size_t i = 0;
  for (size_t n = 0; n < block_count; n++) {
    for (size_t bd = 0; bd < broadcast_dim; bd++) {
      switch (data_type) {
        case QNN_DATATYPE_SFIXED_POINT_8: {
          auto input_span = gsl::make_span<const float>(&data[i], block_size);
          auto output_span = gsl::make_span<uint8_t>(&quant_bytes[i * sizeof(int8_t)], sizeof(int8_t) * block_size);
          ORT_RETURN_IF_ERROR(QuantizeData<int8_t>(input_span, scales[bd], offsets[bd], output_span));
          break;
        }
        case QNN_DATATYPE_UFIXED_POINT_8: {
          auto input_span = gsl::make_span<const float>(&data[i], block_size);
          auto output_span = gsl::make_span<uint8_t>(&quant_bytes[i * sizeof(uint8_t)], sizeof(uint8_t) * block_size);
          ORT_RETURN_IF_ERROR(QuantizeData<uint8_t>(input_span, scales[bd], offsets[bd], output_span));
          break;
        }
        case QNN_DATATYPE_SFIXED_POINT_16: {
          auto input_span = gsl::make_span<const float>(&data[i], block_size);
          auto output_span = gsl::make_span<uint8_t>(&quant_bytes[i * sizeof(int16_t)], sizeof(int16_t) * block_size);
          ORT_RETURN_IF_ERROR(QuantizeData<int16_t>(input_span, scales[bd], offsets[bd], output_span));
          break;
        }
        case QNN_DATATYPE_UFIXED_POINT_16: {
          auto input_span = gsl::make_span<const float>(&data[i], block_size);
          auto output_span = gsl::make_span<uint8_t>(&quant_bytes[i * sizeof(uint16_t)], sizeof(uint16_t) * block_size);
          ORT_RETURN_IF_ERROR(QuantizeData<uint16_t>(input_span, scales[bd], offsets[bd], output_span));
          break;
        }
        case QNN_DATATYPE_SFIXED_POINT_32: {
          auto input_span = gsl::make_span<const float>(&data[i], block_size);
          auto output_span = gsl::make_span<uint8_t>(&quant_bytes[i * sizeof(int32_t)], sizeof(int32_t) * block_size);
          ORT_RETURN_IF_ERROR(QuantizeData<int32_t>(input_span, scales[bd], offsets[bd], output_span));
          break;
        }
        default:
          return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Unsupported quantization data type for QuantizeData");
      }
      i += block_size;
    }
  }
  assert(i == data.size());

  return Status::OK();
}

std::string GetQnnErrorMessage(const QNN_INTERFACE_VER_TYPE& qnn_interface, Qnn_ErrorHandle_t qnn_error_handle) {
  const char* error_msg = nullptr;
  if (qnn_interface.errorGetMessage(qnn_error_handle, &error_msg) == QNN_SUCCESS) {
    return error_msg;
  }
  return MakeString("Unknown error. QNN error handle: ", qnn_error_handle);
}

std::string GetVerboseQnnErrorMessage(const QNN_INTERFACE_VER_TYPE& qnn_interface,
                                      Qnn_ErrorHandle_t qnn_error_handle) {
  const char* error_msg = nullptr;
  if (qnn_interface.errorGetVerboseMessage(qnn_error_handle, &error_msg) == QNN_SUCCESS) {
    auto free_error_msg = gsl::finally([&qnn_interface, error_msg] {
      qnn_interface.errorFreeVerboseMessage(error_msg);
    });
    return error_msg;
  }
  return MakeString("Unknown error. QNN error handle: ", qnn_error_handle);
}

TensorShape GetTensorProtoShape(const ONNX_NAMESPACE::TensorShapeProto& tensor_shape_proto) {
  const auto& onnx_dims = tensor_shape_proto.dim();
  const size_t num_dims = static_cast<size_t>(onnx_dims.size());
  std::vector<int64_t> tensor_shape_vec(num_dims);

  for (int i = 0; i < static_cast<int>(num_dims); i++) {
    const auto& onnx_dim = tensor_shape_proto.dim(i);
    tensor_shape_vec[i] = onnx_dim.has_dim_value() ? onnx_dim.dim_value() : -1;  // -1 is for symbolic dim in ORT
  }

  return TensorShape(std::move(tensor_shape_vec));
}

static Status GetTransposeStrides(const TensorShape& input_shape,
                                  gsl::span<const size_t> perm,
                                  gsl::span<size_t> input_strides,
                                  gsl::span<size_t> output_strides) {
  const size_t rank = input_shape.NumDimensions();
  ORT_RETURN_IF_NOT(perm.size() == rank, "Expected perm size of ", rank);
  ORT_RETURN_IF_NOT(input_strides.size() == rank, "Expected input_strides size of ", rank);
  ORT_RETURN_IF_NOT(output_strides.size() == rank, "Expected output_strides size of ", rank);
  std::vector<int64_t> output_shape_dims(rank);
  ORT_RETURN_IF_ERROR((qnn::utils::PermuteShape<int64_t, size_t>(input_shape.GetDims(), perm, output_shape_dims)));
  const TensorShape output_shape = TensorShape::FromExistingBuffer(output_shape_dims);

  for (size_t i = 0; i < rank; ++i) {
    int64_t stride = (i < rank - 1) ? input_shape.SizeFromDimension(i + 1) : 1;
    ORT_RETURN_IF_NOT(stride > 0, "Expected positive shape dims when computing strides.");
    input_strides[i] = static_cast<size_t>(stride);
  }

  for (size_t i = 0; i < rank; ++i) {
    int64_t stride = (i < rank - 1) ? output_shape.SizeFromDimension(i + 1) : 1;
    ORT_RETURN_IF_NOT(stride > 0, "Expected positive shape dims when computing strides.");
    output_strides[i] = static_cast<size_t>(stride);
  }

  return Status::OK();
}

// Internal function to transpose data of rank 5 with the given permutation.
// Example: transpose input from either (N,C,H,W,D) or (C,N,H,W,D) to (H,W,D,C,N).
static Status TransposeDataRank5(const TensorShape& input_shape,
                                 gsl::span<const size_t> perm,
                                 size_t elem_byte_size,
                                 gsl::span<const uint8_t> input_buffer,
                                 gsl::span<uint8_t> output_buffer) {
  std::array<size_t, 5> input_strides = {};
  std::array<size_t, 5> output_strides = {};
  ORT_RETURN_IF_ERROR(GetTransposeStrides(input_shape, perm, input_strides, output_strides));

  std::vector<size_t> perm_inverse(perm.size());
  ORT_RETURN_IF_ERROR(qnn::utils::InvertPerm<size_t>(perm, perm_inverse));

  for (int64_t d0 = 0; d0 < input_shape[0]; ++d0) {
    for (int64_t d1 = 0; d1 < input_shape[1]; ++d1) {
      for (int64_t d2 = 0; d2 < input_shape[2]; ++d2) {
        for (int64_t d3 = 0; d3 < input_shape[3]; ++d3) {
          for (int64_t d4 = 0; d4 < input_shape[4]; ++d4) {
            const size_t src_elem_index = ((d0 * input_strides[0]) +
                                           (d1 * input_strides[1]) +
                                           (d2 * input_strides[2]) +
                                           (d3 * input_strides[3]) +
                                           (d4 * input_strides[4]));
            const size_t dst_elem_index = ((d0 * output_strides[perm_inverse[0]]) +
                                           (d1 * output_strides[perm_inverse[1]]) +
                                           (d2 * output_strides[perm_inverse[2]]) +
                                           (d3 * output_strides[perm_inverse[3]]) +
                                           (d4 * output_strides[perm_inverse[4]]));

            const size_t src_byte_index = src_elem_index * elem_byte_size;
            const size_t dst_byte_index = dst_elem_index * elem_byte_size;
            assert(src_byte_index < input_buffer.size());
            assert(dst_byte_index < output_buffer.size());

            std::memcpy(&output_buffer[dst_byte_index], &input_buffer[src_byte_index], elem_byte_size);
          }
        }
      }
    }
  }

  return Status::OK();
}

Status TwoDimensionTranspose(const QnnModelWrapper& qnn_model_wrapper,
                             std::vector<uint32_t>& data_shape,
                             const onnx::TensorProto& initializer,
                             std::vector<uint8_t>& transposed_data) {
  ORT_RETURN_IF_NOT(data_shape.size() == 2, "Expected shape of rank 2");

  std::array<size_t, 2> perm = {1, 0};
  std::vector<uint32_t> output_shape(data_shape.size());
  ORT_RETURN_IF_ERROR((qnn::utils::PermuteShape<uint32_t, size_t>(data_shape, perm, output_shape)));

  auto onnx_type = static_cast<ONNX_NAMESPACE::TensorProto_DataType>(initializer.data_type());
  const size_t elem_byte_size = qnn::utils::GetElementSizeByType(onnx_type);
  ORT_RETURN_IF_NOT(elem_byte_size != 0, "Can't get element byte size from given ONNX type");

  std::vector<uint8_t> input_buffer;
  ORT_RETURN_IF_ERROR(qnn_model_wrapper.UnpackInitializerData(initializer, input_buffer));
  transposed_data.resize(input_buffer.size());

  for (size_t row = 0; row < data_shape[0]; row++) {
    for (size_t col = 0; col < data_shape[1]; col++) {
      const size_t src_elem_index = (row * data_shape[1] + col);
      const size_t dst_elem_index = (col * output_shape[1] + row);
      const size_t src_byte_index = src_elem_index * elem_byte_size;
      const size_t dst_byte_index = dst_elem_index * elem_byte_size;
      assert(src_byte_index < input_buffer.size());
      assert(dst_byte_index < transposed_data.size());

      std::memcpy(&transposed_data[dst_byte_index], &input_buffer[src_byte_index], elem_byte_size);
    }
  }

  data_shape = std::move(output_shape);  // Update parameter with final transposed shape
  return Status::OK();
}

Status TransposeFromNchwToHwcn(const QnnModelWrapper& qnn_model_wrapper,
                               const onnx::TensorProto& initializer,
                               std::vector<uint8_t>& transposed_data,
                               bool is_3d) {
  auto onnx_type = static_cast<ONNX_NAMESPACE::TensorProto_DataType>(initializer.data_type());
  const size_t elem_byte_size = qnn::utils::GetElementSizeByType(onnx_type);
  std::vector<int64_t> input_shape = qnn::utils::GetInitializerShape<int64_t>(initializer);
  std::vector<uint8_t> input_buffer;
  ORT_RETURN_IF_ERROR(qnn_model_wrapper.UnpackInitializerData(initializer, input_buffer));
  transposed_data.resize(input_buffer.size());
  return TransposeFromNchwToHwcn(std::move(input_shape), elem_byte_size, input_buffer, transposed_data, is_3d);
}

Status TransposeFromNchwToHwcn(std::vector<int64_t>&& original_input_shape_dims,
                               size_t elem_byte_size,
                               gsl::span<const uint8_t> input_buffer,
                               gsl::span<uint8_t> output_buffer,
                               bool is_3d) {
  std::vector<int64_t> input_shape_dims = std::move(original_input_shape_dims);
  const size_t rank = input_shape_dims.size();
  ORT_RETURN_IF_NOT((is_3d && rank == 5) || (!is_3d && rank == 4), "Only support input of rank 4 or 5 but got rank ",
                    rank);
  ORT_RETURN_IF_NOT(output_buffer.size() == input_buffer.size(),
                    "Expected output buffer's size to equal the input buffer's size: ",
                    output_buffer.size(), " != ", input_buffer.size());
  ORT_RETURN_IF_NOT(elem_byte_size != 0, "Invalid element byte size due to potentially unsupported type");

  if (!is_3d) {
    input_shape_dims.push_back(1);  // Make it 3D by making shape (N,C,H,W,1)
  }

  return TransposeDataRank5(TensorShape::FromExistingBuffer(input_shape_dims),
                            nchw2hwcn_perm_3d,
                            elem_byte_size,
                            input_buffer,
                            output_buffer);
}

Status TransposeFromCnhwToHwcn(const QnnModelWrapper& qnn_model_wrapper,
                               const onnx::TensorProto& initializer,
                               std::vector<uint8_t>& transposed_data,
                               bool is_3d) {
  auto onnx_type = static_cast<ONNX_NAMESPACE::TensorProto_DataType>(initializer.data_type());
  const size_t elem_byte_size = qnn::utils::GetElementSizeByType(onnx_type);
  std::vector<int64_t> input_shape = qnn::utils::GetInitializerShape<int64_t>(initializer);
  std::vector<uint8_t> input_buffer;
  ORT_RETURN_IF_ERROR(qnn_model_wrapper.UnpackInitializerData(initializer, input_buffer));
  transposed_data.resize(input_buffer.size());
  return TransposeFromCnhwToHwcn(std::move(input_shape), elem_byte_size, input_buffer, transposed_data, is_3d);
}

Status TransposeFromCnhwToHwcn(std::vector<int64_t>&& original_input_shape_dims,
                               size_t elem_byte_size,
                               gsl::span<const uint8_t> input_buffer,
                               gsl::span<uint8_t> output_buffer,
                               bool is_3d) {
  std::vector<int64_t> input_shape_dims = std::move(original_input_shape_dims);
  const size_t rank = input_shape_dims.size();
  ORT_RETURN_IF_NOT((is_3d && rank == 5) || (!is_3d && rank == 4), "Only support input of rank 4 or 5 but got rank ",
                    rank);
  ORT_RETURN_IF_NOT(output_buffer.size() == input_buffer.size(),
                    "Expected output buffer's size to equal the input buffer's size: ",
                    output_buffer.size(), " != ", input_buffer.size());
  ORT_RETURN_IF_NOT(elem_byte_size != 0, "Invalid element byte size due to potentially unsupported type");

  if (!is_3d) {
    input_shape_dims.push_back(1);  // Make it 3D by making shape (C,N,H,W,1)
  }

  return TransposeDataRank5(TensorShape::FromExistingBuffer(input_shape_dims),
                            cnhw2hwcn_perm_3d,
                            elem_byte_size,
                            input_buffer,
                            output_buffer);
}

}  // namespace utils
}  // namespace qnn
}  // namespace onnxruntime
