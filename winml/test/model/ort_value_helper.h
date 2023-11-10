#include "testPch.h"
#include "onnxruntime_cxx_api.h"

namespace OrtValueHelpers {
winml::ITensor LoadTensorFromOrtValue(Ort::Value& val);

Ort::Value CreateOrtValueFromITensor(winml::ITensor winmlTensor);
}// namespace OrtValueHelpers

template <ONNXTensorElementDataType T>
struct ONNXTensorElementDataTypeToWinMLTensorKind {
  // Invalid ONNXTensorElementDataType to TensorKind
  static_assert(sizeof(T) == -1, "No WinML TensorKind mapped for given ONNX Tensor Element type!");
};

template <>
struct ONNXTensorElementDataTypeToWinMLTensorKind<ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT> {
  typedef winml::TensorFloat Type;
};

template <>
struct ONNXTensorElementDataTypeToWinMLTensorKind<ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8> {
  typedef winml::TensorUInt8Bit Type;
};

template <>
struct ONNXTensorElementDataTypeToWinMLTensorKind<ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8> {
  typedef winml::TensorInt8Bit Type;
};

template <>
struct ONNXTensorElementDataTypeToWinMLTensorKind<ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16> {
  typedef winml::TensorUInt16Bit Type;
};

template <>
struct ONNXTensorElementDataTypeToWinMLTensorKind<ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16> {
  typedef winml::TensorInt16Bit Type;
};

template <>
struct ONNXTensorElementDataTypeToWinMLTensorKind<ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32> {
  typedef winml::TensorInt32Bit Type;
};

template <>
struct ONNXTensorElementDataTypeToWinMLTensorKind<ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64> {
  typedef winml::TensorInt64Bit Type;
};

template <>
struct ONNXTensorElementDataTypeToWinMLTensorKind<ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING> {
  typedef winml::TensorString Type;
};

template <>
struct ONNXTensorElementDataTypeToWinMLTensorKind<ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL> {
  typedef winml::TensorBoolean Type;
};

template <>
struct ONNXTensorElementDataTypeToWinMLTensorKind<ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16> {
  typedef winml::TensorFloat16Bit Type;
};

template <>
struct ONNXTensorElementDataTypeToWinMLTensorKind<ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE> {
  typedef winml::TensorDouble Type;
};

template <>
struct ONNXTensorElementDataTypeToWinMLTensorKind<ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32> {
  typedef winml::TensorUInt32Bit Type;
};

template <>
struct ONNXTensorElementDataTypeToWinMLTensorKind<ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64> {
  typedef winml::TensorUInt64Bit Type;
};

template <>
struct ONNXTensorElementDataTypeToWinMLTensorKind<ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16> {
  typedef winml::TensorFloat16Bit Type;
};
