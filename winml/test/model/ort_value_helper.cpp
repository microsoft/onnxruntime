#include "testPch.h"
#include "ort_value_helper.h"
using namespace winml;

namespace OrtValueHelpers {
template <ONNXTensorElementDataType T>
winml::ITensor CopyTensorData(Ort::Value& val) {
  using WinMLTensorKind = typename ONNXTensorElementDataTypeToWinMLTensorKind<T>::Type;
  auto tensor = WinMLTensorKind::Create(val.GetTensorTypeAndShapeInfo().GetShape());
  void* actualData;
  uint32_t actualSizeInBytes;
  tensor.as<ITensorNative>()->GetBuffer(reinterpret_cast<BYTE**>(&actualData), &actualSizeInBytes);
  void* ortValueTensorData = nullptr;
  Ort::GetApi().GetTensorMutableData(val, &ortValueTensorData);
  memcpy(actualData, ortValueTensorData, actualSizeInBytes * sizeof(char));
  return tensor;
}

winml::ITensor LoadTensorFromOrtValue(Ort::Value& val) {
  auto tensorTypeAndShape = val.GetTensorTypeAndShapeInfo();
  switch (tensorTypeAndShape.GetElementType()) {
    case (ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT): {
      return CopyTensorData<ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT>(val);
    }
    case (ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8): {
      return CopyTensorData<ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8>(val);
    }
    case (ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8): {
      return CopyTensorData<ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8>(val);
    }
    default:
      throw winrt::hresult_invalid_argument(L"TensorType not implemented yet.");
  }
}
}  // namespace OrtValueHelpers