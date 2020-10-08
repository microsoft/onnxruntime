#include "testPch.h"
#include "ort_value_helper.h"
using namespace winml;

namespace OrtValueHelpers {
template <ONNXTensorElementDataType T>
winml::ITensor CopyTensorData(Ort::Value& val) {
  using WinMLTensorKind = typename ONNXTensorElementDataTypeToWinMLTensorKind<T>::Type;
  ITensor tensor = nullptr;
  WINML_EXPECT_NO_THROW(tensor = WinMLTensorKind::Create(val.GetTensorTypeAndShapeInfo().GetShape()));
  void* actualData = nullptr;
  uint32_t actualSizeInBytes = 0;
  WINML_EXPECT_NO_THROW(tensor.as<ITensorNative>()->GetBuffer(reinterpret_cast<BYTE**>(&actualData), &actualSizeInBytes));
  void* ortValueTensorData = nullptr;
  WINML_EXPECT_NO_THROW(Ort::GetApi().GetTensorMutableData(val, &ortValueTensorData));
  WINML_EXPECT_NO_THROW(memcpy(actualData, ortValueTensorData, actualSizeInBytes * sizeof(char)));
  return tensor;
}

// This function takes in an Ort::Value and returns a copy of winml::ITensor
// TODO: String types still need to be implemented.
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
    case (ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16): {
      return CopyTensorData<ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16>(val);
    }
    case (ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16): {
      return CopyTensorData<ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16>(val);
    }
    case (ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32): {
      return CopyTensorData<ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32>(val);
    }
    case (ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64): {
      return CopyTensorData<ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64>(val);
    }
    case (ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL): {
      return CopyTensorData<ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL>(val);
    }
    case (ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16): {
      return CopyTensorData<ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16>(val);
    }
    case (ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE): {
      return CopyTensorData<ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE>(val);
    }
    case (ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32): {
      return CopyTensorData<ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32>(val);
    }
    case (ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64): {
      return CopyTensorData<ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64>(val);
    }
    case (ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16): {
      return CopyTensorData<ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16>(val);
    }
    default:
      throw winrt::hresult_invalid_argument(L"TensorType not implemented yet.");
  }
}
}  // namespace OrtValueHelpers