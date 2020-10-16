#include "testPch.h"
#include "ort_value_helper.h"
using namespace winml;

namespace OrtValueHelpers {

template <ONNXTensorElementDataType T>
winml::ITensor CreateTensorFromShape(std::vector<int64_t>& shape)
{
  using WinMLTensorKind = typename ONNXTensorElementDataTypeToWinMLTensorKind<T>::Type;
  ITensor tensor = nullptr;
  WINML_EXPECT_NO_THROW(tensor = WinMLTensorKind::Create(shape));
  return tensor;
}

// This function takes in an Ort::Value and returns a copy of winml::ITensor
// TODO: String types still need to be implemented.
winml::ITensor LoadTensorFromOrtValue(Ort::Value& val) {
  ITensor tensor = nullptr;
  auto tensorTypeAndShape = val.GetTensorTypeAndShapeInfo();
  auto shape = tensorTypeAndShape.GetShape();
  switch (tensorTypeAndShape.GetElementType()) {
    case (ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT): {
      tensor = CreateTensorFromShape<ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT>(shape);
      break;
    }
    case (ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8): {
      tensor = CreateTensorFromShape<ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8>(shape);
      break;
    }
    case (ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8): {
      tensor = CreateTensorFromShape<ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8>(shape);
    }
    case (ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16): {
      tensor = CreateTensorFromShape<ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16>(shape);
      break;
    }
    case (ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16): {
      tensor = CreateTensorFromShape<ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16>(shape);
      break;
    }
    case (ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32): {
      tensor = CreateTensorFromShape<ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32>(shape);
      break;
    }
    case (ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64): {
      tensor = CreateTensorFromShape<ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64>(shape);
      break;
    }
    case (ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL): {
      tensor = CreateTensorFromShape<ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL>(shape);
      break;
    }
    case (ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16): {
      tensor = CreateTensorFromShape<ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16>(shape);
      break;
    }
    case (ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE): {
      tensor = CreateTensorFromShape<ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE>(shape);
      break;
    }
    case (ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32): {
      tensor = CreateTensorFromShape<ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32>(shape);
      break;
    }
    case (ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64): {
      tensor = CreateTensorFromShape<ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64>(shape);
      break;
    }
    case (ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16): {
      tensor = CreateTensorFromShape<ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16>(shape);
      break;
    }
    default:
      throw winrt::hresult_invalid_argument(L"TensorType not implemented yet.");
  }
  BYTE* actualData = nullptr;
  uint32_t actualSizeInBytes = 0;
  WINML_EXPECT_NO_THROW(tensor.as<ITensorNative>()->GetBuffer(&actualData, &actualSizeInBytes));
  void* ortValueTensorData = nullptr;
  WINML_EXPECT_NO_THROW(Ort::GetApi().GetTensorMutableData(val, &ortValueTensorData));
  WINML_EXPECT_NO_THROW(memcpy(actualData, ortValueTensorData, actualSizeInBytes * sizeof(char)));
  return tensor;
}
}  // namespace OrtValueHelpers