#include "testPch.h"
#include "ort_value_helper.h"
#include "StringHelpers.h"
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

static int64_t ShapeSize(const int64_t* shape, size_t count) {
  // for each dim
  int64_t size = 1;
  for (size_t i = 0; i < count; i++) {
    // find out it's total size
    size *= shape[i];
    // make sure there are no invalid dimensions (-1 or any invalid shape)
    THROW_HR_IF(E_INVALIDARG, shape[i] <= 0);
  }
  return size;
}

winml::ITensor CreateStringTensor(Ort::Value& val) {
  size_t dimensionCount = 0;
  WINML_EXPECT_NO_THROW(dimensionCount = val.GetTensorTypeAndShapeInfo().GetDimensionsCount());
  std::vector<int64_t> shape;
  if (dimensionCount > 0) {
    WINML_EXPECT_NO_THROW(shape = val.GetTensorTypeAndShapeInfo().GetShape());
  }
  auto length = ShapeSize(shape.data(), shape.size());

  // make a big buffer to hold all the string data
  size_t bufferLength = 0;
  WINML_EXPECT_NO_THROW(bufferLength = val.GetStringTensorDataLength());

  std::vector<winrt::hstring> strings;
  std::unique_ptr<uint8_t[]> buffer(new uint8_t[bufferLength]);
  std::vector<size_t> offsets(static_cast<size_t>(length));

  WINML_EXPECT_NO_THROW(val.GetStringTensorContent(buffer.get(), bufferLength, offsets.data(), offsets.size()));

   // now go build all the strings
  for (auto i = 0; i < length; ++i) {
    size_t strLength = 0;
    // are we on the last one?
    if (i == (length - 1)) {
      strLength = bufferLength - offsets[i];
    } else {
      strLength = offsets[i+1] - offsets[i];
    }
    auto strView = std::string_view(reinterpret_cast<const char*>(buffer.get() + offsets[i]), strLength);
    strings.push_back(_winml::Strings::HStringFromUTF8(strView.data(), strLength));
  }

  TensorString tensor =  nullptr;
  WINML_EXPECT_NO_THROW(tensor = TensorString::CreateFromShapeArrayAndDataArray(shape, strings));
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
    case (ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING): {
      return CreateStringTensor(val);
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