#include "testPch.h"
#include "ort_value_helper.h"
#include "StringHelpers.h"
using namespace winml;
using namespace winrt::Windows::Foundation::Collections;
namespace OrtValueHelpers {

template <ONNXTensorElementDataType T>
winml::ITensor CreateTensorFromShape(std::vector<int64_t>& shape) {
  using WinMLTensorKind = typename ONNXTensorElementDataTypeToWinMLTensorKind<T>::Type;
  ITensor tensor = nullptr;
  WINML_EXPECT_NO_THROW(tensor = WinMLTensorKind::Create(shape));
  return tensor;
}

static uint64_t ShapeSize(const int64_t* shape, size_t count) {
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
  for (size_t i = 0; i < length; ++i) {
    size_t strLength = 0;
    // are we on the last one?
    if (i == (length - 1)) {
      strLength = bufferLength - offsets[i];
    } else {
      strLength = offsets[i + 1] - offsets[i];
    }
    auto strView = std::string_view(reinterpret_cast<const char*>(buffer.get() + offsets[i]), strLength);
    strings.push_back(_winml::Strings::HStringFromUTF8(strView.data(), strLength));
  }

  TensorString tensor = nullptr;
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

static ONNXTensorElementDataType OnnxTensorTypeFromWinMLType(winml::TensorKind tensorKind) {
  switch (tensorKind) {
    case (TensorKind::Float):
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    case (TensorKind::UInt8):
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
    case (TensorKind::Int8):
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8;
    case (TensorKind::UInt16):
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16;
    case (TensorKind::Int16):
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16;
    case (TensorKind::Int32):
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
    case (TensorKind::Int64):
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
    case (TensorKind::Boolean):
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL;
    case (TensorKind::Float16):
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
    case (TensorKind::Double):
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE;
    case (TensorKind::UInt32):
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32;
    case (TensorKind::UInt64):
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64;
    case (TensorKind::Complex64):
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64;
    case (TensorKind::Complex128):
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128;
    default:
      throw std::invalid_argument("No conversion from WinML Type into Onnx TensorType");
  }
}

Ort::Value CreateOrtValueFromITensor(winml::ITensor winmlTensor) {
  Ort::Value ortValueCreated = Ort::Value{nullptr};
  auto memoryInfo = Ort::MemoryInfo{nullptr};
  WINML_EXPECT_NO_THROW(memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault));
  std::vector<int64_t> shape;
  auto vectorViewShape = winmlTensor.Shape();
  for (int64_t dimension : vectorViewShape) {
    shape.push_back(dimension);
  }
  if (winmlTensor.TensorKind() != winml::TensorKind::String) {
    auto winmlTensorNative = winmlTensor.as<ITensorNative>();
    BYTE* actualData;
    uint32_t actualSizeInBytes;
    WINML_EXPECT_HRESULT_SUCCEEDED(winmlTensorNative->GetBuffer(&actualData, &actualSizeInBytes));
    WINML_EXPECT_NO_THROW(
      ortValueCreated = Ort::Value::CreateTensor(
        memoryInfo,
        actualData,
        actualSizeInBytes,
        shape.data(),
        shape.size(),
        OnnxTensorTypeFromWinMLType(winmlTensor.TensorKind())
      )
    );
  } else {
    Ort::AllocatorWithDefaultOptions allocator;
    WINML_EXPECT_NO_THROW(
      ortValueCreated = Ort::Value::CreateTensor(
        allocator, shape.data(), shape.size(), ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING
      )
    );
    std::vector<const char*> strData;
    std::vector<std::string> utf8Strs;
    auto strValues = winmlTensor.as<TensorString>().GetAsVectorView();
    for (winrt::hstring str : strValues) {
      utf8Strs.push_back(_winml::Strings::UTF8FromHString(str));
      strData.push_back(utf8Strs.back().c_str());
    }
    WINML_EXPECT_NO_THROW(ortValueCreated.FillStringTensor(strData.data(), strData.size()));
  }
  return ortValueCreated;
}
}  // namespace OrtValueHelpers
