// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/cpu_execution_provider.h"
#include "gtest/gtest.h"
#include <onnxruntime_cxx_api.h>

namespace onnxruntime {
namespace test {
TEST(CPUExecutionProviderTest, MetadataTest) {
  CPUExecutionProviderInfo info;
  auto provider = std::make_unique<CPUExecutionProvider>(info);
  EXPECT_TRUE(provider != nullptr);
  ASSERT_STREQ(provider->GetAllocator(0, OrtMemTypeDefault)->Info().name, CPU);
}

struct TensorData {
  ONNXTensorElementDataType type{ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED};
  std::unique_ptr<uint8_t[]> buffer;
  size_t size{0};
  std::vector<int64_t> shape;
};

TEST(CPUExecutionProviderTest, ModelTest) {
  const ORTCHAR_T* model_file_name = ORT_TSTR("testdata/coreml_argmax_cast_test.onnx");
  Ort::AllocatorWithDefaultOptions ort_alloc;

  Ort::SessionOptions so;
  so.SetLogId("ModelTest");

  Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "test"};
  Ort::Session session(env, model_file_name, so);

  size_t num_input = session.GetInputCount();
  std::vector<char*> input_name(num_input);
  std::vector<TensorData> input_data(num_input);
  for (size_t i = 0; i < num_input; ++i) {
    input_name[i] = session.GetInputName(i, ort_alloc);
    const auto type_info = session.GetInputTypeInfo(i);
    if (type_info.GetONNXType() != ONNXType::ONNX_TYPE_TENSOR) {
      ORT_CXX_API_THROW("We only accept tensor input", ORT_INVALID_ARGUMENT);
    }

    TensorData tensor_data;
    const auto& tensor_type_shape_info = type_info.GetTensorTypeAndShapeInfo();
    tensor_data.type = tensor_type_shape_info.GetElementType();
    tensor_data.size = tensor_type_shape_info.GetElementCount();
    switch (tensor_data.type) {
      case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
        tensor_data.buffer.reset(new uint8_t[tensor_data.size * 8]);
        break;
      case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
        tensor_data.buffer.reset(new uint8_t[tensor_data.size * 4]);
        break;
      case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
      case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
        tensor_data.buffer.reset(new uint8_t[tensor_data.size]);
        break;
      default:
        ORT_CXX_API_THROW("The input type is not supported for now", ORT_INVALID_ARGUMENT);
    }

    size_t dims_count = tensor_type_shape_info.GetDimensionsCount();
    tensor_data.shape.resize(dims_count);
    tensor_type_shape_info.GetDimensions(tensor_data.shape.data(), dims_count);

    // replace dynamic dim to 1
    for (size_t i = 0; i < dims_count; i++) {
      if (tensor_data.shape[i] == -1)
        tensor_data.shape[i] = 1;
    }
  }
}

}  // namespace test
}  // namespace onnxruntime
