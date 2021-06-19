// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/cpu_execution_provider.h"
#include "gtest/gtest.h"
#include <onnxruntime_cxx_api.h>
#include <time.h>

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

int GetRand(int min, int max) {
  srand(static_cast<unsigned int>(time(NULL)));
  return rand() % (max - min + 1) + min;
}

void FillInputData(std::vector<TensorData>& input_data) {
  for (auto& entry : input_data) {
    for (size_t i = 0; i < entry.size; i++) {
      int r = GetRand(-100, 100);
      switch (entry.type) {
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
          reinterpret_cast<int64_t*>(entry.buffer.get())[i] = static_cast<int64_t>(r);
          break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
          reinterpret_cast<float*>(entry.buffer.get())[i] = static_cast<float>(r);
          break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
          reinterpret_cast<int32_t*>(entry.buffer.get())[i] = static_cast<int32_t>(r);
          break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
          entry.buffer.get()[i] = static_cast<uint8_t>(r);
          break;
        default:
          ORT_CXX_API_THROW("The input type is not supported for now", ORT_INVALID_ARGUMENT);
      }
    }
  }
}

void GetIOInfo(const Ort::Session& session, OrtAllocator* allocator,
               std::vector<char*>& io_names, std::vector<TensorData>& io_data, bool is_input) {
  io_names.clear();
  io_data.clear();
  size_t num_io = is_input ? session.GetInputCount() : session.GetOutputCount();
  io_names.resize(num_io);
  io_data.resize(num_io);
  for (size_t i = 0; i < num_io; ++i) {
    io_names[i] = is_input ? session.GetInputName(i, allocator) : session.GetOutputName(i, allocator);
    const auto type_info = is_input ? session.GetInputTypeInfo(i) : session.GetOutputTypeInfo(i);
    if (type_info.GetONNXType() != ONNXType::ONNX_TYPE_TENSOR) {
      ORT_CXX_API_THROW("We only accept tensor input", ORT_INVALID_ARGUMENT);
    }

    TensorData& tensor_data = io_data[i];
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
        tensor_data.buffer.reset(new uint8_t[tensor_data.size]);
        break;
      default:
        ORT_CXX_API_THROW("The input type is not supported for now", ORT_INVALID_ARGUMENT);
    }

    size_t dims_count = tensor_type_shape_info.GetDimensionsCount();
    tensor_data.shape.resize(dims_count);
    tensor_type_shape_info.GetDimensions(tensor_data.shape.data(), dims_count);

    // replace dynamic dim to 1
    for (size_t j = 0; j < dims_count; j++) {
      if (tensor_data.shape[j] == -1)
        tensor_data.shape[j] = 1;
    }
  }
}

std::vector<Ort::Value> GetIOTensors(std::vector<TensorData>& io_data) {
  std::vector<Ort::Value> io_tensors;
  io_tensors.reserve(io_data.size());
  auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
  for (auto& entry : io_data) {
    switch (entry.type) {
      case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
        io_tensors.push_back(Ort::Value::CreateTensor<int64_t>(
            memory_info, reinterpret_cast<int64_t*>(entry.buffer.get()), entry.size, entry.shape.data(), entry.shape.size()));
        break;
      case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
        io_tensors.push_back(Ort::Value::CreateTensor<float>(
            memory_info, reinterpret_cast<float*>(entry.buffer.get()), entry.size, entry.shape.data(), entry.shape.size()));
        break;
      case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
        io_tensors.push_back(Ort::Value::CreateTensor<int32_t>(
            memory_info, reinterpret_cast<int32_t*>(entry.buffer.get()), entry.size, entry.shape.data(), entry.shape.size()));
        break;
      case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
        io_tensors.push_back(Ort::Value::CreateTensor<uint8_t>(
            memory_info, reinterpret_cast<uint8_t*>(entry.buffer.get()), entry.size, entry.shape.data(), entry.shape.size()));
        break;
      default:
        ORT_CXX_API_THROW("The input type is not supported for now", ORT_INVALID_ARGUMENT);
    }
  }
  return io_tensors;
}

TEST(CPUExecutionProviderTest, ModelTest) {
  const ORTCHAR_T* model_file_name = ORT_TSTR("testdata/coreml_argmax_cast_test.onnx");
  Ort::AllocatorWithDefaultOptions ort_alloc;

  Ort::SessionOptions so;
  so.SetLogId("ModelTest");
  Ort::RunOptions ro;
  ro.SetRunTag("ModelTest");

  Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "test"};
  Ort::Session session(env, model_file_name, so);

  std::vector<char*> input_names, output_names;
  std::vector<TensorData> input_data, output_data;
  GetIOInfo(session, ort_alloc, input_names, input_data, true /* is_input */);
  GetIOInfo(session, ort_alloc, output_names, output_data, false /* is_input */);

  FillInputData(input_data);
  auto input_tensors = GetIOTensors(input_data);
  auto output_tensors = GetIOTensors(output_data);
  session.Run(ro, input_names.data(), input_tensors.data(), input_names.size(), output_names.data(), output_tensors.data(), output_names.size());

  for (auto& name : input_names)
    ort_alloc.Free(name);
}

class OrtModelRunner {
};

}  // namespace test
}  // namespace onnxruntime
