// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <memory>
#include <random>
#include <type_traits>
#include <variant>
#include <vector>

#include "synthetic_data_loader.h"

namespace onnxruntime {
namespace training {
namespace test {
namespace training_api {

namespace {

void RandomFloats(std::vector<float>& rets) {
  const float scale = 1.f;
  const float mean = 0.f;
  const float seed = 123.f;
  static std::default_random_engine generator{static_cast<uint32_t>(seed)};
  std::normal_distribution<float> distribution{mean, scale};
  std::for_each(rets.begin(), rets.end(),
                [&distribution](float& value) { value = distribution(generator); });
}

template <typename IntType>
void RandomInts(std::vector<IntType>& rets, IntType low, IntType high) {
  static std::random_device rd;
  static std::mt19937 generator(rd());
  std::uniform_int_distribution<IntType> distribution(low, high);
  std::for_each(rets.begin(), rets.end(),
                [&distribution](IntType& value) { value = distribution(generator); });
}

}  // namespace

template <typename T>
void SyntheticSampleBatch::AddIntInput(const std::vector<int64_t>& shape, T low, T high) {
  data_vector_.emplace_back(SyntheticInput(shape));

  std::vector<T> values(data_vector_.back().NumOfElements());
  RandomInts(values, low, high);

  SyntheticDataVector& data = data_vector_.back().GetData();
  data = values;
  input_count_++;
}

void SyntheticSampleBatch::AddInt64Input(const std::vector<int64_t>& shape, int64_t low, int64_t high) {
  AddIntInput(shape, low, high);
}

void SyntheticSampleBatch::AddInt32Input(const std::vector<int64_t>& shape, int32_t low, int32_t high) {
  AddIntInput(shape, low, high);
}

void SyntheticSampleBatch::AddBoolInput(const std::vector<int64_t>& shape) {
  // Use uint8_t to store the bool value by intention, because vector<bool> is specialized, we can not create a
  // Tensor leveraging C APIs to reuse the data buffer.
  AddIntInput(shape, static_cast<uint8_t>(0), static_cast<uint8_t>(1));
}

void SyntheticSampleBatch::AddFloatInput(const std::vector<int64_t>& shape) {
  data_vector_.emplace_back(SyntheticInput(shape));

  std::vector<float> values(data_vector_.back().NumOfElements());
  RandomFloats(values);

  SyntheticDataVector& data = data_vector_.back().GetData();
  data = values;
  input_count_++;
}

#define ORT_RETURN_ON_ERROR(expr)                              \
  do {                                                         \
    OrtStatus* onnx_status = (expr);                           \
    if (onnx_status != NULL) {                                 \
      auto code = ort_api->GetErrorCode(onnx_status);          \
      const char* msg = ort_api->GetErrorMessage(onnx_status); \
      printf("Run failed with error code :%d\n", code);        \
      printf("Error message :%s\n", msg);                      \
      ort_api->ReleaseStatus(onnx_status);                     \
      return;                                                  \
    }                                                          \
  } while (0);

bool SyntheticDataLoader::GetNextSampleBatch(std::vector<OrtValue*>& batches) {
  if (sample_batch_iter_index_ >= NumOfSampleBatches()) {
    return false;
  }

  batches.clear();

  Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  auto& sample = sample_batch_collections_[sample_batch_iter_index_];
  const auto* ort_api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  for (size_t i = 0; i < sample.NumOfInput(); ++i) {
    SyntheticInput& input = sample.GetInputAtIndex(i);

    std::visit([&batches, &input, &ort_api, &memory_info](auto&& arg) -> void {
      ONNXTensorElementDataType elem_data_type;
      using T = std::decay_t<decltype(arg)>;
      if constexpr (std::is_same_v<T, std::vector<float>>) {
        elem_data_type = Ort::TypeToTensorType<float>::type;
      } else if constexpr (std::is_same_v<T, std::vector<int32_t>>) {
        elem_data_type = Ort::TypeToTensorType<int32_t>::type;
      } else if constexpr (std::is_same_v<T, std::vector<int64_t>>) {
        elem_data_type = Ort::TypeToTensorType<int64_t>::type;
      } else if constexpr (std::is_same_v<T, std::vector<uint8_t>>) {
        elem_data_type = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL;
      } else {
        throw std::runtime_error("Unsupported element type.");
      }

      void* p_data = arg.data();
      OrtValue* value = nullptr;
      auto shape_vector = input.ShapeVector();
      // Be noted: the created OrtValue won't clean the raw data after its lifetime ended.
      ORT_RETURN_ON_ERROR(ort_api->CreateTensorWithDataAsOrtValue(
          memory_info,
          p_data, (input.NumOfElements() * sizeof(T)),
          shape_vector.data(), shape_vector.size(),
          elem_data_type,
          &value));

      batches.emplace_back(value);
    },
               input.GetData());
  }

  sample_batch_iter_index_ += 1;
  return true;
}

}  // namespace training_api
}  // namespace test
}  // namespace training
}  // namespace onnxruntime
