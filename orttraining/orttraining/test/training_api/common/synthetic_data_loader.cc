// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <memory>
#include <random>
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

void SyntheticSampleBatch::AddInt64Input(const std::vector<int64_t>& shape, int64_t low, int64_t high) {
  data_vector_.emplace_back(std::make_unique<TypedSyntheticInput<int64_t>>(shape));
  RandomInts(data_vector_.back()->GetData<int64_t>(), low, high);
}

void SyntheticSampleBatch::AddInt32Input(const std::vector<int64_t>& shape, int32_t low, int32_t high) {
  data_vector_.emplace_back(std::make_unique<TypedSyntheticInput<int32_t>>(shape));
  RandomInts(data_vector_.back()->GetData<int32_t>(), low, high);
}

void SyntheticSampleBatch::AddFloatInput(const std::vector<int64_t>& shape) {
  data_vector_.emplace_back(std::make_unique<TypedSyntheticInput<float>>(shape));
  RandomFloats(data_vector_.back()->GetData<float>());
}

#define ORT_RETURN_ON_ERROR(expr)                                \
  do {                                                           \
    OrtStatus* onnx_status = (expr);                             \
    if (onnx_status != NULL) {                                   \
      auto code = ort_api->GetErrorCode(onnx_status);          \
      const char* msg = ort_api->GetErrorMessage(onnx_status); \
      ort_api->ReleaseStatus(onnx_status);                     \
      printf("Run failed with error code :%d\n", code);          \
      printf("Error message :%s\n", msg);                        \
      return false;                                                 \
    }                                                            \
  } while (0);

bool SyntheticDataLoader::GetNextSampleBatch(std::vector<OrtValue*>& batches) {
  if (sample_batch_iter_index_ >= NumOfSampleBatches()) {
    return false;
  }

  batches.clear();

  auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  auto& sample = sample_batch_collections_[sample_batch_iter_index_];
  const auto* ort_api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  for (size_t i = 0; i < sample->NumOfInput(); ++i) {
    auto input_ptr = sample->GetInputAtIndex(i);
    auto shape_vector = input_ptr->ShapeVector();
    // Be noted: the created OrtValue won't clean the raw data after its lifetime ended.
    auto ptr_flt = dynamic_cast<TypedSyntheticInput<float>*>(input_ptr);
    if (ptr_flt) {
      OrtValue* value = NULL;
      ORT_RETURN_ON_ERROR(ort_api->CreateTensorWithDataAsOrtValue(memory_info,
      input_ptr->GetData<float>().data(), (input_ptr->NumOfElements() * sizeof(float)),
      shape_vector.data(), shape_vector.size(),
      ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
      &value));
      batches.emplace_back(value);
      continue;
    }

    auto ptr_int = dynamic_cast<TypedSyntheticInput<int64_t>*>(input_ptr);
    if (ptr_int) {
      OrtValue* value = NULL;
      ORT_RETURN_ON_ERROR(ort_api->CreateTensorWithDataAsOrtValue(memory_info,
      input_ptr->GetData<int64_t>().data(), (input_ptr->NumOfElements() * sizeof(int64_t)),
      shape_vector.data(), shape_vector.size(),
      ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
      &value));
      batches.emplace_back(value);
      continue;
    }

    auto ptr_int32 = dynamic_cast<TypedSyntheticInput<int32_t>*>(input_ptr);
    if (ptr_int32) {
      OrtValue* value = nullptr;
      ORT_RETURN_ON_ERROR(ort_api->CreateTensorWithDataAsOrtValue(memory_info,
      input_ptr->GetData<int32_t>().data(), (input_ptr->NumOfElements() * sizeof(int32_t)),
      shape_vector.data(), shape_vector.size(),
      ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,
      &value));
      batches.emplace_back(value);
      continue;
    }

    throw std::runtime_error("unknown data types.");
  }

  sample_batch_iter_index_ += 1;
  return true;
}

}  // namespace training_api
}  // namespace test
}  // namespace training
}  // namespace onnxruntime
