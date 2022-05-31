// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <random>

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
  std::uniform_int_distribution<> distribution(low, high);
  std::for_each(rets.begin(), rets.end(),
                [&distribution](IntType& value) { value = distribution(generator); });
}

}  // namespace

void SyntheticSampleBatch::AddInt64Input(const std::vector<int64_t>& shape, int64_t low, int64_t high) {
  data_vector_.emplace_back(std::make_unique<TypedSynctheticInput<int64_t>>(shape));
  RandomInts(data_vector_.back()->GetData<int64_t>(), low, high);
}

void SyntheticSampleBatch::AddInt32Input(const std::vector<int64_t>& shape, int32_t low, int32_t high) {
  data_vector_.emplace_back(std::make_unique<TypedSynctheticInput<int32_t>>(shape));
  RandomInts(data_vector_.back()->GetData<int32_t>(), low, high);
}

void SyntheticSampleBatch::AddFloatInput(const std::vector<int64_t>& shape) {
  data_vector_.emplace_back(std::make_unique<TypedSynctheticInput<float>>(shape));
  RandomFloats(data_vector_.back()->GetData<float>());
}

bool SyntheticDataLoader::GetNextSampleBatch(std::vector<Ort::Value>& batches) {
  if (sample_batch_iter_index_ >= num_of_sample_batches) {
    return false;
  }

  batches.clear();

  auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  auto& sample = sample_batch_collections_[sample_batch_iter_index_];
  for (size_t i = 0; i < sample->NumOfInput(); ++i) {
    auto input_ptr = sample->GetInputAtIndex(i);
    auto shape_vector = input_ptr->ShapeVector();
    // Be noted: the created Ort::Value won't clean the raw data after its lifetime ended.
    auto ptr_flt = dynamic_cast<TypedSynctheticInput<float>*>(input_ptr);
    if (ptr_flt) {
      batches.push_back(Ort::Value::CreateTensor<float>(
          memory_info, input_ptr->GetData<float>().data(),
          input_ptr->NumOfBytesPerSample(), shape_vector.data(), shape_vector.size()));
      continue;
    }

    auto ptr_int = dynamic_cast<TypedSynctheticInput<int64_t>*>(input_ptr);
    if (ptr_int) {
      batches.push_back(Ort::Value::CreateTensor<int64_t>(
          memory_info, input_ptr->GetData<int64_t>().data(),
          input_ptr->NumOfBytesPerSample(), shape_vector.data(), shape_vector.size()));
      continue;
    }

    auto ptr_int32 = dynamic_cast<TypedSynctheticInput<int32_t>*>(input_ptr);
    if (ptr_int32) {
      batches.push_back(Ort::Value::CreateTensor<int32_t>(
          memory_info, input_ptr->GetData<int32_t>().data(),
          input_ptr->NumOfBytesPerSample(), shape_vector.data(), shape_vector.size()));
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