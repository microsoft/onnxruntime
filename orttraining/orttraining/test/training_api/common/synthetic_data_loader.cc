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
  data_vector_.emplace_back(TypedSyntheticInput(shape));

  std::vector<T> values(data_vector_.back().NumOfElements());
  RandomInts(values, low, high);

  auto& data = data_vector_.back().GetData();
  data.reserve(values.size());
  for (size_t i = 0; i < values.size(); ++i) {
    data.push_back(values[i]);
  }
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
  data_vector_.emplace_back(TypedSyntheticInput(shape));

  std::vector<float> values(data_vector_.back().NumOfElements());
  RandomFloats(values);

  auto& data = data_vector_.back().GetData();
  for (size_t i = 0; i < values.size(); ++i) {
    data[i] = values[i];
  }
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
      return nullptr;                                          \
    }                                                          \
  } while (0);

template <typename T>
OrtValue* SyntheticDataLoader::CreateTensorWithData(const OrtApi* ort_api, Ort::MemoryInfo& memory_info,
                                                    TypedSyntheticInput& input) {
  std::variant<int32_t, int64_t, float, uint8_t>& first_elem = input.GetData()[0];

  if (T* fval = std::get_if<T>(&first_elem)) {
    ONNXTensorElementDataType elem_data_type;
    if (std::is_same<float, T>::value) {
      elem_data_type = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    } else if (std::is_same<int32_t, T>::value) {
      elem_data_type = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
    } else if (std::is_same<int64_t, T>::value) {
      elem_data_type = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
    } else if (std::is_same<uint8_t, T>::value) {
      elem_data_type = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL;
    } else {
      throw std::runtime_error("Unsupported element type.");
    }

    OrtValue* value = nullptr;
    auto shape_vector = input.ShapeVector();
    // Be noted: the created OrtValue won't clean the raw data after its lifetime ended.
    ORT_RETURN_ON_ERROR(ort_api->CreateTensorWithDataAsOrtValue(
        memory_info,
        input.GetData().data(), (input.NumOfElements() * sizeof(T)),
        shape_vector.data(), shape_vector.size(),
        elem_data_type,
        &value));
    return value;
  }

  return nullptr;
}

bool SyntheticDataLoader::GetNextSampleBatch(std::vector<OrtValue*>& batches) {
  if (sample_batch_iter_index_ >= NumOfSampleBatches()) {
    return false;
  }

  batches.clear();

  Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  auto& sample = sample_batch_collections_[sample_batch_iter_index_];
  const auto* ort_api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  for (size_t i = 0; i < sample->NumOfInput(); ++i) {
    auto& input = sample->GetInputAtIndex(i);

    OrtValue* value = CreateTensorWithData<float>(ort_api, memory_info, input);
    if (value) {
      batches.emplace_back(value);
      continue;
    }

    value = CreateTensorWithData<int32_t>(ort_api, memory_info, input);
    if (value) {
      batches.emplace_back(value);
      continue;
    }

    value = CreateTensorWithData<int64_t>(ort_api, memory_info, input);
    if (value) {
      batches.emplace_back(value);
      continue;
    }

    value = CreateTensorWithData<uint8_t>(ort_api, memory_info, input);
    if (value) {
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
