// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

/**
 * @brief Synthetic sample batch generator.
 *
 * Avoid introducing non-public data structure dependencies here, as beside our internal unit tests, this header will
 * also be included by trainer which only use our public C API, has no hard dependencies on ORT internal libs.
 *
 */

#pragma once

#include <onnxruntime_cxx_api.h>

#include <memory>
#include <utility>
#include <variant>
#include <vector>

namespace onnxruntime {
namespace training {
namespace test {
namespace training_api {

typedef std::vector<std::variant<int32_t, int64_t, float, uint8_t>> SyntheticDataVector;

struct TypedSyntheticInput {
  explicit TypedSyntheticInput(const std::vector<int64_t>& shape) : shape_(shape) {
    for (auto d : shape) {
      num_of_elements_ *= d;
    }
  }

  size_t NumOfElements() {
    return num_of_elements_;
  }

  std::vector<int64_t> ShapeVector() const {
    return shape_;
  }

  SyntheticDataVector& GetData() {
    return data_;
  }

 protected:
  std::vector<int64_t> shape_;
  size_t num_of_elements_{1};
  SyntheticDataVector data_;
};

struct SyntheticSampleBatch {
  SyntheticSampleBatch() {}

  void AddInt32Input(const std::vector<int64_t>& shape, int32_t low, int32_t high);
  void AddInt64Input(const std::vector<int64_t>& shape, int64_t low, int64_t high);
  void AddFloatInput(const std::vector<int64_t>& shape);
  void AddBoolInput(const std::vector<int64_t>& shape);

  size_t NumOfInput() {
    return data_vector_.size();
  }

  TypedSyntheticInput& GetInputAtIndex(size_t index) {
    return data_vector_[index];
  }

 private:
  template <typename T>
  void AddIntInput(const std::vector<int64_t>& shape, T low, T high);

  std::vector<TypedSyntheticInput> data_vector_;
};

struct SyntheticDataLoader {
  SyntheticDataLoader() {}

  void AddSyntheticSampleBatch(std::unique_ptr<SyntheticSampleBatch> samples) {
    sample_batch_collections_.emplace_back(std::move(samples));
  }

  bool GetNextSampleBatch(std::vector<OrtValue*>& batches);

  size_t NumOfSampleBatches() {
    return sample_batch_collections_.size();
  }

  void ResetIterateIndex() {
    sample_batch_iter_index_ = 0;
  }

 private:
  template <typename T>
  OrtValue* CreateTensorWithData(const OrtApi* ort_api, Ort::MemoryInfo& memory_info, TypedSyntheticInput& input);

  // Be noted: all raw data MUST remain during the training, because all OrtValue created as session inputs
  // did not explicitly copy the data in.
  // And also, the created OrtValue also won't clean the raw data pointer. The raw data should be removed when
  // the life time of this struct ends.
  std::vector<std::unique_ptr<SyntheticSampleBatch>> sample_batch_collections_;
  size_t sample_batch_iter_index_{0};
};

}  // namespace training_api
}  // namespace test
}  // namespace training
}  // namespace onnxruntime
