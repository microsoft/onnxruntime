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

#include "core/common/gsl.h"

#include <onnxruntime_cxx_api.h>

#include <memory>
#include <utility>
#include <variant>
#include <vector>

namespace onnxruntime {
namespace training {
namespace test {
namespace training_api {

using SyntheticDataVector = std::variant<std::vector<int32_t>, std::vector<int64_t>, std::vector<float>,
                                         std::vector<uint8_t>>;

struct SyntheticInput {
  explicit SyntheticInput(gsl::span<const int64_t> shape) : shape_(shape.begin(), shape.end()) {
    for (auto d : shape) {
      num_of_elements_ *= d;
    }
  }

  size_t NumOfElements() const {
    return num_of_elements_;
  }

  gsl::span<const int64_t> ShapeVector() const {
    return shape_;
  }

  SyntheticDataVector& GetData() {
    return data_;
  }

 private:
  std::vector<int64_t> shape_;
  size_t num_of_elements_{1};
  SyntheticDataVector data_;
};

struct SyntheticSampleBatch {
  SyntheticSampleBatch() = default;

  void AddInt32Input(gsl::span<const int64_t> shape, int32_t low, int32_t high);
  void AddInt64Input(gsl::span<const int64_t> shape, int64_t low, int64_t high);
  void AddFloatInput(gsl::span<const int64_t> shape);
  void AddBoolInput(gsl::span<const int64_t> shape);

  void GetBatch(std::vector<Ort::Value>& batches);

 private:
  template <typename T>
  void AddIntInput(gsl::span<const int64_t> shape, T low, T high);

  std::vector<SyntheticInput> data_vector_;
};

struct SyntheticDataLoader {
  SyntheticDataLoader() = default;

  void AddSyntheticSampleBatch(SyntheticSampleBatch&& samples) {
    sample_batch_collections_.emplace_back(samples);
  }

  bool GetNextSampleBatch(std::vector<Ort::Value>& batches);

  size_t NumOfSampleBatches() const {
    return sample_batch_collections_.size();
  }

  void ResetIterateIndex() {
    sample_batch_iter_index_ = 0;
  }

 private:
  // Be noted: all raw data MUST remain during the training, because all OrtValue created as session inputs
  // did not explicitly copy the data in.
  // And also, the created OrtValue also won't clean the raw data pointer. The raw data should be removed when
  // the life time of this struct ends.
  std::vector<SyntheticSampleBatch> sample_batch_collections_;
  size_t sample_batch_iter_index_{0};
};

}  // namespace training_api
}  // namespace test
}  // namespace training
}  // namespace onnxruntime
