// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <random>
#include <onnxruntime_cxx_api.h>

#include "cxxopts.hpp"
#include "core/util/math.h"
#include "core/common/common.h"
#include "core/common/logging/logging.h"
#include "core/common/logging/sinks/clog_sink.h"
#include "core/providers/cpu/cpu_execution_provider.h"
#include "core/session/environment.h"
#include "core/session/inference_session.h"
#include "core/providers/cpu/cpu_provider_factory_creator.h"
#include "orttraining/core/framework/tensorboard/event_writer.h"
#include "orttraining/training_api/include/utils.h"
#include "orttraining/training_api/include/interfaces.h"
#include "orttraining/test/training_api/synthetic_data_loader.h"

namespace onnxruntime {
namespace training {
namespace test {
namespace training_api {

SyntheticDataLoader::SyntheticDataLoader(
    size_t sample_count,
    size_t batch_size)
    : sample_count_(sample_count),
      batch_size_(batch_size) {
  ORT_ENFORCE(sample_count >= batch_size && sample_count % batch_size == 0);

  const float scale = 1.f;
  const float mean = 0.f;
  const float seed = 123.f;
  std::default_random_engine generator{gsl::narrow_cast<uint32_t>(seed)};
  std::normal_distribution<float> distribution{mean, scale};
  auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  std::vector<int64_t> dims_input{static_cast<int64_t>(sample_count), static_cast<int64_t>(hidden_size_)};
  input1_.resize(TensorShape(dims_input).Size());
  std::for_each(input1_.begin(), input1_.end(),
                [&generator, &distribution](float& value) { value = distribution(generator); });

  std::vector<int64_t> dims_label{static_cast<int64_t>(sample_count)};
  label_.resize(sample_count, 1);

  num_of_batches_ = sample_count / batch_size;
}

void SyntheticDataLoader::GetNextBatch(std::vector<Ort::Value>& batches) {
  batches.clear();

  if (batch_index_ >= num_of_batches_) {
    batch_index_ = 0;
  }

  auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  size_t input1_offset = batch_index_ * batch_size_ * hidden_size_;
  std::vector<int64_t> dims_input{static_cast<int64_t>(batch_size_), static_cast<int64_t>(hidden_size_)};
  // Be noted: the created Ort::Value won't clean the raw data after its lifetime ended.
  batches.push_back(Ort::Value::CreateTensor<float>(
      memory_info, input1_.data() + input1_offset,
      batch_size_ * hidden_size_, dims_input.data(), dims_input.size()));
  assert(batches[0].IsTensor());

  size_t label_offset = batch_index_ * batch_size_;
  std::vector<int64_t> dims_label{static_cast<int64_t>(batch_size_)};
  batches.push_back(Ort::Value::CreateTensor<int32_t>(
      memory_info, label_.data() + label_offset,
      batch_size_, dims_label.data(), dims_label.size()));
  assert(batches[1].IsTensor());

  batch_index_ += 1;
}

}  // namespace training_api
}  // namespace test
}  // namespace training
}  // namespace onnxruntime