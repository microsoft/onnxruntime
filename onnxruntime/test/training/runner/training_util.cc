// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "test/training/runner/training_util.h"

using namespace std;
namespace onnxruntime {
namespace training {

DataSet::DataSet(const vector<string>& tensor_names) : tensor_names_(tensor_names) {
}

const vector<string> DataSet::TensorNames() const {
  return tensor_names_;
}

common::Status DataSet::AddData(DataSet::SampleType&& single_sample) {
  if (single_sample->size() != tensor_names_.size()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "DataSet::AddData failed");
  }

  data_.emplace_back(move(single_sample));
  return Status::OK();
}

size_t DataSet::TotalBatch(size_t batch_size) const {
  batch_size = min(batch_size, data_.size());
  return data_.size() / batch_size + ((data_.size() % batch_size > 0) ? 1 : 0);
}

pair<DataSet::IteratorType, DataSet::IteratorType> DataSet::KthBatchRange(size_t batch_size, size_t k_th) const {
  batch_size = min(batch_size, data_.size());

  auto startIt = data_.cbegin();
  advance(startIt, min(data_.size(), batch_size * k_th));

  auto endIt = data_.cbegin();
  advance(endIt, min(data_.size(), batch_size * k_th + batch_size));
  return {startIt, endIt};
}

pair<DataSet::IteratorType, DataSet::IteratorType> DataSet::AllDataRange() const {
  return {data_.cbegin(), data_.cend()};
}

void DataSet::RandomShuffle() {
  random_shuffle(data_.begin(), data_.end());
}

AllocatorPtr TrainingUtil::GetCpuAllocator() {
  static CPUExecutionProviderInfo info;
  static CPUExecutionProvider cpu_provider(info);
  return cpu_provider.GetAllocator(0, OrtMemTypeDefault);
}

}  // namespace training
}  // namespace onnxruntime
