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
  if (single_sample->size() != NumInputs()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "DataSet::AddData failed");
  }

  data_.emplace_back(move(single_sample));
  return Status::OK();
}

size_t DataSet::TotalBatch(size_t batch_size) const {
  batch_size = min(batch_size, data_.size());
  return data_.size() / batch_size + ((data_.size() % batch_size > 0) ? 1 : 0);
}

std::vector<MLValue> DataSet::GetKthBatch(size_t batch_size, size_t k_th) const {
  if (batch_size == 1) {
    return *data_[k_th];
  }

  batch_size = min(batch_size, data_.size());
  AllocatorPtr alloc = TrainingUtil::GetCpuAllocator();
  auto location = alloc->Info();

  std::vector<MLValue> result;
  for (int input_index = 0; input_index < NumInputs(); ++input_index) {
    const Tensor& first_tensor = data_[0]->at(input_index).Get<Tensor>();

    MLDataType element_type = first_tensor.DataType();
    TensorShape shape = first_tensor.Shape();
    shape[0] = batch_size;

    size_t memory_size_per_sample = first_tensor.Size();
    size_t buffer_size = memory_size_per_sample * batch_size;
    void* buffer = alloc->Alloc(buffer_size);
    void* origin_buffer = buffer;

    size_t offset = k_th * batch_size;
    for (size_t i = offset; i < offset + batch_size; ++i) {
      size_t index = (offset + i) % NumSamples();
      const void* raw_value = data_[index]->at(input_index).Get<Tensor>().DataRaw();
      memcpy(buffer, raw_value, memory_size_per_sample);
      buffer = static_cast<char*>(buffer) + memory_size_per_sample;
    }

    std::unique_ptr<Tensor> p_tensor = std::make_unique<Tensor>(element_type,
                                                                shape,
                                                                origin_buffer,
                                                                location);

    result.emplace_back(p_tensor.release(),
                        DataTypeImpl::GetType<Tensor>(),
                        DataTypeImpl::GetType<Tensor>()->GetDeleteFunc());
  }

  return result;
}

void DataSet::RandomShuffle() {
  random_shuffle(data_.begin(), data_.end());
}

AllocatorPtr TrainingUtil::GetCpuAllocator() {
  static CPUExecutionProviderInfo info;
  static CPUExecutionProvider cpu_provider(info);
  return cpu_provider.GetAllocator(0, OrtMemTypeDefault);
}

void TrainingUtil::PrintNameMLValMap(const NameMLValMap& mlvalue_map) {
  for (auto pair : mlvalue_map) {
    auto name = pair.first;
    MLValue value = pair.second;
    const Tensor& tensor = value.Get<Tensor>();

    printf("Name: %s \n", name.c_str());
    const float* data = tensor.template Data<float>();
    int64_t size = tensor.Shape().Size();
    for (int i = 0; i < size; ++i) {
      printf("%0.04f\t ", *data);
      data++;
    }
    printf("\n\n");
  }
}

}  // namespace training
}  // namespace onnxruntime
