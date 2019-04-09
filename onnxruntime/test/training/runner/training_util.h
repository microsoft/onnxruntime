// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <vector>
#include "core/framework/ml_value.h"
#include "core/providers/cpu/cpu_execution_provider.h"

#define RETURN_IF_FAIL(expr)                                \
  do {                                                      \
    auto status = (expr);                                   \
    if ((!status.IsOK())) {                                 \
      printf("Fail: %s \n", status.ErrorMessage().c_str()); \
      return -1;                                            \
    }                                                       \
  } while (0);

namespace onnxruntime {
namespace training {

// A class to hold a dataset.
// which contains:
// 1. tensor names, we make a simple assumption that the last one is the label !!!
// 2. data samples: each sample contains ml_values for all the tensor names above.
class DataSet {
 public:
  typedef std::unique_ptr<std::vector<MLValue>> SampleType;

  typedef typename std::vector<SampleType>::const_iterator IteratorType;

  explicit DataSet(const std::vector<std::string>& tensor_names);

  // Get all tensor names
  const std::vector<std::string> TensorNames() const;

  // Add a data sample
  common::Status AddData(SampleType&& single_sample);

  // Given a batch_size, get the total num of batches.
  size_t TotalBatch(size_t batch_size) const;

  // Given a batch_size, get the [start, end) iterator to access the k-th batch.
  // Caller should make sure no new data is added when using the iterator.
  // Otherwise the iterators are invalid.
  std::pair<IteratorType, IteratorType> KthBatchRange(size_t batch_size, size_t k_th) const;

  // Get the [start, end) iterator to loop over all data.
  std::pair<IteratorType, IteratorType> AllDataRange() const;

  void RandomShuffle();

 private:
  // The names of the tensors.
  std::vector<std::string> tensor_names_;

  // The data of multiple training samples.
  // data_[i] points to a vector of MLValues, whose order matches the above names_.
  std::vector<SampleType> data_;
};

class TrainingUtil {
 public:
  template <typename T>
  static void CreateMLValue(AllocatorPtr alloc,
                            const std::vector<int64_t>& dims,
                            const std::vector<T>& value,
                            MLValue* p_mlvalue) {
    TensorShape shape(dims);
    auto location = alloc->Info();
    auto element_type = DataTypeImpl::GetType<T>();
    void* buffer = alloc->Alloc(element_type->Size() * shape.Size());
    if (value.size() > 0) {
      memcpy(buffer, &value[0], element_type->Size() * shape.Size());
    }

    std::unique_ptr<Tensor> p_tensor = std::make_unique<Tensor>(element_type,
                                                                shape,
                                                                buffer,
                                                                location);
    p_mlvalue->Init(p_tensor.release(),
                    DataTypeImpl::GetType<Tensor>(),
                    DataTypeImpl::GetType<Tensor>()->GetDeleteFunc());
  }

  static AllocatorPtr GetCpuAllocator();
};
}  // namespace training
}  // namespace onnxruntime
