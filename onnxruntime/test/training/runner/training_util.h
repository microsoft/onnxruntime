// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <vector>
#include "core/graph/onnx_protobuf.h"
#include "core/framework/ml_value.h"
#include "core/framework/framework_common.h"
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
// 2. data samples: each sample contains ort_values for all the tensor names above.
class DataSet {
 public:
  typedef std::unique_ptr<std::vector<OrtValue>> SampleType;

  typedef typename std::vector<SampleType>::const_iterator IteratorType;

  DataSet(const std::vector<std::string>& tensor_names);

  virtual ~DataSet();

  // Get all tensor names
  const std::vector<std::string> TensorNames() const;

  size_t NumInputs() const { return tensor_names_.size(); }

  common::Status AddData(SampleType&& single_sample);

  common::Status AddData(const std::vector<ONNX_NAMESPACE::TensorProto>& features);

  virtual size_t NumSamples() const { return data_.size(); }

  // Given a batch_size, get the total num of batches.
  size_t TotalBatch(size_t batch_size) const;

  virtual std::vector<OrtValue> GetKthBatch(size_t batch_size, size_t k_th) const;

  void RandomShuffle();

 private:
  // The names of the tensors.
  std::vector<std::string> tensor_names_;

  // The data of multiple training samples.
  // data_[i] points to a vector of ORTValues, whose order matches the above names_.
  std::vector<SampleType> data_;

  std::vector<std::unique_ptr<char[]>> ortvalue_buffers_;

  std::vector<OrtCallback> ortvalue_deleters_;
};

class RandomDataSet : public DataSet {
 public:
  explicit RandomDataSet(int num_samples,
                         const std::vector<std::string>& tensor_names,
                         const std::vector<TensorShape> tensor_shapes,
                         const std::vector<onnx::TensorProto_DataType> tensor_types)
      : DataSet(tensor_names),
        num_samples_(num_samples),
        tensor_shapes_(tensor_shapes),
        tensor_types_(tensor_types){};

  virtual ~RandomDataSet() {}

  virtual size_t NumSamples() const override { return num_samples_; }

  virtual std::vector<OrtValue> GetKthBatch(size_t batch_size, size_t k_th) const override;

 private:
  int num_samples_;
  const std::vector<TensorShape> tensor_shapes_;
  const std::vector<onnx::TensorProto_DataType> tensor_types_;
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

  static void PrintNameMLValMap(const NameMLValMap& mlvalue_map);

  static void PrintTensor(const std::string& name, const Tensor& tensor, std::ostream& os = std::cout);
};
}  // namespace training
}  // namespace onnxruntime
