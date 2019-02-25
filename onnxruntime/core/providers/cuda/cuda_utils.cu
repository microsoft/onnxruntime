// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Thrust code needs to be compiled with nvcc
#include <memory>
#include <mutex>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include "core/providers/cuda/shared_inc/cuda_utils.h"
#include "core/providers/cuda/cu_inc/common.cuh"
#include "cudnn_common.h"

namespace onnxruntime {
namespace cuda {

template <typename T>
class ConstantBufferImpl : public IConstantBuffer<T> {
 public:
  ConstantBufferImpl(T val) : val_(val) {}

  virtual const T* GetBuffer(size_t count) {
    buffer_.resize(count);
    thrust::fill(buffer_.begin(), buffer_.end(), val_);
    return buffer_.data().get();
  }

 private:
  thrust::device_vector<T> buffer_;
  T val_;
};

template <typename T>
std::unique_ptr<IConstantBuffer<T>> CreateConstantOnes() {
  return std::make_unique<ConstantBufferImpl<T>>(Consts<T>::One);
}

template std::unique_ptr<IConstantBuffer<float>> CreateConstantOnes<float>();
template std::unique_ptr<IConstantBuffer<double>> CreateConstantOnes<double>();
template std::unique_ptr<IConstantBuffer<half>> CreateConstantOnes<half>();

}  // namespace cuda
}  // namespace onnxruntime
