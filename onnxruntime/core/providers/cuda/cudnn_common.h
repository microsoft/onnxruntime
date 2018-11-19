// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "cuda_common.h"
#include "core/framework/tensor.h"
#include <cfloat>

namespace onnxruntime {
namespace cuda {

class CudnnTensor final {
 public:
  CudnnTensor();
  ~CudnnTensor();

  Status Set(const std::vector<int64_t>& input_dims, cudnnDataType_t dataType);
  Status Set(const CudnnTensor& x_desc, cudnnBatchNormMode_t mode);

  operator cudnnTensorDescriptor_t() const { return tensor_; }

  template <typename T>
  static cudnnDataType_t GetDataType();

 private:
  Status CreateTensorIfNeeded();

  cudnnTensorDescriptor_t tensor_;
};

class CudnnFilterDescriptor final {
 public:
  CudnnFilterDescriptor();
  ~CudnnFilterDescriptor();

  Status Set(const std::vector<int64_t>& filter_dims, cudnnDataType_t data_typ);

  operator cudnnFilterDescriptor_t() const { return desc_; }

 private:
  cudnnFilterDescriptor_t desc_;
};

template <typename ElemType>
struct Consts {
  static const ElemType Zero;
  static const ElemType One;
};

template <>
struct Consts<half> {
  static const float Zero;
  static const float One;
};

inline double ClampCudnnBatchNormEpsilon(double epsilon) {
  if (epsilon < CUDNN_BN_MIN_EPSILON) {
    if (CUDNN_BN_MIN_EPSILON - epsilon > FLT_EPSILON)
      LOGS_DEFAULT(WARNING) << "Provided epsilon is smaller than CUDNN_BN_MIN_EPSILON. Setting it to CUDNN_BN_MIN_EPSILON";
    return CUDNN_BN_MIN_EPSILON;
  }
  return epsilon;
}

}  // namespace cuda
}  // namespace onnxruntime
