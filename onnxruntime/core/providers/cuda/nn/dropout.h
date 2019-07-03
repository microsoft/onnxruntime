// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/common/common.h"
#include "core/platform/ort_mutex.h"
#include "core/framework/op_kernel.h"
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/rnn/cudnn_rnn_base.h"

namespace onnxruntime {
namespace cuda {

struct CudnnDropoutState {
  cudnnDropoutDescriptor_t dropout_desc = nullptr;
  cudnnTensorDescriptor_t dropout_in_out_desc = nullptr;
  size_t dropout_state_size;
  size_t dropout_reserve_size;
  void* states;
  void* dropout_reserve_space;

  OrtMutex mutex;

  Status Set(cudnnHandle_t handle, const TensorShape& shape, cudnnDataType_t type, float ratio);
  Status Release();
};

template <typename T>
class TrainableDropout final : public CudaKernel {
 public:
  TrainableDropout(const OpKernelInfo& info) : CudaKernel{info} {
    info.GetAttrOrDefault("seed", &seed_, static_cast<float>(0.5));
  }

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  float seed_;
  mutable CudnnDropoutState s_;
  const float default_ratio_ = 0.5f;
};

template <typename T>
class TrainableDropoutGrad final : public CudaKernel {
 public:
  TrainableDropoutGrad(const OpKernelInfo& info) : CudaKernel{info} {
    info.GetAttrOrDefault("seed", &seed_, static_cast<float>(0.5));
  }

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  float seed_;
  mutable CudnnDropoutState s_;
  const float default_ratio_ = 0.5f;
};
}  // namespace cuda
}  // namespace onnxruntime
