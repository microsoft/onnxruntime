// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/rnn/cudnn_rnn_base.h"
#include "core/platform/ort_mutex.h"

namespace onnxruntime {
namespace cuda {

class DropoutBase : public CudaKernel {
 protected:
  struct CudnnDropoutState {
    explicit CudnnDropoutState(cudnnHandle_t handle);
    cudnnDropoutDescriptor_t dropout_desc;
    cudnnTensorDescriptor_t dropout_in_out_desc;
    size_t dropout_state_size;
    size_t dropout_reserve_size;
    void* states;
    void* dropout_reserve_space;
    OrtMutex mutex;
    float ratio_;
    Status Set(cudnnHandle_t handle, const TensorShape& shape, cudnnDataType_t type, float ratio);
    ~CudnnDropoutState();
  };

  DropoutBase(const OpKernelInfo& info) : CudaKernel{info}, s_(CudnnHandle()), default_ratio_(0.5) {}
  ~DropoutBase() = default;
  //TODO: We need to change this. The kernel should be stateless, which means we should not have mutable field.
  mutable CudnnDropoutState s_;
  const float default_ratio_;
};

template <typename T>
class DropoutCudnn final : public DropoutBase {
 public:
  DropoutCudnn(const OpKernelInfo& info) : DropoutBase{info} {
    info.GetAttrOrDefault("seed", &seed_, static_cast<float>(0.5));
  }

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  float seed_;
};

template <typename T>
class DropoutCudnnGrad final : public DropoutBase {
 public:
  DropoutCudnnGrad(const OpKernelInfo& info) : DropoutBase{info} {
  }

  Status ComputeInternal(OpKernelContext* context) const override;
};

}  // namespace cuda
}  // namespace onnxruntime
