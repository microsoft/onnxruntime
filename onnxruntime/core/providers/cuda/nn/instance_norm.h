// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/cuda_kernel.h"
#include "core/providers/cuda/cudnn_common.h"

namespace onnxruntime {
namespace cuda {

template <typename T>
class InstanceNorm final : public CudaKernel {
 public:
  InstanceNorm(const OpKernelInfo& op_kernel_info);
  Status ComputeInternal(OpKernelContext* p_op_kernel_context) const override;

 private:
  double epsilon_;
  // mutex for set cudnn stream
  mutable OrtMutex cudnn_stream_mutex_;
};

}  // namespace cuda
}  // namespace onnxruntime
