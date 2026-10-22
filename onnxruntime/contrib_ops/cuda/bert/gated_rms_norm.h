// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/cuda_kernel.h"

namespace onnxruntime::contrib::cuda {

class SigmoidMul final : public onnxruntime::cuda::CudaKernel {
 public:
  explicit SigmoidMul(const OpKernelInfo& info) : CudaKernel(info) {}
  Status ComputeInternal(OpKernelContext* context) const override;
};

Status LaunchSigmoidMul(cudaStream_t stream, const void* input, const void* gate,
                        void* output, int64_t count);

class GatedRMSNormExact final : public onnxruntime::cuda::CudaKernel {
 public:
  explicit GatedRMSNormExact(const OpKernelInfo& info);
  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  float epsilon_;
};

Status LaunchGatedRMSNorm(cudaStream_t stream, const void* input, const void* gate,
                          const float* scale, void* output, int rows, float epsilon);

}  // namespace onnxruntime::contrib::cuda
