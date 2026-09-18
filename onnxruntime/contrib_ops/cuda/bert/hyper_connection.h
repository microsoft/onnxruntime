// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/cuda_kernel.h"

namespace onnxruntime::contrib::cuda {

template <typename T>
class BranchwiseRMSNorm final : public onnxruntime::cuda::CudaKernel {
 public:
  explicit BranchwiseRMSNorm(const OpKernelInfo& info);
  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  float epsilon_;
  int64_t num_branches_;
};

template <typename T>
class ScaledSiLU final : public onnxruntime::cuda::CudaKernel {
 public:
  explicit ScaledSiLU(const OpKernelInfo& info);
  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  float alpha_;
};

template <typename T>
class HyperConnectionPreMix final : public onnxruntime::cuda::CudaKernel {
 public:
  explicit HyperConnectionPreMix(const OpKernelInfo& info);
  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  int64_t num_branches_;
  float reduction_scale_;
};

template <typename T>
class HyperConnectionPostMix final : public onnxruntime::cuda::CudaKernel {
 public:
  explicit HyperConnectionPostMix(const OpKernelInfo& info);
  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  int64_t num_branches_;
};

}  // namespace onnxruntime::contrib::cuda
