// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/common/common.h"
#include "core/providers/cuda/cuda_kernel.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

using namespace onnxruntime::cuda;

template <typename T>
class RelPosAttnBias final : public CudaKernel {
 public:
  RelPosAttnBias(const OpKernelInfo& op_kernel_info);
  Status ComputeInternal(OpKernelContext* ctx) const override;

 private:
  int max_distance_;
  bool is_bidirectional_;
};

template <typename T>
class GatedRelativePositionBias final : public CudaKernel {
 public:
  GatedRelativePositionBias(const OpKernelInfo& op_kernel_info);

  Status ComputeInternal(OpKernelContext* ctx) const override;

 private:
  int num_heads_;
};


}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
