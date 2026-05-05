// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/cuda_kernel.h"
#include "core/providers/cpu/tensor/space_depth_ops.h"

namespace onnxruntime {
namespace cuda {

template <bool Layout>
class SpaceToDepth final : public CudaKernel, SpaceDepthBase {
 public:
  explicit SpaceToDepth(const OpKernelInfo& info)
      : CudaKernel(info), SpaceDepthBase(info) {}

  Status ComputeInternal(OpKernelContext* context) const override;
};

template <bool Layout>
class DepthToSpace final : public CudaKernel, SpaceDepthBase {
 public:
  explicit DepthToSpace(const OpKernelInfo& info)
      : CudaKernel(info), SpaceDepthBase(info), is_dcr_(space_depth_internal::ReadIsDCR(info)) {}

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  bool is_dcr_ = true;
};

}  // namespace cuda
}  // namespace onnxruntime
