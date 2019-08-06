// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/providers/cuda/cuda_common.h"

namespace onnxruntime {
namespace cuda {

using namespace onnxruntime::cuda;

class GatherNDBase : public CudaKernel {
 public:
  GatherNDBase(const OpKernelInfo& info) : CudaKernel(info) {
    info.GetAttrOrDefault("axis", &axis_, static_cast<int64_t>(0));
  }

template <typename Tind>
  Status CommonComputeKernel(
      const int64_t last_indice_dimension,
      const int64_t axis,
      const TensorShape& input_shape,
      const Tensor* input_tensor,
      Tensor* output_tensor,
      const TensorShape& indice_shape,
      const Tensor* indice_tensor,
      const bool fwd) const;
 protected:
  int64_t axis_;
};

template <typename Tind>
class GatherND final : public GatherNDBase{
 public:
  GatherND(const OpKernelInfo& info) : GatherNDBase(info) {}
  Status ComputeInternal(OpKernelContext* context) const override;
};

template <typename Tind>
class GatherNDGrad final : public GatherNDBase{
 public:
  GatherNDGrad(const OpKernelInfo& info) : GatherNDBase(info) {
  }
  Status ComputeInternal(OpKernelContext* context) const override;
};

}  // namespace cuda
}  // namespace onnxruntime