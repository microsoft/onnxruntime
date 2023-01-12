// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/cuda/cuda_kernel.h"
#include "core/providers/cpu/tensor/split.h"

namespace onnxruntime {
namespace cuda {

class SplitKernel : public CudaKernel, public SplitBase {
 public:
  SplitKernel(const OpKernelInfo& info, uint32_t opset) : CudaKernel(info), SplitBase(info, opset) {}

  Status ComputeInternal(OpKernelContext* context) const override;
};

// versions 2, 11 and 13
class Split_2_13 final : public SplitKernel {
 public:
  // use opset 1 for all versions earlier than 18
  Split_2_13(const OpKernelInfo& info) : SplitKernel(info, /* opset */ 1) {}
};

class Split_18 final : public SplitKernel {
 public:
  Split_18(const OpKernelInfo& info) : SplitKernel(info, 18) {}
};

}  // namespace cuda
}  // namespace onnxruntime
