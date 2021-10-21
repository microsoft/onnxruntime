// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/cuda/cuda_kernel.h"
#include "core/providers/cpu/tensor/split.h"

namespace onnxruntime {
namespace cuda {

class Split final : public CudaKernel, public SplitBase {
 public:
  Split(const OpKernelInfo& info) : CudaKernel(info), SplitBase(info) {}
  Status ComputeInternal(OpKernelContext* context) const override;
};

}  // namespace cuda
}  // namespace onnxruntime
