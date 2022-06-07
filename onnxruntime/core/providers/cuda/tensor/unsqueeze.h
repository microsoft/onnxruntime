// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/cuda/cuda_kernel.h"
#include "core/providers/cpu/tensor/unsqueeze.h"

namespace onnxruntime {
namespace cuda {

class Unsqueeze final : public UnsqueezeBase, public CudaKernel {
 public:
  Unsqueeze(const OpKernelInfo& info) : UnsqueezeBase(info), CudaKernel(info) {}
  Status ComputeInternal(OpKernelContext* context) const override;
};

}  // namespace cuda
}  // namespace onnxruntime
