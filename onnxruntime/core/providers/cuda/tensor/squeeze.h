// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/cuda/cuda_kernel.h"
#include "core/providers/cpu/tensor/squeeze.h"

namespace onnxruntime {
namespace cuda {

class Squeeze final : public SqueezeBase, public CudaKernel {
 public:
  Squeeze(const OpKernelInfo& info) : SqueezeBase(info), CudaKernel(info) {}
  Status ComputeInternal(OpKernelContext* context) const override;
};

}  // namespace cuda
}  // namespace onnxruntime
