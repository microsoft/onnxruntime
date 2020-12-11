// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
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
