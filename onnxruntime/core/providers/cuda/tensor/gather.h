// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cpu/tensor/gather.h"

namespace onnxruntime {
namespace cuda {

class Gather final : public CudaKernel, public GatherBase {
 public:
  Gather(const OpKernelInfo& info) : CudaKernel(info), GatherBase(info) {}
  Status ComputeInternal(OpKernelContext* context) const override;
};

}  // namespace cuda
}  // namespace onnxruntime
