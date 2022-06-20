// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/shared_library/provider_api.h"
#include "core/common/common.h"
#include "core/providers/cuda/cuda_kernel.h"
#include "core/providers/cpu/tensor/concatbase.h"

namespace onnxruntime {
namespace cuda {

class Concat final : public CudaKernel, public ConcatBase {
 public:
  Concat(const OpKernelInfo& info) : CudaKernel(info), ConcatBase(info) {}
  Status ComputeInternal(OpKernelContext* context) const override;
};

}  // namespace cuda
}  // namespace onnxruntime
