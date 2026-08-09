// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/cuda/cuda_kernel.h"
#include "core/providers/cpu/tensor/padbase.h"

using onnxruntime::PadBase;

namespace onnxruntime {
namespace cuda {

template <typename T>
class Pad final : public PadBase, public CudaKernel {
 public:
  Pad(const OpKernelInfo& info) : PadBase(info), CudaKernel(info) {}

  Status ComputeInternal(OpKernelContext* context) const override;
};

}  // namespace cuda
}  // namespace onnxruntime
