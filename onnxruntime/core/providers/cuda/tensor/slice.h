// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/common.h"
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cpu/tensor/slice.h"

namespace onnxruntime {
namespace cuda {

class Slice final : public CudaKernel, public SliceBase {
 public:
  Slice(const OpKernelInfo& info) : CudaKernel(info), SliceBase(info) {}
  Status ComputeInternal(OpKernelContext* context) const override;
};

}  // namespace cuda
}  // namespace onnxruntime
