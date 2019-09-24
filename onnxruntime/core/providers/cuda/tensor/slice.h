// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/common.h"
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cpu/tensor/slice.h"
#include "core/providers/cpu/tensor/utils.h"

namespace onnxruntime {
namespace cuda {

template<typename Tind, bool dynamic>
class Slice final : public CudaKernel, public SliceBase {
 public:
  Slice(const OpKernelInfo& info) : CudaKernel(info), SliceBase(info, dynamic) {}

  Status ComputeInternal(OpKernelContext* ctx) const override;
};

}  // namespace cuda
}  // namespace onnxruntime
