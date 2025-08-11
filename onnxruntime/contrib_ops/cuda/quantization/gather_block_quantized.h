// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/cuda_kernel.h"

#include <iostream>

using namespace onnxruntime::cuda;

namespace onnxruntime {
namespace contrib {
namespace cuda {

using namespace onnxruntime::cuda;

template <typename T1, typename T2, typename Tind>
class GatherBlockQuantized final : public CudaKernel {
 public:
  GatherBlockQuantized(const OpKernelInfo& info);
  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  int64_t bits_;
  int64_t block_size_;
  int64_t gather_axis_;
  int64_t quantize_axis_;
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
