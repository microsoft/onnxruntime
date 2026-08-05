// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/rocm/rocm_kernel.h"
#include "core/framework/int4.h"

namespace onnxruntime {
namespace contrib {
namespace rocm {

using namespace onnxruntime::rocm;

template <typename T1, typename T2, typename Tind>
class GatherBlockQuantized final : public RocmKernel {
 public:
  GatherBlockQuantized(const OpKernelInfo& info);
  Status ComputeInternal(OpKernelContext* ctx) const override;

 private:
  int64_t bits_;
  int64_t block_size_;
  int64_t gather_axis_;
  int64_t quantize_axis_;
};

}  // namespace rocm
}  // namespace contrib
}  // namespace onnxruntime
