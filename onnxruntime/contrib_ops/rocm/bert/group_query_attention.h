// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>
#include "core/providers/rocm/rocm_kernel.h"

namespace onnxruntime {
namespace contrib {
namespace rocm {

using namespace onnxruntime::rocm;

template <typename T>
class GroupQueryAttention final : public RocmKernel {
 public:
  GroupQueryAttention(const OpKernelInfo& info);
  Status ComputeInternal(OpKernelContext* context) const override;

 protected:
  int num_heads_;     // number of attention heads
  int kv_num_heads_;  // different for k and v for group query attention
  int local_window_size_;
  bool is_unidirectional_;
  bool is_past_bsnh_;
  bool do_rotary_;
  bool rotary_interleaved_;
  float scale_;
};

}  // namespace rocm
}  // namespace contrib
}  // namespace onnxruntime
