// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <vector>

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "mrotary_embedding_helper.h"

namespace onnxruntime {
namespace contrib {

template <typename T>
class MRotaryEmbedding final : public OpKernel {
 public:
  MRotaryEmbedding(const OpKernelInfo& info);
  Status Compute(OpKernelContext* context) const override;

 protected:
  float scale;
  int num_heads;
  int rotary_embedding_dim;
  bool interleaved;
  bool is_packed_batching;
  std::vector<int64_t> mrope_section;
  int64_t mrope_layout;
};

}  // namespace contrib
}  // namespace onnxruntime
