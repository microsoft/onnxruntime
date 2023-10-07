// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/common/common.h"
#include "core/framework/op_kernel.h"

namespace onnxruntime {
namespace contrib {

template <typename T>
class RotaryEmbedding final : public OpKernel {
 public:
  RotaryEmbedding(const OpKernelInfo& info);
  Status Compute(OpKernelContext* context) const override;

 protected:
  float scale;
  bool interleaved;
};

}  // namespace contrib
}  // namespace onnxruntime
