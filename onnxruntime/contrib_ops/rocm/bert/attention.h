// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/rocm/rocm_kernel.h"
#include "contrib_ops/cpu/bert/attention_base.h"
#include "contrib_ops/rocm/bert/attention_impl.h"

namespace onnxruntime {
namespace contrib {
namespace rocm {

using namespace onnxruntime::rocm;

template <typename T>
class Attention final : public RocmKernel, public AttentionBase {
 public:
  Attention(const OpKernelInfo& info);
  Status ComputeInternal(OpKernelContext* context) const override;

 public:
  AttentionType attn_type_;

  // type-erased GemmSoftmaxGemmPermuteTunableOp<HipT>, the reason for this is:
  //   1. We don't want to include the cuh file where GemmSoftmaxGemmPermuteTunableOp<HipT> is defined.
  //   2. We don't want to construct the object repeatly (which is expansive) during Compute.
  std::shared_ptr<void> tunable_op_;
};

}  // namespace rocm
}  // namespace contrib
}  // namespace onnxruntime
