// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/common.h"
#include "core/common/narrow.h"
#include "core/framework/op_kernel.h"
#include "core/platform/threadpool.h"
#include "core/util/math_cpuonly.h"
#include <functional>

#include <vector>

namespace onnxruntime {
namespace contrib {
class MultiHeadLatentAttention final : public OpKernel {
 public:
  explicit MultiHeadLatentAttention(const OpKernelInfo& info) : OpKernel(info) {

  }

  Status Compute(OpKernelContext* ctx) const override;
};


ONNX_OPERATOR_KERNEL_EX(
    MultiHeadLatentAttention,
    kMSDomain,
    1,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T", BuildKernelDefConstraints<float>()),
    MultiHeadLatentAttention);


Status MultiHeadLatentAttention::Compute(OpKernelContext* ctx) const {
  const auto& qk_nope = ctx->Input<Tensor>(0);
  const auto& past_key = ctx->Input<Tensor>(8);
  const auto& past_value = ctx->Input<Tensor>(9);

  auto* output = ctx->Output(0, qk_nope->Shape());

  // Assume past - present buffer sharing
  auto* present_key = ctx->Output(1, past_key->Shape());
//   auto* present_value = ctx->Output(2, past_value->Shape());

  return Status::OK();
}

}  // namespace contrib
}  // namespace onnxruntime
