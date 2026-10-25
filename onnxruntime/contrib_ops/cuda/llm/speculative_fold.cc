// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/llm/speculative_fold.h"
#include "core/providers/cuda/cuda_kernel.h"
#include "core/common/safeint.h"
#include <climits>

namespace onnxruntime::contrib::cuda {
using namespace onnxruntime::cuda;
namespace {

bool Count(const Tensor* tensor, int64_t count) {
  return tensor && tensor->Shape().Size() == count;
}

class SpeculativePrefixCommit final : public CudaKernel {
 public:
  explicit SpeculativePrefixCommit(const OpKernelInfo& info) : CudaKernel(info) {
    head_rows_ = SafeInt<int>(info.GetAttrOrDefault<int64_t>("head_rows", 2));
    capacity_ = SafeInt<int>(info.GetAttrOrDefault<int64_t>("head_capacity", 262144));
    ORT_ENFORCE(head_rows_ >= 2 && head_rows_ <= 8 && capacity_ >= head_rows_,
                "SpeculativePrefixCommit: invalid head_rows/head_capacity");
  }

  Status ComputeInternal(OpKernelContext* context) const override {
    const auto* tokens = context->Input<Tensor>(0);
    const auto* ids = context->Input<Tensor>(1);
    const auto* status = context->Input<Tensor>(2);
    const auto* ctl = context->Input<Tensor>(3);
    const auto* committed = context->Input<Tensor>(4);
    const auto* prefixes = context->Input<Tensor>(5);
    const auto* prefix2 = context->Input<Tensor>(6);
    const auto* ptrs = context->Input<Tensor>(7);
    const auto* cosine = context->Input<Tensor>(8);
    const auto* sine = context->Input<Tensor>(9);
    ORT_RETURN_IF_NOT(tokens && ids && status && ctl && committed && prefixes && prefix2 && ptrs && cosine && sine,
                      "SpeculativePrefixCommit: missing required input");
    const int64_t rows = tokens->Shape().Size();
    ORT_RETURN_IF_NOT(rows > 0 && rows <= INT_MAX && ids->Shape().Size() >= rows &&
                          status->Shape().Size() <= INT_MAX,
                      "SpeculativePrefixCommit: invalid token/status shape");
    ORT_RETURN_IF_NOT(ctl->Shape().Size() >= (head_rows_ > 2 ? 16 : 8) &&
                          Count(committed, 2) && prefixes->Shape().Size() >= (head_rows_ > 2 ? 2 * head_rows_ : 2) &&
                          Count(prefix2, 2) && Count(ptrs, 2),
                      "SpeculativePrefixCommit: invalid control/context shape");
    const auto& rope = cosine->Shape();
    ORT_RETURN_IF_NOT(rope.NumDimensions() == 2 && rope[0] > 0 && rope[0] <= INT_MAX &&
                          rope[1] > 0 && rope[1] <= INT_MAX && sine->Shape() == rope,
                      "SpeculativePrefixCommit: rotary caches must have matching [positions, width] shapes");
    auto* out_ids = context->Output(0, TensorShape({1, head_rows_}));
    auto* head_c = context->Output(1, TensorShape({1}));
    auto* slots = context->Output(2, TensorShape({1, head_rows_}));
    auto* lengths = context->Output(3, TensorShape({1}));
    auto* out_cos = context->Output(4, TensorShape({1, head_rows_, rope[1]}));
    auto* out_sin = context->Output(5, TensorShape({1, head_rows_, rope[1]}));
    auto* selection = context->Output(6, TensorShape({1}));
    int64_t* ctl_ptr = const_cast<int64_t*>(ctl->Data<int64_t>());
    int64_t* context_ptr = const_cast<int64_t*>(committed->Data<int64_t>());
    if (context->OutputCount() > 7) {
      auto* updated_ctl = context->Output(7, ctl->Shape());
      if (updated_ctl) {
        ctl_ptr = updated_ctl->MutableData<int64_t>();
      }
    }
    if (context->OutputCount() > 8) {
      auto* updated_context = context->Output(8, committed->Shape());
      if (updated_context) {
        context_ptr = updated_context->MutableData<int64_t>();
      }
    }
    return speculative_fold::LaunchMid(
        tokens->Data<int64_t>(), ids->Data<int64_t>(), status->Data<int32_t>(), static_cast<int>(status->Shape().Size()),
        ctl_ptr, context_ptr, prefixes->Data<int64_t>(),
        prefix2->Data<int64_t>(), ptrs->Data<int64_t>(),
        reinterpret_cast<const uint16_t*>(cosine->Data<BFloat16>()),
        reinterpret_cast<const uint16_t*>(sine->Data<BFloat16>()),
        out_ids->MutableData<int64_t>(), head_c->MutableData<int32_t>(), slots->MutableData<int64_t>(),
        lengths->MutableData<int32_t>(), reinterpret_cast<uint16_t*>(out_cos->MutableData<BFloat16>()),
        reinterpret_cast<uint16_t*>(out_sin->MutableData<BFloat16>()), selection->MutableData<int64_t>(),
        static_cast<int>(rows), head_rows_, static_cast<int>(rope[0]), static_cast<int>(rope[1]), capacity_, Stream(context),
        true, ctl->Data<int64_t>(), static_cast<int>(ctl->Shape().Size()), committed->Data<int64_t>());
  }

 private:
  int head_rows_;
  int capacity_;
};
}  // namespace

ONNX_OPERATOR_KERNEL_EX(
    SpeculativePrefixCommit, kMSDomain, 1, kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .MayInplace(3, 7)
        .MayInplace(4, 8)
        .TypeConstraint("I64", DataTypeImpl::GetTensorType<int64_t>())
        .TypeConstraint("I32", DataTypeImpl::GetTensorType<int32_t>())
        .TypeConstraint("B", DataTypeImpl::GetTensorType<BFloat16>()),
    SpeculativePrefixCommit);

}  // namespace onnxruntime::contrib::cuda
