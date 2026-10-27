// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/llm/speculative_fold.h"
#include "core/providers/cuda/cuda_kernel.h"
#include "core/common/safeint.h"
#include <climits>

namespace onnxruntime::contrib::cuda {
using namespace onnxruntime::cuda;
namespace {

constexpr int64_t kCtlWords = 16;

// Aliased (identical) buffers are supported; partially overlapping ones are not.
bool IdenticalOrDisjoint(const Tensor& a, const Tensor& b) {
  const auto* a_begin = static_cast<const char*>(a.DataRaw());
  const auto* b_begin = static_cast<const char*>(b.DataRaw());
  return a_begin == b_begin || a_begin + a.SizeInBytes() <= b_begin || b_begin + b.SizeInBytes() <= a_begin;
}

class SpeculativePrefixCommit final : public CudaKernel {
 public:
  explicit SpeculativePrefixCommit(const OpKernelInfo& info) : CudaKernel(info) {
    head_rows_ = SafeInt<int>(info.GetAttrOrDefault<int64_t>("head_rows", 8));
    ORT_ENFORCE(head_rows_ >= 3 && head_rows_ <= 8, "SpeculativePrefixCommit: head_rows must be in [3, 8]");
  }

  Status ComputeInternal(OpKernelContext* context) const override {
    const auto* tokens = context->Input<Tensor>(0);
    const auto* ids = context->Input<Tensor>(1);
    const auto* status = context->Input<Tensor>(2);
    const auto* ctl = context->Input<Tensor>(3);
    const auto* committed = context->Input<Tensor>(4);
    const auto* prefixes = context->Input<Tensor>(5);
    const auto* ptrs = context->Input<Tensor>(6);
    ORT_RETURN_IF_NOT(tokens && ids && status && ctl && committed && prefixes && ptrs,
                      "SpeculativePrefixCommit: missing required input");
    const auto& tok_shape = tokens->Shape();
    const size_t tok_rank = tok_shape.NumDimensions();
    ORT_RETURN_IF_NOT(tok_rank == 1 || tok_rank == 2,
                      "SpeculativePrefixCommit: next_tokens must have rank 1 ([rows]) or 2 ([batch, rows])");
    const int64_t batch = tok_rank == 2 ? tok_shape[0] : int64_t{1};
    const int64_t rows = tok_shape[tok_rank - 1];
    ORT_RETURN_IF_NOT(batch > 0 && batch <= INT_MAX && rows > 0 && rows <= INT_MAX &&
                          ids->Shape().Size() >= batch * rows && ids->Shape().Size() % batch == 0 &&
                          ids->Shape().Size() / batch <= INT_MAX &&
                          status->Shape().Size() <= INT_MAX,
                      "SpeculativePrefixCommit: invalid token/status shape");
    ORT_RETURN_IF_NOT(ctl->Shape().Size() >= batch * kCtlWords && ctl->Shape().Size() % batch == 0 &&
                          ctl->Shape().Size() / batch <= INT_MAX &&
                          committed->Shape().Size() == 2 * batch &&
                          prefixes->Shape().Size() >= batch * 2 * head_rows_ && prefixes->Shape().Size() % batch == 0 &&
                          prefixes->Shape().Size() / batch <= INT_MAX &&
                          ptrs->Shape().Size() == 2,
                      "SpeculativePrefixCommit: invalid control/context shape");
    auto* head_c = context->Output(0, TensorShape({batch}));
    auto* updated_ctl = context->Output(1, ctl->Shape());
    auto* updated_context = context->Output(2, committed->Shape());
    ORT_RETURN_IF_NOT(head_c && updated_ctl && updated_context,
                      "SpeculativePrefixCommit: updated_ctl and updated_ctx_commit outputs are required");
    ORT_RETURN_IF_NOT(IdenticalOrDisjoint(*ctl, *updated_ctl) && IdenticalOrDisjoint(*committed, *updated_context),
                      "SpeculativePrefixCommit: state outputs must alias or be disjoint from their inputs");
    return speculative_fold::LaunchPrefixCommit(
        tokens->Data<int64_t>(), ids->Data<int64_t>(), status->Data<int32_t>(),
        static_cast<int>(status->Shape().Size()), ctl->Data<int64_t>(), updated_ctl->MutableData<int64_t>(),
        committed->Data<int64_t>(), updated_context->MutableData<int64_t>(), prefixes->Data<int64_t>(),
        ptrs->Data<int64_t>(), head_c->MutableData<int32_t>(), static_cast<int>(rows), head_rows_,
        static_cast<int>(batch), static_cast<int>(ids->Shape().Size() / batch),
        static_cast<int>(ctl->Shape().Size() / batch), static_cast<int>(prefixes->Shape().Size() / batch),
        Stream(context));
  }

 private:
  int head_rows_;
};
}  // namespace

ONNX_OPERATOR_KERNEL_EX(
    SpeculativePrefixCommit, kMSDomain, 1, kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .MayInplace(3, 1)
        .MayInplace(4, 2)
        .TypeConstraint("I64", DataTypeImpl::GetTensorType<int64_t>())
        .TypeConstraint("I32", DataTypeImpl::GetTensorType<int32_t>()),
    SpeculativePrefixCommit);

}  // namespace onnxruntime::contrib::cuda
