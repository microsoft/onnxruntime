// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/llm/speculative_fold.h"

#include <limits>

#include "core/common/common.h"
#include "core/providers/cuda/cuda_kernel.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

using namespace onnxruntime::cuda;

namespace {

bool HasElementCount(const Tensor* tensor, int64_t count) {
  return tensor != nullptr && tensor->Shape().Size() == count;
}

class FoldMid2 final : public CudaKernel {
 public:
  explicit FoldMid2(const OpKernelInfo& info)
      : CudaKernel(info), head_rows_(info.GetAttrOrDefault<int64_t>("head_rows", 2)) {}

  Status ComputeInternal(OpKernelContext* context) const override {
    const Tensor* next_tokens = context->Input<Tensor>(0);
    const Tensor* input_ids = context->Input<Tensor>(1);
    const Tensor* status = context->Input<Tensor>(2);
    const Tensor* ctl = context->Input<Tensor>(3);
    const Tensor* ctx_commit = context->Input<Tensor>(4);
    const Tensor* ctx_a = context->Input<Tensor>(5);
    const Tensor* ctx_b = context->Input<Tensor>(6);
    const Tensor* ptrs = context->Input<Tensor>(7);
    const Tensor* cos_cache = context->Input<Tensor>(8);
    const Tensor* sin_cache = context->Input<Tensor>(9);
    const Tensor* hidden_states = context->InputCount() > 10 ? context->Input<Tensor>(10) : nullptr;
    ORT_ENFORCE(next_tokens != nullptr && input_ids != nullptr && status != nullptr &&
                ctl != nullptr && ctx_commit != nullptr && ctx_a != nullptr &&
                ctx_b != nullptr && ptrs != nullptr && cos_cache != nullptr &&
                sin_cache != nullptr);

    const int64_t rows = next_tokens->Shape().Size();
    ORT_RETURN_IF_NOT(rows >= 1 && input_ids->Shape().Size() >= 1 &&
                          HasElementCount(ctl, 8) &&
                          HasElementCount(ctx_commit, 2) &&
                          ctx_a->Shape().Size() >= 2 &&
                          HasElementCount(ctx_b, 2) &&
                          HasElementCount(ptrs, 2),
                      "FoldMid2 received an invalid control tensor shape");
    ORT_RETURN_IF_NOT(cos_cache->Shape().NumDimensions() == 2 &&
                          sin_cache->Shape().NumDimensions() == 2 &&
                          cos_cache->Shape()[1] == 32 &&
                          sin_cache->Shape()[1] == 32 &&
                          cos_cache->Shape()[0] == sin_cache->Shape()[0],
                      "FoldMid2 rotary caches must have matching shape [positions, 32]");
    const int64_t status_count = status->Shape().Size();
    ORT_RETURN_IF_NOT(status_count <= std::numeric_limits<int>::max(),
                      "FoldMid2 status tensor is too large");

    Tensor* head_ids = context->Output(0, TensorShape({1, head_rows_}));
    Tensor* head_c = context->Output(1, TensorShape({1}));
    Tensor* slot_ids = context->Output(2, TensorShape({1, head_rows_}));
    Tensor* seqlens = context->Output(3, TensorShape({1}));
    Tensor* cos_out = context->Output(4, TensorShape({1, head_rows_, 32}));
    Tensor* sin_out = context->Output(5, TensorShape({1, head_rows_, 32}));
    Tensor* sel = context->Output(6, TensorShape({1}));
    ORT_ENFORCE(head_ids != nullptr && head_c != nullptr && slot_ids != nullptr &&
                seqlens != nullptr && cos_out != nullptr && sin_out != nullptr &&
                sel != nullptr);

    int64_t* updated_ctl_ptr = const_cast<int64_t*>(ctl->Data<int64_t>());
    int64_t* updated_ctx_commit_ptr = const_cast<int64_t*>(ctx_commit->Data<int64_t>());
    __nv_bfloat16* head_hidden_ptr = nullptr;
    int hidden_rows = 0;
    int hidden_width = 0;

    if (hidden_states != nullptr && context->OutputCount() == 8) {
      const auto& hshape = hidden_states->Shape();
      hidden_rows = static_cast<int>(hshape[1]);
      hidden_width = static_cast<int>(hshape[2]);
      Tensor* head_hidden = context->Output(7, TensorShape({1, head_rows_, hshape[2]}));
      if (head_hidden != nullptr) {
        head_hidden_ptr = reinterpret_cast<__nv_bfloat16*>(head_hidden->MutableData<BFloat16>());
      }
    } else {
      if (context->OutputCount() > 7) {
        Tensor* updated_ctl = context->Output(7, ctl->Shape());
        if (updated_ctl != nullptr) updated_ctl_ptr = updated_ctl->MutableData<int64_t>();
      }
      if (context->OutputCount() > 8) {
        Tensor* updated_ctx_commit = context->Output(8, ctx_commit->Shape());
        if (updated_ctx_commit != nullptr) updated_ctx_commit_ptr = updated_ctx_commit->MutableData<int64_t>();
      }
      if (hidden_states != nullptr && context->OutputCount() > 9) {
        const auto& hshape = hidden_states->Shape();
        hidden_rows = static_cast<int>(hshape[1]);
        hidden_width = static_cast<int>(hshape[2]);
        Tensor* head_hidden = context->Output(9, TensorShape({1, head_rows_, hshape[2]}));
        if (head_hidden != nullptr) {
          head_hidden_ptr = reinterpret_cast<__nv_bfloat16*>(head_hidden->MutableData<BFloat16>());
        }
      }
    }

    return speculative_fold::LaunchFoldMid2(
        next_tokens->Data<int64_t>(), input_ids->Data<int64_t>(),
        status->Data<int32_t>(), static_cast<int>(status_count),
        ctl->Data<int64_t>(), ctx_commit->Data<int64_t>(),
        ctx_a->Data<int64_t>(), ctx_b->Data<int64_t>(), ptrs->Data<int64_t>(),
        reinterpret_cast<const __nv_bfloat16*>(cos_cache->Data<BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(sin_cache->Data<BFloat16>()),
        head_ids->MutableData<int64_t>(), head_c->MutableData<int32_t>(),
        slot_ids->MutableData<int64_t>(), seqlens->MutableData<int32_t>(),
        reinterpret_cast<__nv_bfloat16*>(cos_out->MutableData<BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(sin_out->MutableData<BFloat16>()),
        sel->MutableData<int64_t>(), updated_ctl_ptr,
        updated_ctx_commit_ptr, cos_cache->Shape()[0], Stream(context),
        static_cast<int>(rows), static_cast<int>(head_rows_),
        hidden_states != nullptr ? reinterpret_cast<const __nv_bfloat16*>(hidden_states->Data<BFloat16>()) : nullptr,
        head_hidden_ptr, hidden_rows, hidden_width);
  }

 private:
  int64_t head_rows_;
};

class FoldTail final : public CudaKernel {
 public:
  explicit FoldTail(const OpKernelInfo& info) : CudaKernel(info) {}

  Status ComputeInternal(OpKernelContext* context) const override {
    const Tensor* next_token = context->Input<Tensor>(0);
    const Tensor* status = context->Input<Tensor>(1);
    const Tensor* ctl = context->Input<Tensor>(2);
    ORT_ENFORCE(next_token != nullptr && status != nullptr && ctl != nullptr);
    ORT_RETURN_IF_NOT(next_token->Shape().Size() >= 1 && HasElementCount(ctl, 8),
                      "FoldTail received an invalid control tensor shape");
    const int64_t status_count = status->Shape().Size();
    ORT_RETURN_IF_NOT(status_count <= std::numeric_limits<int>::max(),
                      "FoldTail status tensor is too large");
    Tensor* done = context->Output(0, TensorShape({1}));
    int64_t* updated_ctl_ptr = const_cast<int64_t*>(ctl->Data<int64_t>());
    if (context->OutputCount() > 1) {
      Tensor* updated_ctl = context->Output(1, ctl->Shape());
      if (updated_ctl != nullptr) updated_ctl_ptr = updated_ctl->MutableData<int64_t>();
    }
    ORT_ENFORCE(done != nullptr);

    return speculative_fold::LaunchFoldTail(
        next_token->Data<int64_t>(), status->Data<int32_t>(),
        static_cast<int>(status_count), ctl->Data<int64_t>(),
        done->MutableData<int32_t>(), updated_ctl_ptr, Stream(context));
  }
};

}  // namespace

ONNX_OPERATOR_KERNEL_EX(
    FoldMid2, kMSDomain, 1, kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .MayInplace(3, 7)
        .MayInplace(4, 8)
        .TypeConstraint("I64", DataTypeImpl::GetTensorType<int64_t>())
        .TypeConstraint("I32", DataTypeImpl::GetTensorType<int32_t>())
        .TypeConstraint("B", DataTypeImpl::GetTensorType<BFloat16>())
        .TypeConstraint("T_OUT7", BuildKernelDefConstraints<int64_t, BFloat16>()),
    FoldMid2);

ONNX_OPERATOR_KERNEL_EX(
    FoldTail, kMSDomain, 1, kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .MayInplace(2, 1)
        .TypeConstraint("I64", DataTypeImpl::GetTensorType<int64_t>())
        .TypeConstraint("I32", DataTypeImpl::GetTensorType<int32_t>()),
    FoldTail);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
