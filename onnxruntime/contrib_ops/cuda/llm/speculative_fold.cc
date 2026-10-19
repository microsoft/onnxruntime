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
  explicit FoldMid2(const OpKernelInfo& info) : CudaKernel(info) {}

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
    ORT_ENFORCE(next_tokens != nullptr && input_ids != nullptr && status != nullptr &&
                ctl != nullptr && ctx_commit != nullptr && ctx_a != nullptr &&
                ctx_b != nullptr && ptrs != nullptr && cos_cache != nullptr &&
                sin_cache != nullptr);

    ORT_RETURN_IF_NOT(HasElementCount(next_tokens, 2) &&
                          HasElementCount(input_ids, 2) &&
                          HasElementCount(ctl, 8) &&
                          HasElementCount(ctx_commit, 2) &&
                          HasElementCount(ctx_a, 2) &&
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

    Tensor* head_ids = context->Output(0, TensorShape({1, 2}));
    Tensor* head_c = context->Output(1, TensorShape({1}));
    Tensor* slot_ids = context->Output(2, TensorShape({1, 2}));
    Tensor* seqlens = context->Output(3, TensorShape({1}));
    Tensor* cos_out = context->Output(4, TensorShape({1, 2, 32}));
    Tensor* sin_out = context->Output(5, TensorShape({1, 2, 32}));
    Tensor* sel = context->Output(6, TensorShape({1}));
    ORT_ENFORCE(head_ids != nullptr && head_c != nullptr && slot_ids != nullptr &&
                seqlens != nullptr && cos_out != nullptr && sin_out != nullptr &&
                sel != nullptr);

    return speculative_fold::LaunchFoldMid2(
        next_tokens->Data<int64_t>(), input_ids->Data<int64_t>(),
        status->Data<int32_t>(), static_cast<int>(status_count),
        const_cast<int64_t*>(ctl->Data<int64_t>()),
        const_cast<int64_t*>(ctx_commit->Data<int64_t>()),
        ctx_a->Data<int64_t>(), ctx_b->Data<int64_t>(), ptrs->Data<int64_t>(),
        reinterpret_cast<const __nv_bfloat16*>(cos_cache->Data<BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(sin_cache->Data<BFloat16>()),
        head_ids->MutableData<int64_t>(), head_c->MutableData<int32_t>(),
        slot_ids->MutableData<int64_t>(), seqlens->MutableData<int32_t>(),
        reinterpret_cast<__nv_bfloat16*>(cos_out->MutableData<BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(sin_out->MutableData<BFloat16>()),
        sel->MutableData<int64_t>(), Stream(context));
  }
};

class FoldTail final : public CudaKernel {
 public:
  explicit FoldTail(const OpKernelInfo& info) : CudaKernel(info) {}

  Status ComputeInternal(OpKernelContext* context) const override {
    const Tensor* next_token = context->Input<Tensor>(0);
    const Tensor* status = context->Input<Tensor>(1);
    const Tensor* ctl = context->Input<Tensor>(2);
    ORT_ENFORCE(next_token != nullptr && status != nullptr && ctl != nullptr);
    ORT_RETURN_IF_NOT(HasElementCount(next_token, 1) && HasElementCount(ctl, 8),
                      "FoldTail received an invalid control tensor shape");
    const int64_t status_count = status->Shape().Size();
    ORT_RETURN_IF_NOT(status_count <= std::numeric_limits<int>::max(),
                      "FoldTail status tensor is too large");
    Tensor* done = context->Output(0, TensorShape({1}));
    ORT_ENFORCE(done != nullptr);

    return speculative_fold::LaunchFoldTail(
        next_token->Data<int64_t>(), status->Data<int32_t>(),
        static_cast<int>(status_count), const_cast<int64_t*>(ctl->Data<int64_t>()),
        done->MutableData<int32_t>(), Stream(context));
  }
};

}  // namespace

ONNX_OPERATOR_KERNEL_EX(
    FoldMid2, kMSDomain, 1, kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("I64", DataTypeImpl::GetTensorType<int64_t>())
        .TypeConstraint("I32", DataTypeImpl::GetTensorType<int32_t>())
        .TypeConstraint("B", DataTypeImpl::GetTensorType<BFloat16>()),
    FoldMid2);

ONNX_OPERATOR_KERNEL_EX(
    FoldTail, kMSDomain, 1, kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("I64", DataTypeImpl::GetTensorType<int64_t>())
        .TypeConstraint("I32", DataTypeImpl::GetTensorType<int32_t>()),
    FoldTail);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
