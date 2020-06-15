// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "transpose_matmul.h"
#include "core/providers/cpu/math/matmul_helper.h"
#include "core/util/math.h"

namespace onnxruntime {
namespace contrib {

ONNX_OPERATOR_KERNEL_EX(
    TransposeMatMul,
    kMSDomain,
    1,
    kCpuExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    TransposeMatMul);

TransposeMatMul::TransposeMatMul(const OpKernelInfo& info)
    : OpKernel{info} {
  ORT_ENFORCE(info.GetAttr("transA", &trans_a_attr_).IsOK());
  ORT_ENFORCE(info.GetAttr("transB", &trans_b_attr_).IsOK());
}

Status TransposeMatMul::Compute(OpKernelContext* context) const {
  concurrency::ThreadPool* thread_pool = context->GetOperatorThreadPool();

  const Tensor* A = context->Input<Tensor>(0);
  const Tensor* B = context->Input<Tensor>(1);

  // match CUDA kernel implementation, ignore transpose for vectors
  const bool trans_a = trans_a_attr_ && A->Shape().NumDimensions() != 1;
  const bool trans_b = trans_b_attr_ && B->Shape().NumDimensions() != 1;

  MatMulComputeHelper helper;
  ORT_RETURN_IF_ERROR(helper.Compute(A->Shape(), B->Shape(), trans_a, trans_b));

  Tensor* Y = context->Output(0, helper.OutputShape());

  const size_t num_offsets = helper.OutputOffsets().size();
  for (size_t i = 0; i < num_offsets; ++i) {
    math::Gemm<float, concurrency::ThreadPool>(
        trans_a ? CblasTrans : CblasNoTrans,
        trans_b ? CblasTrans : CblasNoTrans,
        helper.M(), helper.N(), helper.K(),
        1.0f,
        A->Data<float>() + helper.LeftOffsets()[i],
        B->Data<float>() + helper.RightOffsets()[i],
        0.0f,
        Y->MutableData<float>() + helper.OutputOffsets()[i],
        thread_pool);
  }

  return Status::OK();
}

}  // namespace contrib
}  // namespace onnxruntime
