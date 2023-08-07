// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

//
// This module define MatMulFp32Q4 operator, it is basically
// matmul float32 with right hand side being a 2-D matrix
// pre-packed and block-compacted into int4
//

#ifndef ORT_MINIMAL_BUILD

#include "core/common/safeint.h"
#include "core/framework/op_kernel.h"
#include "core/providers/cpu/math/matmul_helper.h"
#include "core/providers/common.h"
#include "core/mlas/inc/mlas_q4.h"

namespace onnxruntime {
namespace contrib {

class MatMulFpQ4 final : public OpKernel {
 public:
  MatMulFpQ4(const OpKernelInfo& info) : OpKernel(info) {
    const auto t = info.GetAttrOrDefault<int64_t>("blk_quant_type", static_cast<int64_t>(1));
    blk_quant_type_ = t == 0 ? BlkQ4Sym : BlkQ4Zp8;
  }

  Status Compute(OpKernelContext* context) const override;
  MLAS_BLK_QUANT_TYPE blk_quant_type_{BlkQ4Zp8};
};

Status MatMulFpQ4::Compute(OpKernelContext* ctx) const {
  concurrency::ThreadPool* thread_pool = ctx->GetOperatorThreadPool();

  const Tensor* a = ctx->Input<Tensor>(0);

  const Tensor* b = ctx->Input<Tensor>(1);
  const auto blob_shape = b->Shape();
  ORT_ENFORCE(blob_shape.NumDimensions() == 1, "Second input of MatMulFpQ4 must be a 1D blob!");
  const auto blob_len = blob_shape[0];

  const Tensor* bshape_tr = ctx->Input<Tensor>(2);
  TensorShape b_shape(bshape_tr->DataAsSpan<int64_t>());
  ORT_ENFORCE(b_shape.NumDimensions() == 2, "Right hand side of MatMulFpQ4 must be a 2D matrix!");

  MatMulComputeHelper helper;
  ORT_RETURN_IF_ERROR(helper.Compute(a->Shape(), b_shape));
  const size_t max_len = helper.OutputOffsets().size();
  const size_t M = static_cast<size_t>(helper.M());
  const size_t N = static_cast<size_t>(helper.N());
  const size_t K = static_cast<size_t>(helper.K());
  const size_t lda = helper.Lda(false);

  auto buf_size = MlasQ4GemmPackBSize(blk_quant_type_, N, K);
  ORT_ENFORCE(buf_size > 0, "Operator MatMulFpQ4 not yet supported on this hardware platform.");
  ORT_ENFORCE(
      (size_t)blob_len == buf_size,
      "Quantized and packed blob size differ from expected!");

  Tensor* y = ctx->Output(0, helper.OutputShape());

  // Bail out early if the output is going to be empty
  if (y->Shape().Size() == 0)
    return Status::OK();

  const auto* a_data = a->Data<float>();
  const auto* blob_data = b->Data<uint8_t>();
  auto* y_data = y->MutableData<float>();

  std::vector<MLAS_Q4_GEMM_DATA_PARAMS> gemm_params(max_len);
  for (size_t i = 0; i < max_len; i++) {
    gemm_params[i].A = a_data + helper.LeftOffsets()[i];
    gemm_params[i].lda = lda;
    gemm_params[i].B = blob_data;
    gemm_params[i].C = y_data + helper.OutputOffsets()[i];
    gemm_params[i].ldc = N;
  }
  MlasQ4GemmBatch(blk_quant_type_, M, N, K, max_len, gemm_params.data(), thread_pool);

  return Status::OK();
}

ONNX_OPERATOR_KERNEL_EX(
    MatMulFpQ4,
    kMSDomain,
    1,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<float>())
        .TypeConstraint("T2", DataTypeImpl::GetTensorType<uint8_t>())
        .TypeConstraint("T3", DataTypeImpl::GetTensorType<int64_t>()),
    MatMulFpQ4);

}  // namespace contrib
}  // namespace onnxruntime

#endif  // ORT_MINIMAL_BUILD
