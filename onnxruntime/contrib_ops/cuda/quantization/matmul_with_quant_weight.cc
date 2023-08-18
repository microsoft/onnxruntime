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
#include "core/providers/cuda/cuda_kernel.h"
#include "core/providers/common.h"
#include "core/mlas/inc/mlas_q4.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

template <Typename T>
class MatMulWithQuantWeight final : public CudaKernel {
 public:
  MatMulWithQuantWeight(const OpKernelInfo& info) : OpKernel(info) {
    ORT_ENFORCE(Status::OK() == info.GetAttr<int64_t>("K", &K_));
    ORT_ENFORCE(Status::OK() == info.GetAttr<int64_t>("N", &N_));
    ORT_ENFORCE(Status::OK() == info.GetAttr<int64_t>("block_size", &block_size_));
    ORT_ENFORCE(Status::OK() == info.GetAttr<int64_t>("bits", &nbits_));
    int64_t has_zero_point = 0;
    ORT_ENFORCE(Status::OK() == info.GetAttr<int64_t>("has_zero_point", &has_zero_point));
    has_zero_point_ = has_zero_point != 0;
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  int64_t K_;
  int64_t N_;
  int64_t block_size_;
  int64_t nbits_;
  bool has_zero_point_;
};

template <typename T>
Status MatMulWithQuantWeight<T>::Compute(OpKernelContext* ctx) const {
  concurrency::ThreadPool* thread_pool = ctx->GetOperatorThreadPool();

  const Tensor* a = ctx->Input<Tensor>(0);

  const Tensor* b = ctx->Input<Tensor>(1);
  const auto blob_shape = b->Shape();
  ORT_ENFORCE(blob_shape.NumDimensions() == 1, "Second input of MatMulWithQuantWeight must be a 1D blob!");
  const auto blob_len = blob_shape[0];

  ORT_ENFORCE(nbits_ == 4, "only 4 bits is supported now");
  ORT_ENFORCE(block_size_ == 32, "only block size 32 is supported now");

  const Tensor* bshape_tr = ctx->Input<Tensor>(2);
  TensorShape b_shape({K_, N_});

  MatMulComputeHelper helper;
  ORT_RETURN_IF_ERROR(helper.Compute(a->Shape(), b_shape));
  const size_t max_len = helper.OutputOffsets().size();
  const size_t M = static_cast<size_t>(helper.M());
  const size_t N = static_cast<size_t>(helper.N());
  const size_t K = static_cast<size_t>(helper.K());
  const size_t lda = helper.Lda(false);

  MLAS_BLK_QUANT_TYPE blk_quant_type = has_zero_point_ ? MLAS_BLK_QUANT_TYPE::BlkQ4Zp8 : MLAS_BLK_QUANT_TYPE::BlkQ4Sym;

  auto buf_size = MlasQ4GemmPackBSize(blk_quant_type, N, K);
  ORT_ENFORCE(buf_size > 0, "Operator MatMulWithQuantWeight not yet supported on this hardware platform.");
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
  MlasQ4GemmBatch(blk_quant_type, M, N, K, max_len, gemm_params.data(), thread_pool);

  return Status::OK();
}

ONNX_OPERATOR_KERNEL_EX(
    MatMulWithQuantWeight,
    kMSDomain,
    1,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<MLFloat16>())
        .TypeConstraint("T2", DataTypeImpl::GetTensorType<uint8_t>()),
    MatMulWithQuantWeight);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime

#endif  // ORT_MINIMAL_BUILD
