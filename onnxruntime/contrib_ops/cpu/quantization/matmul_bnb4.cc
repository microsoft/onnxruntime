// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/safeint.h"
#include "core/framework/op_kernel.h"
#include "core/providers/cpu/math/matmul_helper.h"
#include "core/providers/common.h"
#include "dequantize_blockwise_bnb4.h"
#include "core/mlas/inc/mlas.h"

namespace onnxruntime {
namespace contrib {

class MatMulBnb4 final : public OpKernel {
 public:
  MatMulBnb4(const OpKernelInfo& info) : OpKernel(info) {
    ORT_ENFORCE(Status::OK() == info.GetAttr<int64_t>("K", &K_));
    ORT_ENFORCE(Status::OK() == info.GetAttr<int64_t>("N", &N_));
    ORT_ENFORCE(Status::OK() == info.GetAttr<int64_t>("block_size", &block_size_));
    ORT_ENFORCE(Status::OK() == info.GetAttr<int64_t>("quant_type", &quant_type_));
    ORT_ENFORCE(quant_type_ == FP4 || quant_type_ == NF4, "Invalid quant_type, only 0 (FP4) and 1 (NF4) are supported.");
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  int64_t K_;
  int64_t N_;
  int64_t block_size_;
  int64_t quant_type_;
};

Status MatMulBnb4::Compute(OpKernelContext* ctx) const {
  concurrency::ThreadPool* thread_pool = ctx->GetOperatorThreadPool();

  const Tensor* a = ctx->Input<Tensor>(0);
  const Tensor* b_quant = ctx->Input<Tensor>(1);
  const Tensor* absmax = ctx->Input<Tensor>(2);

  const float* a_data = a->Data<float>();
  const uint8_t* b_quant_data = b_quant->Data<uint8_t>();
  const float* absmax_data = absmax->Data<float>();

  AllocatorPtr allocator;
  auto status = ctx->GetTempSpaceAllocator(&allocator);
  ORT_RETURN_IF_ERROR(status);
  auto tmp_b_data_ptr = IAllocator::MakeUniquePtr<float>(allocator, SafeInt<size_t>(K_) * N_);
  DequantizeBlockwiseBnb4<float>(tmp_b_data_ptr.get(),
                                 b_quant_data,
                                 absmax_data,
                                 static_cast<int32_t>(block_size_),
                                 static_cast<int32_t>(quant_type_),
                                 static_cast<int32_t>(N_),
                                 static_cast<int32_t>(K_),
                                 thread_pool);

  constexpr bool transa = false;
  constexpr bool transb = true;
  TensorShape b_shape({N_, K_});
  MatMulComputeHelper helper;
  ORT_RETURN_IF_ERROR(helper.Compute(a->Shape(), b_shape, transa, transb));

  Tensor* y = ctx->Output(0, helper.OutputShape());

  // Bail out early if the output is going to be empty
  if (y->Shape().Size() == 0)
    return Status::OK();

  auto* y_data = y->MutableData<float>();

  const size_t max_len = helper.OutputOffsets().size();
  const size_t M = static_cast<size_t>(helper.M());
  const size_t N = static_cast<size_t>(helper.N());
  const size_t K = static_cast<size_t>(helper.K());
  const size_t lda = helper.Lda(transa);
  const size_t ldb = helper.Ldb(transb);

  // TODO: implement with native kernel
  std::vector<MLAS_SGEMM_DATA_PARAMS> data(max_len);
  for (size_t i = 0; i < max_len; i++) {
    data[i].BIsPacked = false;
    data[i].A = a_data + helper.LeftOffsets()[i];
    data[i].lda = lda;
    data[i].B = tmp_b_data_ptr.get() + helper.RightOffsets()[i];
    data[i].ldb = ldb;
    data[i].C = y_data + helper.OutputOffsets()[i];
    data[i].ldc = N;
    data[i].alpha = 1.f;
    data[i].beta = 0.0f;
  }
  MlasGemmBatch(CblasNoTrans, CblasTrans,
                M, N, K, data.data(), max_len, thread_pool);

  return Status::OK();
}

ONNX_OPERATOR_KERNEL_EX(
    MatMulBnb4,
    kMSDomain,
    1,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<float>())
        .TypeConstraint("T2", DataTypeImpl::GetTensorType<uint8_t>()),
    MatMulBnb4);

}  // namespace contrib
}  // namespace onnxruntime
