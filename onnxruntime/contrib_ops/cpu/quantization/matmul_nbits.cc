// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/safeint.h"
#include "core/framework/op_kernel.h"
#include "core/providers/cpu/math/matmul_helper.h"
#include "core/providers/common.h"
#include "dequantize_blockwise.h"
#include "core/mlas/inc/mlas.h"

#define DQ_INT4_IN_PACKING

extern const float* G_scales_data;
extern const uint8_t* G_zero_points_data;
extern int64_t G_block_size_;
extern int64_t G_nbits_;
extern size_t G_ldfb;
extern size_t G_K;

namespace onnxruntime {
namespace contrib {

class MatMulNBits final : public OpKernel {
 public:
  MatMulNBits(const OpKernelInfo& info) : OpKernel(info) {
    ORT_ENFORCE(Status::OK() == info.GetAttr<int64_t>("K", &K_));
    ORT_ENFORCE(Status::OK() == info.GetAttr<int64_t>("N", &N_));
    ORT_ENFORCE(Status::OK() == info.GetAttr<int64_t>("block_size", &block_size_));
    ORT_ENFORCE(Status::OK() == info.GetAttr<int64_t>("bits", &nbits_));
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  int64_t K_;
  int64_t N_;
  int64_t block_size_;
  int64_t nbits_;
};

Status MatMulNBits::Compute(OpKernelContext* ctx) const {
  concurrency::ThreadPool* thread_pool = ctx->GetOperatorThreadPool();

  const Tensor* a = ctx->Input<Tensor>(0);
  const Tensor* b = ctx->Input<Tensor>(1);
  const Tensor* scales = ctx->Input<Tensor>(2);
  const Tensor* zero_points = ctx->Input<Tensor>(3);

  const auto* a_data = a->Data<float>();
  const uint8_t* b_data = b->Data<uint8_t>();
  const auto* scales_data = scales->Data<float>();
  const auto* zero_points_data = zero_points == nullptr ? nullptr : zero_points->Data<uint8_t>();
#ifdef DQ_INT4_IN_PACKING
  G_scales_data = scales_data;
  G_zero_points_data = zero_points_data;
#endif

  AllocatorPtr allocator;
  auto status = ctx->GetTempSpaceAllocator(&allocator);
  ORT_RETURN_IF_ERROR(status);
#ifdef DQ_INT4_IN_PACKING
  G_block_size_ = this->block_size_;
  G_nbits_ = this->nbits_;
#endif
  auto tmp_b_data_ptr = IAllocator::MakeUniquePtr<float>(allocator, SafeInt<size_t>(K_) * N_);
  DequantizeBlockwise<float>(tmp_b_data_ptr.get(),
                             b_data,
                             scales_data,
                             zero_points_data,
                             static_cast<int32_t>(block_size_),
                             static_cast<int32_t>(nbits_),
                             static_cast<int32_t>(N_),
                             static_cast<int32_t>(K_),
                             thread_pool);

#if 0  // for debug
  auto tm_b_data_ptr_trans = IAllocator::MakeUniquePtr<float>(allocator, SafeInt<size_t>(K_) * N_);
  MlasTranspose(tmp_b_data_ptr.get(), tm_b_data_ptr_trans.get(), N_, K_);
#endif

  TensorShape b_shape({N_, K_});

  MatMulComputeHelper helper;
  ORT_RETURN_IF_ERROR(helper.Compute(a->Shape(), b_shape, false, true));

  Tensor* y = ctx->Output(0, helper.OutputShape());

  // Bail out early if the output is going to be empty
  if (y->Shape().Size() == 0)
    return Status::OK();

  auto* y_data = y->MutableData<float>();

  const size_t max_len = helper.OutputOffsets().size();
  //std::cout << "max_len: " << max_len << std::endl;
  const size_t M = static_cast<size_t>(helper.M());
  const size_t N = static_cast<size_t>(helper.N());
  const size_t K = static_cast<size_t>(helper.K());
  const size_t lda = helper.Lda(false);
#if 0
  const int64_t n_blocks_per_col = (K + block_size_ - 1) / block_size_;
  const int64_t blob_size = block_size_ / 8 * nbits_;
  const size_t ldb = static_cast<size_t>(n_blocks_per_col * blob_size);
  ldfb = helper.Ldb(true);
#else
  const size_t ldb = helper.Ldb(true);
#endif
  G_K = K;

  // TODO: implement with native kernel
  std::vector<MLAS_SGEMM_DATA_PARAMS> data(max_len);
  for (size_t i = 0; i < max_len; i++) {
    data[i].BIsPacked = false;
    data[i].A = a_data + helper.LeftOffsets()[i];
    data[i].lda = lda;
#ifdef DQ_INT4_IN_PACKING
    data[i].B = b_data + helper.RightOffsets()[i];
    data[i].ldb = -1;
#else
    data[i].B = tmp_b_data_ptr.get() + helper.RightOffsets()[i];
    data[i].ldb = ldb;
#endif
    data[i].C = y_data + helper.OutputOffsets()[i];
    data[i].ldc = N;
    data[i].alpha = 1.f;
    data[i].beta = 0.0f;
  }
  const char* env_s = std::getenv("SEQ");
#if 0
  std::cerr << "data[i].B: " << (void*)(data[0].B) << std::endl;
  for (int y = 0; y < N; ++y) {
      for (int x = 0; x < K; ++x) {
          std::cerr << "data[0].B[" << y << "][" << x << "]" << (tmp_b_data_ptr.get() + helper.RightOffsets()[0])[y * ldb + x] << "\n";
      }
  }
#endif

  MlasGemmBatch(CblasNoTrans, CblasTrans,
                M, N, K, data.data(), max_len, thread_pool);

  return Status::OK();
}

ONNX_OPERATOR_KERNEL_EX(
    MatMulNBits,
    kMSDomain,
    1,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<float>())
        .TypeConstraint("T2", DataTypeImpl::GetTensorType<uint8_t>()),
    MatMulNBits);

}  // namespace contrib
}  // namespace onnxruntime
