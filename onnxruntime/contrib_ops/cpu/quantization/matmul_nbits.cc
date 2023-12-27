// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/narrow.h"
#include "core/common/safeint.h"
#include "core/framework/op_kernel.h"
#include "core/mlas/inc/mlas.h"
#include "core/mlas/inc/mlas_qnbit.h"
#include "core/mlas/inc/mlas_q4.h"
#include "core/providers/cpu/math/matmul_helper.h"
#include "core/providers/common.h"

namespace onnxruntime {
namespace contrib {

class MatMulNBits final : public OpKernel {
 public:
  MatMulNBits(const OpKernelInfo& info)
      : OpKernel(info),
        K_{narrow<size_t>(info.GetAttr<int64_t>("K"))},
        N_{narrow<size_t>(info.GetAttr<int64_t>("N"))},
        block_size_{narrow<size_t>(info.GetAttr<int64_t>("block_size"))},
        nbits_{narrow<size_t>(info.GetAttr<int64_t>("bits"))},
        accuracy_level_{info.GetAttr<int64_t>("accuracy_level")} {
    ORT_ENFORCE(nbits_ == 4,
                "Only 4b quantization is supported for MatMulNBits op, additional bits support is planned.");
    is_asym_ = info.GetInputCount() >= 4;
    const Tensor* tensor_B = nullptr;
    const Tensor* tensor_scale = nullptr;
    const Tensor* tensor_zero_point = nullptr;
    bool B_constant = info.TryGetConstantInput(1, &tensor_B);
    bool scale_constant = info.TryGetConstantInput(2, &tensor_scale);
    bool zero_point_constant = info.TryGetConstantInput(3, &tensor_zero_point);
    all_constant_ = B_constant && scale_constant;
    all_constant_ = is_asym_ ? all_constant_ && zero_point_constant : all_constant_;
  }

  Status Compute(OpKernelContext* context) const override;

  Status PrePack(const Tensor& tensor, int input_idx, AllocatorPtr alloc,
                 /*out*/ bool& is_packed,
                 /*out*/ PrePackedWeights* prepacked_weights) override;

  Status UseSharedPrePackedBuffers(std::vector<BufferUniquePtr>& prepacked_buffers, int input_idx,
                                   /*out*/ bool& used_shared_buffers) override;

 private:
  const size_t K_;
  const size_t N_;
  const size_t block_size_;
  const size_t nbits_;
  const int64_t accuracy_level_;
  const bool column_wise_quant_{true};
  IAllocatorUniquePtr<void> packed_b_;
  size_t packed_b_size_{0};
  bool is_asym_{false};
  bool all_constant_{false};
};

Status MatMulNBits::PrePack(const Tensor& tensor, int input_idx, /*out*/ AllocatorPtr alloc,
                            /*out*/ bool& is_packed,
                            /*out*/ PrePackedWeights* prepacked_weights) {
  is_packed = false;
  if (!all_constant_) {
    return Status::OK();
  }
  auto compt_type = static_cast<MLAS_SQNBIT_COMPUTE_TYPE>(accuracy_level_);
  MLAS_THREADPOOL* pool = NULL;
  if (input_idx == 1) {
    packed_b_size_ = MlasNBitsGemmPackBSize(N_, K_, block_size_, static_cast<int>(nbits_), is_asym_, compt_type);
    if (packed_b_size_ == 0) return Status::OK();
    auto qptr = tensor.Data<uint8_t>();
    packed_b_ = IAllocator::MakeUniquePtr<void>(alloc, packed_b_size_, true);
    if (packed_b_ == nullptr) {
      return Status::OK();
    }
    std::memset(packed_b_.get(), 0, packed_b_size_);
    MlasNBitsGemmPackB(packed_b_.get(), qptr, nullptr, nullptr, N_, K_, K_, block_size_, static_cast<int>(nbits_),
                       is_asym_, false, compt_type, pool);
    if (prepacked_weights) {
      prepacked_weights->buffers_.push_back(std::move(packed_b_));
      prepacked_weights->buffer_sizes_.push_back(packed_b_size_);
    }
    is_packed = true;
  }
  if (input_idx == 2 && packed_b_ != nullptr) {
    auto sptr = tensor.Data<float>();
    MlasNBitsGemmPackB(packed_b_.get(), nullptr, sptr, nullptr, N_, K_, K_, block_size_, static_cast<int>(nbits_),
                       is_asym_, !is_asym_, compt_type, pool);
    if (prepacked_weights) {
      prepacked_weights->buffers_.push_back(std::move(packed_b_));
      prepacked_weights->buffer_sizes_.push_back(packed_b_size_);
    }
    is_packed = true;
  }
  if (input_idx == 3 && packed_b_ != nullptr) {
    auto zptr = tensor.Data<uint8_t>();
    MlasNBitsGemmPackB(packed_b_.get(), nullptr, nullptr, zptr, N_, K_, K_, block_size_, static_cast<int>(nbits_),
                       is_asym_, is_asym_, compt_type, pool);
    if (prepacked_weights) {
      prepacked_weights->buffers_.push_back(std::move(packed_b_));
      prepacked_weights->buffer_sizes_.push_back(packed_b_size_);
    }
    is_packed = true;
  }

  return Status::OK();
}

Status MatMulNBits::UseSharedPrePackedBuffers(std::vector<BufferUniquePtr>& prepacked_buffers, int input_idx,
                                              /*out*/ bool& used_shared_buffers) {
  used_shared_buffers = false;
  // Pack three tensors into one buffer
  if (input_idx == 1) {
    used_shared_buffers = true;
    packed_b_ = std::move(prepacked_buffers[0]);
  }
  if (input_idx == 2) {
    used_shared_buffers = true;
    packed_b_ = std::move(prepacked_buffers[0]);
  }
  if (input_idx == 3) {
    used_shared_buffers = true;
    packed_b_ = std::move(prepacked_buffers[0]);
  }
  return Status::OK();
}

Status MatMulNBits::Compute(OpKernelContext* ctx) const {
  concurrency::ThreadPool* thread_pool = ctx->GetOperatorThreadPool();

  const Tensor* a = ctx->Input<Tensor>(0);
  const auto* a_data = a->Data<float>();

  if (packed_b_.get()) {
    TensorShape b_shape({static_cast<int64_t>(N_), static_cast<int64_t>(K_)});

    MatMulComputeHelper helper;
    ORT_RETURN_IF_ERROR(helper.Compute(a->Shape(), b_shape, false, true));

    Tensor* y = ctx->Output(0, helper.OutputShape());

    // Bail out early if the output is going to be empty
    if (y->Shape().Size() == 0) return Status::OK();

    auto* y_data = y->MutableData<float>();

    const size_t max_len = helper.OutputOffsets().size();
    const size_t M = static_cast<size_t>(helper.M());
    const size_t N = static_cast<size_t>(helper.N());
    const size_t K = static_cast<size_t>(helper.K());
    const size_t lda = helper.Lda(false);
    std::vector<MLAS_SQNBITS_GEMM_DATA_PACKED_PARAMS> gemm_params(max_len);
    AllocatorPtr allocator;
    auto status = ctx->GetTempSpaceAllocator(&allocator);
    ORT_RETURN_IF_ERROR(status);
    for (size_t i = 0; i < max_len; i++) {
      gemm_params[i].A = a_data + helper.LeftOffsets()[i];
      gemm_params[i].lda = lda;
      gemm_params[i].B = packed_b_.get();
      gemm_params[i].C = y_data + helper.OutputOffsets()[i];
      gemm_params[i].ldc = N;
    }
    auto ws_size = MlasSQNBitsGemmBatchPackedBWorkspaceSize(M, N, K, max_len, gemm_params.data());
    // workspace for activation process(dynamic quantization and others)
    auto ws_ptr = IAllocator::MakeUniquePtr<int8_t>(allocator, ws_size);
    MlasSQNBitsGemmBatchPackedB(M, N, K, max_len, gemm_params.data(), ws_ptr.get(),
                                thread_pool);
    return Status::OK();
  }

  const Tensor* b = ctx->Input<Tensor>(1);
  const Tensor* scales = ctx->Input<Tensor>(2);
  const Tensor* zero_points = ctx->Input<Tensor>(3);
  const uint8_t* b_data = b->Data<uint8_t>();
  const auto* scales_data = scales->Data<float>();
  const auto* zero_points_data = zero_points == nullptr ? nullptr : zero_points->Data<uint8_t>();

  TensorShape b_shape({static_cast<int64_t>(N_), static_cast<int64_t>(K_)});

  MatMulComputeHelper helper;
  ORT_RETURN_IF_ERROR(helper.Compute(a->Shape(), b_shape, false, true));

  Tensor* y = ctx->Output(0, helper.OutputShape());

  // Bail out early if the output is going to be empty
  if (y->Shape().Size() == 0)
    return Status::OK();

  auto* y_data = y->MutableData<float>();

  const size_t batch_count = helper.OutputOffsets().size();
  const size_t M = static_cast<size_t>(helper.M());
  const size_t N = static_cast<size_t>(helper.N());
  const size_t K = static_cast<size_t>(helper.K());
  const size_t lda = helper.Lda(false);

  const MLAS_SQNBITGEMM_COMPUTE_TYPE compute_type = CompFp32;
  if (MlasIsSQNBitGemmAvailable(M, N, K, nbits_, block_size_, compute_type)) {
    // number of bytes or elements between adjacent matrices
    size_t b_data_matrix_stride_in_bytes, b_scale_matrix_stride, b_zero_point_matrix_stride_in_bytes;
    MlasBlockwiseQuantizedBufferSizes(static_cast<int>(nbits_), static_cast<int>(block_size_), /* columnwise */ true,
                                      static_cast<int>(K), static_cast<int>(N),
                                      b_data_matrix_stride_in_bytes, b_scale_matrix_stride,
                                      &b_zero_point_matrix_stride_in_bytes);

    const size_t b_matrix_size = K * N;

    IAllocatorUniquePtr<std::byte> workspace{};
    if (const size_t workspace_size = MlasSQNBitGemmBatchWorkspaceSize(M, N, K, batch_count,
                                                                       nbits_, block_size_, compute_type);
        workspace_size > 0) {
      AllocatorPtr allocator;
      ORT_RETURN_IF_ERROR(ctx->GetTempSpaceAllocator(&allocator));
      workspace = IAllocator::MakeUniquePtr<std::byte>(allocator, workspace_size);
    }

    InlinedVector<MLAS_SQNBIT_GEMM_DATA_PARAMS> data(batch_count);
    for (size_t i = 0; i < batch_count; ++i) {
      const size_t b_matrix_offset = helper.RightOffsets()[i] / b_matrix_size;

      data[i].A = a_data + helper.LeftOffsets()[i];
      data[i].lda = lda;
      data[i].QuantBData = b_data + b_matrix_offset * b_data_matrix_stride_in_bytes;
      data[i].QuantBScale = scales_data + b_matrix_offset * b_scale_matrix_stride;
      data[i].QuantBZeroPoint = zero_points_data != nullptr
                                    ? zero_points_data + b_matrix_offset * b_zero_point_matrix_stride_in_bytes
                                    : nullptr;
      data[i].C = y_data + helper.OutputOffsets()[i];
      data[i].ldc = N;
    }

    MlasSQNBitGemmBatch(M, N, K, batch_count, nbits_, block_size_, compute_type, data.data(), workspace.get(),
                        thread_pool);

    return Status::OK();
  }

  const size_t ldb = helper.Ldb(true);

  AllocatorPtr allocator;
  ORT_RETURN_IF_ERROR(ctx->GetTempSpaceAllocator(&allocator));
  auto tmp_b_data_ptr = IAllocator::MakeUniquePtr<float>(allocator, SafeInt<size_t>(K_) * N_);
  // dequantize b, only 4b quantization is supported for now
  MlasDequantizeBlockwise<float, 4>(
      tmp_b_data_ptr.get(),               // dequantized output
      b_data,                             // quantized input
      scales_data,                        // quantization scales
      zero_points_data,                   // quantization zero points
      static_cast<int32_t>(block_size_),  // quantization block size
      column_wise_quant_,                 // columnwise quantization or row-wise
      static_cast<int32_t>(K_),           // number of rows in quantized input
      static_cast<int32_t>(N_),           // number of columns in quantized input
      thread_pool);

#if 0  // for debug
  auto tm_b_data_ptr_trans = IAllocator::MakeUniquePtr<float>(allocator, SafeInt<size_t>(K_) * N_);
  MlasTranspose(tmp_b_data_ptr.get(), tm_b_data_ptr_trans.get(), N_, K_);
#endif

  std::vector<MLAS_SGEMM_DATA_PARAMS> data(batch_count);
  for (size_t i = 0; i < batch_count; i++) {
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
                M, N, K, data.data(), batch_count, thread_pool);

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
