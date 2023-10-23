// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/math/matmul.h"
#include "core/providers/cpu/math/gemm_matmul_common.h"
#include "core/providers/cpu/math/matmul_helper.h"
#include "core/util/math.h"
#include "core/util/math_cpuonly.h"
#include "core/mlas/inc/mlas.h"

namespace onnxruntime {

ONNX_CPU_OPERATOR_VERSIONED_TYPED_KERNEL(
    MatMul,
    1, 8,
    float,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    MatMul<float>);

ONNX_CPU_OPERATOR_VERSIONED_TYPED_KERNEL(
    MatMul,
    1, 8,
    double,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<double>()),
    MatMul<double>);

// opset 9 supports more types
ONNX_CPU_OPERATOR_VERSIONED_TYPED_KERNEL(
    MatMul,
    9,
    12,
    float,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    MatMul<float>);

ONNX_CPU_OPERATOR_VERSIONED_TYPED_KERNEL(
    MatMul,
    9,
    12,
    double,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<double>()),
    MatMul<double>);

ONNX_CPU_OPERATOR_VERSIONED_TYPED_KERNEL(
    MatMul,
    9,
    12,
    int32_t,
    KernelDefBuilder()
        .TypeConstraint("T", BuildKernelDefConstraints<int32_t, uint32_t>()),
    MatMul<int32_t>);

ONNX_CPU_OPERATOR_VERSIONED_TYPED_KERNEL(
    MatMul,
    9,
    12,
    int64_t,
    KernelDefBuilder()
        .TypeConstraint("T", BuildKernelDefConstraints<int64_t, uint64_t>()),
    MatMul<int64_t>);

ONNX_CPU_OPERATOR_TYPED_KERNEL(
    MatMul,
    13,
    float,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    MatMul<float>);

ONNX_CPU_OPERATOR_TYPED_KERNEL(
    MatMul,
    13,
    double,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<double>()),
    MatMul<double>);

ONNX_CPU_OPERATOR_TYPED_KERNEL(
    MatMul,
    13,
    int32_t,
    KernelDefBuilder()
        .TypeConstraint("T", BuildKernelDefConstraints<int32_t, uint32_t>()),
    MatMul<int32_t>);

ONNX_CPU_OPERATOR_TYPED_KERNEL(
    MatMul,
    13,
    int64_t,
    KernelDefBuilder()
        .TypeConstraint("T", BuildKernelDefConstraints<int64_t, uint64_t>()),
    MatMul<int64_t>);

template <typename T>
Status MatMul<T>::Compute(OpKernelContext* ctx) const {
  concurrency::ThreadPool* thread_pool = ctx->GetOperatorThreadPool();

  const auto* a = ctx->Input<Tensor>(0);
  const auto* b = ctx->Input<Tensor>(1);

  MatMulComputeHelper helper;
  ORT_RETURN_IF_ERROR(helper.Compute(a->Shape(), b->Shape()));
  Tensor* y = ctx->Output(0, helper.OutputShape());

  // Bail out early if the output is going to be empty
  if (y->Shape().Size() == 0)
    return Status::OK();

  // Using DataRaw as int32_t/uint32_t and int64_t/uint64_t share a common
  // operator body.
  const auto* a_data = reinterpret_cast<const T*>(a->DataRaw());
  const auto* b_data = reinterpret_cast<const T*>(b->DataRaw());
  auto* y_data = reinterpret_cast<T*>(y->MutableDataRaw());

  // TODO: replace it with GemmBatch for performance, it's OK for now as GemmBatch unrolls as well
  size_t max_len = helper.OutputOffsets().size();
  for (size_t i = 0; i < max_len; i++) {
    math::MatMul<T>(
        static_cast<int>(helper.M()),
        static_cast<int>(helper.N()),
        static_cast<int>(helper.K()),
        a_data + helper.LeftOffsets()[i],
        b_data + helper.RightOffsets()[i],
        y_data + helper.OutputOffsets()[i],
        thread_pool);
  }

  return Status::OK();
}

Status MatMul<float>::PrePack(const Tensor& tensor, int input_idx, /*out*/ AllocatorPtr alloc,
                              /*out*/ bool& is_packed,
                              /*out*/ PrePackedWeights* prepacked_weights) {
  is_packed = false;

  // only pack Matrix B
  if (input_idx == 1) {
    size_t packed_b_size;
    is_packed = GemmPackBFp32(alloc, tensor, trans_b_attr_ != 0, packed_b_, packed_b_size, b_shape_);
    bool share_prepacked_weights = (prepacked_weights != nullptr);
    if (is_packed && share_prepacked_weights) {
      prepacked_weights->buffers_.push_back(std::move(packed_b_));
      prepacked_weights->buffer_sizes_.push_back(packed_b_size);
    }
  }
  return Status::OK();
}

Status MatMul<float>::UseSharedPrePackedBuffers(std::vector<BufferUniquePtr>& prepacked_buffers,
                                                int input_idx,
                                                /*out*/ bool& used_shared_buffers) {
  used_shared_buffers = false;

  if (input_idx == 1) {
    used_shared_buffers = true;
    packed_b_ = std::move(prepacked_buffers[0]);
  }

  return Status::OK();
}

Status MatMul<float>::Compute(OpKernelContext* ctx) const {
  concurrency::ThreadPool* thread_pool = ctx->GetOperatorThreadPool();

  const Tensor* a = ctx->Input<Tensor>(0);
  const Tensor* b = packed_b_ ? nullptr : ctx->Input<Tensor>(1);
  const auto& b_shape = b ? b->Shape() : b_shape_;

  // match CUDA kernel implementation, ignore transpose for vectors
  const bool trans_a = trans_a_attr_ && a->Shape().NumDimensions() != 1;
  const bool trans_b = trans_b_attr_ && b_shape.NumDimensions() != 1;

  MatMulComputeHelper helper;
  ORT_RETURN_IF_ERROR(helper.Compute(a->Shape(), b_shape, trans_a, trans_b, trans_batch_a_, trans_batch_b_));
  Tensor* y = ctx->Output(0, helper.OutputShape());

  // Bail out early if the output is going to be empty
  if (y->Shape().Size() == 0)
    return Status::OK();

  const auto* a_data = a->Data<float>();
  const auto* b_data = b ? b->Data<float>() : nullptr;
  auto* y_data = y->MutableData<float>();

  const size_t max_len = helper.OutputOffsets().size();
  const size_t M = static_cast<size_t>(helper.M());
  const size_t N = static_cast<size_t>(helper.N());
  const size_t K = static_cast<size_t>(helper.K());
  const size_t lda = helper.Lda(trans_a);
  const size_t ldb = helper.Ldb(trans_b);

  std::vector<MLAS_SGEMM_DATA_PARAMS> data(max_len);
  for (size_t i = 0; i < max_len; i++) {
    data[i].BIsPacked = bool(packed_b_);
    data[i].A = a_data + helper.LeftOffsets()[i];
    data[i].lda = lda;
#if 0 // comment for prototype
    data[i].B = data[i].BIsPacked ? (float*)packed_b_.get() : b_data + helper.RightOffsets()[i];
#endif
    data[i].ldb = ldb;
    data[i].C = y_data + helper.OutputOffsets()[i];
    data[i].ldc = N;
    data[i].alpha = alpha_attr_;
    data[i].beta = 0.0f;
  }
  MlasGemmBatch(trans_a ? CblasTrans : CblasNoTrans, trans_b ? CblasTrans : CblasNoTrans,
                M, N, K, data.data(), max_len, thread_pool);

  return Status::OK();
}

}  // namespace onnxruntime
