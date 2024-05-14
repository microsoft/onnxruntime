// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/rocm/math/matmul_impl.h"

#include "core/providers/rocm/rocm_allocator.h"
#include "core/providers/rocm/rocm_kernel.h"
#include "core/providers/rocm/tunable/gemm.h"

namespace onnxruntime {
namespace rocm {

// StridedBatchedGemm can be used for the following GEMM computation
// C[pnm] = A[pnk]*B[km] or C[pnm] = A[pnk]*B[pkm]
static bool CanUseStridedBatchedGemm(const TensorShape& left_shape,
                                     const TensorShape& right_shape,
                                     bool transa, bool transb,
                                     bool trans_batch_a, bool trans_batch_b,
                                     int64_t& stride_A, int64_t& stride_B,
                                     int64_t& stride_C, int64_t& batch_count) {
  size_t left_num_dims = left_shape.NumDimensions();
  size_t right_num_dims = right_shape.NumDimensions();

  if (!(left_num_dims >= 3 && right_num_dims >= 2)) {
    return false;
  }

  size_t left_leading_axis = trans_batch_a ? 0 : left_num_dims - 2;
  size_t right_leading_axis = trans_batch_b ? 0 : right_num_dims - 2;
  int64_t left_p = left_shape.SizeToDimension(left_num_dims - 2);
  if (trans_batch_a) {
    left_p = left_p * left_shape[left_num_dims - 2] / left_shape[0];
  }
  int64_t left_k =
      transa ? left_shape[left_leading_axis] : left_shape[left_num_dims - 1];

  if (right_num_dims >= 3) {
    int64_t right_p = right_shape.SizeToDimension(right_num_dims - 2);
    if (trans_batch_b) {
      right_p = right_p * right_shape[right_num_dims - 2] / right_shape[0];
    }
    if (left_p != right_p) {
      return false;
    }
  }

  int64_t right_k = transb ? right_shape[right_num_dims - 1]
                           : right_shape[right_leading_axis];
  if (left_k != right_k) {
    return false;
  }

  int64_t n =
      transa ? left_shape[left_num_dims - 1] : left_shape[left_leading_axis];
  int64_t m = transb ? right_shape[right_leading_axis]
                     : right_shape[right_num_dims - 1];
  stride_A = n * left_k / (trans_batch_a ? left_shape[0] : 1);
  stride_B = right_num_dims == 2 ? 0 : right_k * m / (trans_batch_b ? right_shape[0] : 1);
  stride_C = n * m;
  batch_count = left_p;
  return true;
}

template <typename T>
Status MatMulImpl(const RocmKernel* op, MatMulComputeHelper& helper,
                  const T* left_x_data, const T* right_x_data, T* output_y_data,
                  const TensorShape& left_shape, const TensorShape& right_shape,
                  bool transa, bool transb, bool trans_batch_a, bool trans_batch_b,
                  const float alpha, onnxruntime::Stream* stream) {
  typedef typename ToHipType<T>::MappedType HipT;

  using tunable::blas::BlasOp;
  BlasOp transA = transa ? BlasOp::Trans : BlasOp::NonTrans;
  BlasOp transB = transb ? BlasOp::Trans : BlasOp::NonTrans;

  const int lda = helper.Lda(transa);
  const int ldb = helper.Ldb(transb);
  const int ldc = helper.Ldc();
  int64_t stride_A, stride_B, stride_C, batch_count;

  auto rocblas_handle = op->GetRocblasHandle(static_cast<RocmStream*>(stream));

  if (helper.OutputOffsets().size() == 1) {
    return tunable::blas::column_major::Gemm(
        op->GetTuningContext(), stream, rocblas_handle,
        transB, transA,
        helper.N(), helper.M(), helper.K(),
        alpha,
        reinterpret_cast<const HipT*>(right_x_data), ldb,
        reinterpret_cast<const HipT*>(left_x_data), lda,
        /*beta=*/0.0f,
        reinterpret_cast<HipT*>(output_y_data), ldc);
  } else if (CanUseStridedBatchedGemm(left_shape, right_shape,
                                      transa, transb, trans_batch_a, trans_batch_b,
                                      stride_A, stride_B, stride_C, batch_count)) {
    return tunable::blas::column_major::StridedBatchedGemm(
        op->GetTuningContext(), stream, rocblas_handle,
        transB, transA,
        helper.N(), helper.M(), helper.K(),
        alpha,
        reinterpret_cast<const HipT*>(right_x_data), ldb, stride_B,
        reinterpret_cast<const HipT*>(left_x_data), lda, stride_A,
        /*beta=*/0.0f,
        reinterpret_cast<HipT*>(output_y_data), ldc, stride_C,
        batch_count);
  }

  // Fill offsets when needed.
  helper.FillOffsets();
  RocmKernel::RocmAsyncBuffer<const HipT*> left_arrays(op, helper.LeftOffsets().size());
  RocmKernel::RocmAsyncBuffer<const HipT*> right_arrays(op, helper.RightOffsets().size());
  RocmKernel::RocmAsyncBuffer<HipT*> output_arrays(op, helper.OutputOffsets().size());
  MatMulComputeHelper::OffsetToArrays(
      reinterpret_cast<const HipT*>(left_x_data),
      helper.LeftOffsets(), left_arrays.CpuSpan());
  MatMulComputeHelper::OffsetToArrays(
      reinterpret_cast<const HipT*>(right_x_data),
      helper.RightOffsets(), right_arrays.CpuSpan());
  MatMulComputeHelper::OffsetToArrays(
      reinterpret_cast<HipT*>(output_y_data),
      helper.OutputOffsets(), output_arrays.CpuSpan());
  ORT_RETURN_IF_ERROR(left_arrays.CopyToGpu(stream));
  ORT_RETURN_IF_ERROR(right_arrays.CopyToGpu(stream));
  ORT_RETURN_IF_ERROR(output_arrays.CopyToGpu(stream));

  // note that onnxruntime OrtValue is row major, while rocblas is column major,
  // so swap left/right operands
  return tunable::blas::column_major::BatchedGemm(
      op->GetTuningContext(), stream, rocblas_handle,
      transB, transA,
      helper.N(), helper.M(), helper.K(),
      alpha,
      right_arrays.GpuPtr(), ldb,
      left_arrays.GpuPtr(), lda,
      /*beta=*/0.0f,
      output_arrays.GpuPtr(), ldc,
      static_cast<int64_t>(helper.OutputOffsets().size()));
}

#define SPECIALIZED_IMPL(T)                                                                    \
  template Status MatMulImpl<T>(const RocmKernel* op, MatMulComputeHelper& helper,             \
                                const T* left_x_data, const T* right_x_data, T* output_y_data, \
                                const TensorShape& left_shape, const TensorShape& right_shape, \
                                bool transa, bool transb,                                      \
                                bool trans_batch_a, bool trans_batch_b,                        \
                                const float t_alpha, onnxruntime::Stream* stream);

SPECIALIZED_IMPL(float)
SPECIALIZED_IMPL(double)
SPECIALIZED_IMPL(MLFloat16)
SPECIALIZED_IMPL(BFloat16)

}  // namespace rocm
}  // namespace onnxruntime
