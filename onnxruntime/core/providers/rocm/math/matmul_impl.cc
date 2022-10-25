// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Modifications: Remove cudaDeviceProp in LaunchFastGeluKernel.
// Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
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
                  const float t_alpha, const float t_zero) {
  typedef typename ToHipType<T>::MappedType HipT;

  const HipT alpha = ToHipType<T>::FromFloat(t_alpha);
  const HipT zero = ToHipType<T>::FromFloat(t_zero);

  rocblas_operation transA = transa ? rocblas_operation_transpose : rocblas_operation_none;
  rocblas_operation transB = transb ? rocblas_operation_transpose : rocblas_operation_none;
  const int lda = helper.Lda(transa);
  const int ldb = helper.Ldb(transb);
  const int ldc = helper.Ldc();
  int64_t stride_A, stride_B, stride_C, batch_count;

  if (helper.OutputOffsets().size() == 1) {
    using tunable::blas::BlasOp;
    BlasOp transA = transa ? BlasOp::Trans : BlasOp::NonTrans;
    BlasOp transB = transb ? BlasOp::Trans : BlasOp::NonTrans;
    return tunable::blas::column_major::Gemm(
        op->IsTunableOpEnabled(), op->Stream(),
        op->RocblasHandle(), transB, transA, static_cast<int64_t>(helper.N()),
        static_cast<int64_t>(helper.M()), static_cast<int64_t>(helper.K()), t_alpha,
        reinterpret_cast<const HipT*>(right_x_data), ldb,
        reinterpret_cast<const HipT*>(left_x_data), lda, t_zero,
        reinterpret_cast<HipT*>(output_y_data), ldc);
  } else if (CanUseStridedBatchedGemm(left_shape, right_shape,
                                      transa, transb, trans_batch_a, trans_batch_b,
                                      stride_A, stride_B, stride_C, batch_count)) {
    ROCBLAS_RETURN_IF_ERROR(rocblasGemmStridedBatchedHelper(
        op->RocblasHandle(), transB, transA, static_cast<int>(helper.N()),
        static_cast<int>(helper.M()), static_cast<int>(helper.K()), &alpha,
        reinterpret_cast<const HipT*>(right_x_data), ldb, stride_B,
        reinterpret_cast<const HipT*>(left_x_data), lda, stride_A, &zero,
        reinterpret_cast<HipT*>(output_y_data), ldc, stride_C,
        static_cast<int>(batch_count)));
    return Status::OK();
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
  ORT_RETURN_IF_ERROR(left_arrays.CopyToGpu());
  ORT_RETURN_IF_ERROR(right_arrays.CopyToGpu());
  ORT_RETURN_IF_ERROR(output_arrays.CopyToGpu());

  // note that onnxruntime OrtValue is row major, while rocblas is column major,
  // so swap left/right operands
  ROCBLAS_RETURN_IF_ERROR(rocblasGemmBatchedHelper(
      op->RocblasHandle(), transB, transA, static_cast<int>(helper.N()),
      static_cast<int>(helper.M()), static_cast<int>(helper.K()), &alpha,
      right_arrays.GpuPtr(), ldb,
      left_arrays.GpuPtr(), lda, &zero,
      output_arrays.GpuPtr(), ldc,
      static_cast<int>(helper.OutputOffsets().size())));
  return Status::OK();
}

template Status MatMulImpl<float>(const RocmKernel* op,
                                  MatMulComputeHelper& helper,
                                  const float* left_x_data,
                                  const float* right_x_data,
                                  float* output_y_data,
                                  const TensorShape& left_shape,
                                  const TensorShape& right_shape,
                                  bool transa,
                                  bool transb,
                                  bool trans_batch_a,
                                  bool trans_batch_b,
                                  const float t_alpha,
                                  const float t_zero);
template Status MatMulImpl<double>(const RocmKernel* op,
                                   MatMulComputeHelper& helper,
                                   const double* left_x_data,
                                   const double* right_x_data,
                                   double* output_y_data,
                                   const TensorShape& left_shape,
                                   const TensorShape& right_shape,
                                   bool transa,
                                   bool transb,
                                   bool trans_batch_a,
                                   bool trans_batch_b,
                                   const float t_alpha,
                                   const float t_zero);
template Status MatMulImpl<MLFloat16>(const RocmKernel* op,
                                      MatMulComputeHelper& helper,
                                      const MLFloat16* left_x_data,
                                      const MLFloat16* right_x_data,
                                      MLFloat16* output_y_data,
                                      const TensorShape& left_shape,
                                      const TensorShape& right_shape,
                                      bool transa,
                                      bool transb,
                                      bool trans_batch_a,
                                      bool trans_batch_b,
                                      const float t_alpha,
                                      const float t_zero);
template Status MatMulImpl<BFloat16>(const RocmKernel* op,
                                     MatMulComputeHelper& helper,
                                     const BFloat16* left_x_data,
                                     const BFloat16* right_x_data,
                                     BFloat16* output_y_data,
                                     const TensorShape& left_shape,
                                     const TensorShape& right_shape,
                                     bool transa,
                                     bool transb,
                                     bool trans_batch_a,
                                     bool trans_batch_b,
                                     const float t_alpha,
                                     const float t_zero);

}  // namespace rocm
}  // namespace onnxruntime
