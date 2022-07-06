// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/rocm/bert/gemm_fast_gelu.h"
#include "core/providers/cpu/math/matmul_helper.h"
#include "core/providers/rocm/rocm_common.h"
#include "core/providers/rocm/shared_inc/fpgeneric.h"
#include "contrib_ops/rocm/bert/fast_gelu_impl.h"
#include "contrib_ops/rocm/bert/transformer_common.h"

namespace onnxruntime {
namespace contrib {
namespace rocm {

#define REGISTER_KERNEL_TYPED(T)                                     \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                     \
      GemmFastGelu,                                                  \
      kMSDomain,                                                     \
      1,                                                             \
      T,                                                             \
      kRocmExecutionProvider,                                        \
      (*KernelDefBuilder::Create())                                  \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()),    \
      GemmFastGelu<T>);

REGISTER_KERNEL_TYPED(float)
REGISTER_KERNEL_TYPED(MLFloat16)
REGISTER_KERNEL_TYPED(BFloat16)

template <typename T>
GemmFastGelu<T>::GemmFastGelu(const OpKernelInfo& op_kernel_info) : RocmKernel(op_kernel_info) {
  const TransformerOptions* options = TransformerOptions::GetInstance();
  use_half2_ = !options->DisableHalf2();
}

// StridedBatchedGemm can be used for the following GEMM computation
// C[pnm] = A[pnk]*B[km] or C[pnm] = A[pnk]*B[pkm]
static bool CanUseStridedBatchedGemm(const TensorShape& left_shape, const TensorShape& right_shape,
                                     bool transa, bool transb, int64_t& stride_A, int64_t& stride_B, int64_t& stride_C, int64_t& batch_count) {
  size_t left_num_dims = left_shape.NumDimensions();
  size_t right_num_dims = right_shape.NumDimensions();

  if (!(left_num_dims >= 3 && right_num_dims >= 2)) {
    return false;
  }

  size_t left_leading_axis = left_num_dims - 2;
  size_t right_leading_axis = right_num_dims - 2;
  int64_t left_p = left_shape.SizeToDimension(left_num_dims - 2);
  int64_t left_k = transa ? left_shape[left_leading_axis] : left_shape[left_num_dims - 1];

  if (right_num_dims >= 3) {
    int64_t right_p = right_shape.SizeToDimension(right_num_dims - 2);
    if (left_p != right_p) {
      return false;
    }
  }

  int64_t right_k = transb ? right_shape[right_num_dims - 1] : right_shape[right_leading_axis];
  if (left_k != right_k) {
    return false;
  }

  int64_t n = transa ? left_shape[left_num_dims - 1] : left_shape[left_leading_axis];
  int64_t m = transb ? right_shape[right_leading_axis] : right_shape[right_num_dims - 1];
  stride_A = n * left_k;
  stride_B = right_num_dims == 2 ? 0 : right_k * m;
  stride_C = n * m;
  batch_count = left_p;
  return true;
}

template <typename T>
Status GemmFastGelu<T>::ComputeInternal(OpKernelContext* ctx) const {
  typedef typename ToHipType<T>::MappedType HipT;

  const auto* X = ctx->Input<Tensor>(0);
  const auto* W = ctx->Input<Tensor>(1);
  const auto* bias = ctx->Input<Tensor>(2);

  bool transa = false;
  bool transb = false;

  MatMulComputeHelper helper;
  ORT_RETURN_IF_ERROR(helper.Compute(X->Shape(), W->Shape(), transa, transb, false, false, false));

  const int N = static_cast<int>(helper.N());
  const int M = static_cast<int>(helper.M());
  const int K = static_cast<int>(helper.K());

  auto gemm_buffer = GetScratchBuffer<T>(helper.OutputShape().Size());
  Tensor* Y = ctx->Output(0, helper.OutputShape());

  // Bail out early if the output is going to be empty
  if (Y->Shape().Size() == 0)
    return Status::OK();

  const HipT alpha = ToHipType<T>::FromFloat(1.0f);
  const HipT zero = ToHipType<T>::FromFloat(0.0f);

  rocblas_operation transA = transa ? rocblas_operation_transpose : rocblas_operation_none;
  rocblas_operation transB = transb ? rocblas_operation_transpose : rocblas_operation_none;
  const int lda = helper.Lda(transa);
  const int ldb = helper.Ldb(transb);
  const int ldc = helper.Ldc();
  int64_t stride_A, stride_B, stride_C, batch_count;

  if (helper.OutputOffsets().size() == 1) {
    ROCBLAS_RETURN_IF_ERROR(rocblasGemmHelper(
        RocblasHandle(),
        transB,
        transA,
        N,
        M,
        K,
        &alpha,
        reinterpret_cast<const HipT*>(W->template Data<T>()),
        ldb,
        reinterpret_cast<const HipT*>(X->template Data<T>()),
        lda,
        &zero,
        reinterpret_cast<HipT*>(gemm_buffer.get()),
        ldc));
  } else if (CanUseStridedBatchedGemm(X->Shape(), W->Shape(),
                                      transa, transb, stride_A, stride_B, stride_C, batch_count)) {
    ROCBLAS_RETURN_IF_ERROR(rocblasGemmStridedBatchedHelper(RocblasHandle(),
                                                          transB,
                                                          transA,
                                                          N,
                                                          M,
                                                          K,
                                                          &alpha,
                                                          reinterpret_cast<const HipT*>(W->template Data<T>()),
                                                          ldb,
                                                          stride_B,
                                                          reinterpret_cast<const HipT*>(X->template Data<T>()),
                                                          lda,
                                                          stride_A,
                                                          &zero,
                                                          reinterpret_cast<HipT*>(gemm_buffer.get()),
                                                          ldc,
                                                          stride_C,
                                                          static_cast<int>(batch_count)));
  } else {
    // Fill offsets when needed.
    helper.FillOffsets();
    RocmAsyncBuffer<const HipT*> left_arrays(this, helper.LeftOffsets().size());
    RocmAsyncBuffer<const HipT*> right_arrays(this, helper.RightOffsets().size());
    RocmAsyncBuffer<HipT*> output_arrays(this, helper.OutputOffsets().size());
    MatMulComputeHelper::OffsetToArrays(reinterpret_cast<const HipT*>(X->template Data<T>()), helper.LeftOffsets(), left_arrays.CpuSpan());
    MatMulComputeHelper::OffsetToArrays(reinterpret_cast<const HipT*>(W->template Data<T>()), helper.RightOffsets(), right_arrays.CpuSpan());
    MatMulComputeHelper::OffsetToArrays(reinterpret_cast<HipT*>(gemm_buffer.get()), helper.OutputOffsets(), output_arrays.CpuSpan());
    ORT_RETURN_IF_ERROR(left_arrays.CopyToGpu());
    ORT_RETURN_IF_ERROR(right_arrays.CopyToGpu());
    ORT_RETURN_IF_ERROR(output_arrays.CopyToGpu());

    // note that onnxruntime OrtValue is row major, while rocblas is column major,
    // so swap left/right operands
    ROCBLAS_RETURN_IF_ERROR(rocblasGemmBatchedHelper(
        RocblasHandle(),
        transB,
        transA,
        N,
        M,
        K,
        &alpha,
        right_arrays.GpuPtr(),
        ldb,
        left_arrays.GpuPtr(),
        lda,
        &zero,
        output_arrays.GpuPtr(),
        ldc,
        static_cast<int>(helper.OutputOffsets().size())));
  }

  int64_t fast_gelu_input_length = Y->Shape().Size();
  int64_t bias_length = (nullptr == bias) ? 0 : bias->Shape().Size();

  if (!LaunchFastGeluKernel<HipT>(GetDeviceProp(),
                                   Stream(),
                                   static_cast<int>(fast_gelu_input_length),
                                   static_cast<int>(bias_length),
                                   reinterpret_cast<HipT*>(gemm_buffer.get()),
                                   (nullptr != bias) ? reinterpret_cast<const HipT*>(bias->template Data<T>()) : nullptr,
                                   reinterpret_cast<HipT*>(Y->template MutableData<T>()),
                                   use_half2_)) {
    HIP_CALL(hipGetLastError());
    return Status(common::ONNXRUNTIME, common::FAIL);
  }
  return Status::OK();
}

}  // namespace rocm
}  // namespace contrib
}  // namespace onnxruntime
