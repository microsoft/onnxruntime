// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/math/matmul.h"

#include "core/providers/cuda/shared_inc/fpgeneric.h"
#include "core/providers/cuda/cuda_allocator.h"
#include "core/providers/cuda/tunable/math/matmul.h"

namespace onnxruntime {
namespace cuda {

#define REGISTER_KERNEL_TYPED(T)                                  \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                        \
      MatMul,                                                     \
      kOnnxDomain,                                                \
      1, 8,                                                       \
      T,                                                          \
      kCudaExecutionProvider,                                     \
      (*KernelDefBuilder::Create())                               \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      MatMul<T>);                                                 \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                        \
      MatMul,                                                     \
      kOnnxDomain,                                                \
      9, 12,                                                      \
      T,                                                          \
      kCudaExecutionProvider,                                     \
      (*KernelDefBuilder::Create())                               \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      MatMul<T>);                                                 \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      MatMul,                                                     \
      kOnnxDomain,                                                \
      13,                                                         \
      T,                                                          \
      kCudaExecutionProvider,                                     \
      (*KernelDefBuilder::Create())                               \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      MatMul<T>);

REGISTER_KERNEL_TYPED(float)
REGISTER_KERNEL_TYPED(double)
REGISTER_KERNEL_TYPED(MLFloat16)
REGISTER_KERNEL_TYPED(BFloat16)

// StridedBatchedGemm can be used for the following GEMM computation
// C[pnm] = A[pnk]*B[km] or C[pnm] = A[pnk]*B[pkm]
static bool CanUseStridedBatchedGemm(const TensorShape& left_shape, const TensorShape& right_shape,
                                     bool transa, bool transb, bool trans_batch_a, bool trans_batch_b,
                                     int64_t& stride_A, int64_t& stride_B, int64_t& stride_C, int64_t& batch_count) {
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
  int64_t left_k = transa ? left_shape[left_leading_axis] : left_shape[left_num_dims - 1];

  if (right_num_dims >= 3) {
    int64_t right_p = right_shape.SizeToDimension(right_num_dims - 2);
    if (trans_batch_b) {
      right_p = right_p * right_shape[right_num_dims - 2] / right_shape[0];
    }
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
  stride_A = n * left_k / (trans_batch_a ? left_shape[0] : 1);
  stride_B = right_num_dims == 2 ? 0 : right_k * m / (trans_batch_b ? right_shape[0] : 1);
  stride_C = n * m;
  batch_count = left_p;
  return true;
}

template <typename T>
Status MatMul<T>::ComputeInternal(OpKernelContext* ctx) const {
  const Tensor* left_X = ctx->Input<Tensor>(0);
  const Tensor* right_X = ctx->Input<Tensor>(1);

  // Ignore the transpose flag if rank of input being 1.
  // Be noted: numpy.transpose on vector does not change anything.
  bool trans_a = trans_A_;
  bool trans_b = trans_B_;
  if (left_X->Shape().NumDimensions() == 1) {
    trans_a = false;
  }
  if (right_X->Shape().NumDimensions() == 1) {
    trans_b = false;
  }

  MatMulComputeHelper helper;
  ORT_RETURN_IF_ERROR(
      helper.Compute(left_X->Shape(), right_X->Shape(), trans_a, trans_b, trans_batch_a_, trans_batch_b_, false));

  Tensor* Y = ctx->Output(0, helper.OutputShape());
  // Bail out early if the output is going to be empty
  if (Y->Shape().Size() == 0) return Status::OK();

  if (GetTuningContext()->IsTunableOpEnabled()) {
    return tunable::TunableMatMul<T>(alpha_, trans_a, trans_b, trans_batch_a_, trans_batch_b_, helper, this, ctx);
  }

  return ComputeDefault(ctx, helper);
}

template <typename T>
Status FuncMatMul(
    const CudaKernel* cuda_kernel,
    OpKernelContext* ctx,
    const Tensor* A,
    const Tensor* B,
    float alpha,
    bool trans_A,
    bool trans_B,
    bool trans_batch_A,
    bool trans_batch_B,
    Tensor* Y) {
  typedef typename ToCudaType<T>::MappedType CudaT;

  // Ignore the transpose flag if rank of input being 1.
  // Be noted: numpy.transpose on vector does not change anything.
  if (A->Shape().NumDimensions() == 1) {
    trans_A = false;
  }
  if (B->Shape().NumDimensions() == 1) {
    trans_B = false;
  }

  const CudaT cuda_alpha = ToCudaType<T>::FromFloat(alpha);
  const CudaT cuda_zero = ToCudaType<T>::FromFloat(0.0f);

  cublasOperation_t cuda_trans_A = trans_A ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t cuda_trans_B = trans_B ? CUBLAS_OP_T : CUBLAS_OP_N;

  MatMulComputeHelper helper;
  ORT_RETURN_IF_ERROR(
      helper.Compute(A->Shape(), B->Shape(), trans_A, trans_B, trans_batch_A, trans_batch_B, false));
  const int lda = helper.Lda(trans_A);
  const int ldb = helper.Ldb(trans_B);
  const int ldc = helper.Ldc();
  int64_t stride_A, stride_B, stride_C, batch_count;
  auto& device_prop = cuda_kernel->GetDeviceProp();

  if (helper.OutputOffsets().size() == 1) {
    CUBLAS_RETURN_IF_ERROR(cublasGemmHelper(
        cuda_kernel->GetCublasHandle(ctx),
        cuda_trans_B,
        cuda_trans_A,
        static_cast<int>(helper.N()),
        static_cast<int>(helper.M()),
        static_cast<int>(helper.K()),
        &cuda_alpha,
        reinterpret_cast<const CudaT*>(B->Data<T>()),
        ldb,
        reinterpret_cast<const CudaT*>(A->Data<T>()),
        lda,
        &cuda_zero,
        reinterpret_cast<CudaT*>(Y->MutableData<T>()),
        ldc,
        device_prop));
    return Status::OK();
  } else if (CanUseStridedBatchedGemm(A->Shape(), B->Shape(),
                                      trans_A, trans_B, trans_batch_B, trans_batch_B, stride_A, stride_B, stride_C, batch_count)) {
    CUBLAS_RETURN_IF_ERROR(cublasGemmStridedBatchedHelper(cuda_kernel->GetCublasHandle(ctx),
                                                          cuda_trans_B,
                                                          cuda_trans_A,
                                                          static_cast<int>(helper.N()),
                                                          static_cast<int>(helper.M()),
                                                          static_cast<int>(helper.K()),
                                                          &cuda_alpha,
                                                          reinterpret_cast<const CudaT*>(B->Data<T>()),
                                                          ldb,
                                                          stride_B,
                                                          reinterpret_cast<const CudaT*>(A->Data<T>()),
                                                          lda,
                                                          stride_A,
                                                          &cuda_zero,
                                                          reinterpret_cast<CudaT*>(Y->MutableData<T>()),
                                                          ldc,
                                                          stride_C,
                                                          static_cast<int>(batch_count),
                                                          device_prop));

    return Status::OK();
  }

  // Fill offsets when needed.
  helper.FillOffsets();
  CudaKernel::CudaAsyncBuffer<const CudaT*> A_arrays(cuda_kernel, helper.LeftOffsets().size());
  CudaKernel::CudaAsyncBuffer<const CudaT*> B_arrays(cuda_kernel, helper.RightOffsets().size());
  CudaKernel::CudaAsyncBuffer<CudaT*> Y_arrays(cuda_kernel, helper.OutputOffsets().size());
  MatMulComputeHelper::OffsetToArrays(reinterpret_cast<const CudaT*>(A->Data<T>()), helper.LeftOffsets(), A_arrays.CpuSpan());
  MatMulComputeHelper::OffsetToArrays(reinterpret_cast<const CudaT*>(B->Data<T>()), helper.RightOffsets(), B_arrays.CpuSpan());
  MatMulComputeHelper::OffsetToArrays(reinterpret_cast<CudaT*>(Y->MutableData<T>()), helper.OutputOffsets(), Y_arrays.CpuSpan());
  ORT_RETURN_IF_ERROR(A_arrays.CopyToGpu(ctx->GetComputeStream()));
  ORT_RETURN_IF_ERROR(B_arrays.CopyToGpu(ctx->GetComputeStream()));
  ORT_RETURN_IF_ERROR(Y_arrays.CopyToGpu(ctx->GetComputeStream()));

  // TF32 provides a huge performance gain for training and inference while preserving FP32 levels of accuracy.
  // It requires Ampere or newer GPU, and pointers of matrics shall be aligned (ideal alignment is 16-byte).
  // Assume that start memory of input/output tensor is aligned, we only check offsets of sub-matrix per batch here.
  cublasMath_t mode = (std::is_same<T, float>::value && device_prop.major >= 8 && helper.IsBatchedGemmAligned())
                          ? CUBLAS_TF32_TENSOR_OP_MATH
                          : CUBLAS_DEFAULT_MATH;
  CublasMathModeSetter math_mode_setter(device_prop, cuda_kernel->GetCublasHandle(ctx), mode);

  // note that onnxruntime OrtValue is row major, while cublas is column major,
  // so swap left/right operands
  CUBLAS_RETURN_IF_ERROR(cublasGemmBatchedHelper(
      cuda_kernel->GetCublasHandle(ctx),
      cuda_trans_B,
      cuda_trans_A,
      static_cast<int>(helper.N()),
      static_cast<int>(helper.M()),
      static_cast<int>(helper.K()),
      &cuda_alpha,
      B_arrays.GpuPtr(),
      ldb,
      A_arrays.GpuPtr(),
      lda,
      &cuda_zero,
      Y_arrays.GpuPtr(),
      ldc,
      static_cast<int>(helper.OutputOffsets().size()),
      device_prop));
  return Status::OK();
}

template Status FuncMatMul<float>(
    // Use OpKernel and do a pointer cast to unify functional calls with other eps.
    // TODO: remove CudaKernel and OpKernelContext.
    const CudaKernel* cuda_kernel,
    // Do NOT use ctx to access inputs and outputs.
    // Inputs and outputs are passed in as function arguments.
    OpKernelContext* ctx,
    const Tensor* A,
    const Tensor* B,
    float alpha,
    bool trans_A,
    bool trans_B,
    bool trans_batch_A,
    bool trans_batch_B,
    Tensor* Y);

template Status FuncMatMul<MLFloat16>(
    // Use OpKernel and do a pointer cast to unify functional calls with other eps.
    // TODO: remove CudaKernel and OpKernelContext.
    const CudaKernel* cuda_kernel,
    // Do NOT use ctx to access inputs and outputs.
    // Inputs and outputs are passed in as function arguments.
    OpKernelContext* ctx,
    const Tensor* A,
    const Tensor* B,
    float alpha,
    bool trans_A,
    bool trans_B,
    bool trans_batch_A,
    bool trans_batch_B,
    Tensor* Y);

template <typename T>
Status MatMul<T>::ComputeDefault(OpKernelContext* ctx, MatMulComputeHelper& helper) const {
  typedef typename ToCudaType<T>::MappedType CudaT;

  const Tensor* left_X = ctx->Input<Tensor>(0);
  const Tensor* right_X = ctx->Input<Tensor>(1);

  // Ignore the transpose flag if rank of input being 1.
  // Be noted: numpy.transpose on vector does not change anything.
  bool transa = trans_A_;
  bool transb = trans_B_;
  if (left_X->Shape().NumDimensions() == 1) {
    transa = false;
  }
  if (right_X->Shape().NumDimensions() == 1) {
    transb = false;
  }

  Tensor* Y = ctx->Output(0, helper.OutputShape());

  const CudaT alpha = ToCudaType<T>::FromFloat(alpha_);
  const CudaT zero = ToCudaType<T>::FromFloat(0.0f);

  cublasOperation_t transA = transa ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t transB = transb ? CUBLAS_OP_T : CUBLAS_OP_N;
  const int lda = helper.Lda(transa);
  const int ldb = helper.Ldb(transb);
  const int ldc = helper.Ldc();
  int64_t stride_A, stride_B, stride_C, batch_count;
  auto& device_prop = GetDeviceProp();

  if (helper.OutputOffsets().size() == 1) {
    CUBLAS_RETURN_IF_ERROR(cublasGemmHelper(
        GetCublasHandle(ctx),
        transB,
        transA,
        static_cast<int>(helper.N()),
        static_cast<int>(helper.M()),
        static_cast<int>(helper.K()),
        &alpha,
        reinterpret_cast<const CudaT*>(right_X->Data<T>()),
        ldb,
        reinterpret_cast<const CudaT*>(left_X->Data<T>()),
        lda,
        &zero,
        reinterpret_cast<CudaT*>(Y->MutableData<T>()),
        ldc,
        device_prop));
    return Status::OK();
  } else if (CanUseStridedBatchedGemm(left_X->Shape(), right_X->Shape(),
                                      transa, transb, trans_batch_a_, trans_batch_b_, stride_A, stride_B, stride_C, batch_count)) {
    CUBLAS_RETURN_IF_ERROR(cublasGemmStridedBatchedHelper(GetCublasHandle(ctx),
                                                          transB,
                                                          transA,
                                                          static_cast<int>(helper.N()),
                                                          static_cast<int>(helper.M()),
                                                          static_cast<int>(helper.K()),
                                                          &alpha,
                                                          reinterpret_cast<const CudaT*>(right_X->Data<T>()),
                                                          ldb,
                                                          stride_B,
                                                          reinterpret_cast<const CudaT*>(left_X->Data<T>()),
                                                          lda,
                                                          stride_A,
                                                          &zero,
                                                          reinterpret_cast<CudaT*>(Y->MutableData<T>()),
                                                          ldc,
                                                          stride_C,
                                                          static_cast<int>(batch_count),
                                                          device_prop));

    return Status::OK();
  }

  // Fill offsets when needed.
  helper.FillOffsets();
  CudaAsyncBuffer<const CudaT*> left_arrays(this, helper.LeftOffsets().size());
  CudaAsyncBuffer<const CudaT*> right_arrays(this, helper.RightOffsets().size());
  CudaAsyncBuffer<CudaT*> output_arrays(this, helper.OutputOffsets().size());
  MatMulComputeHelper::OffsetToArrays(reinterpret_cast<const CudaT*>(left_X->Data<T>()), helper.LeftOffsets(), left_arrays.CpuSpan());
  MatMulComputeHelper::OffsetToArrays(reinterpret_cast<const CudaT*>(right_X->Data<T>()), helper.RightOffsets(), right_arrays.CpuSpan());
  MatMulComputeHelper::OffsetToArrays(reinterpret_cast<CudaT*>(Y->MutableData<T>()), helper.OutputOffsets(), output_arrays.CpuSpan());
  ORT_RETURN_IF_ERROR(left_arrays.CopyToGpu(ctx->GetComputeStream()));
  ORT_RETURN_IF_ERROR(right_arrays.CopyToGpu(ctx->GetComputeStream()));
  ORT_RETURN_IF_ERROR(output_arrays.CopyToGpu(ctx->GetComputeStream()));

  // TF32 provides a huge performance gain for training and inference while preserving FP32 levels of accuracy.
  // It requires Ampere or newer GPU, and pointers of matrics shall be aligned (ideal alignment is 16-byte).
  // Assume that start memory of input/output tensor is aligned, we only check offsets of sub-matrix per batch here.
  cublasMath_t mode = (std::is_same<T, float>::value && device_prop.major >= 8 && helper.IsBatchedGemmAligned())
                          ? CUBLAS_TF32_TENSOR_OP_MATH
                          : CUBLAS_DEFAULT_MATH;
  CublasMathModeSetter math_mode_setter(device_prop, GetCublasHandle(ctx), mode);

  // note that onnxruntime OrtValue is row major, while cublas is column major,
  // so swap left/right operands
  CUBLAS_RETURN_IF_ERROR(cublasGemmBatchedHelper(
      GetCublasHandle(ctx),
      transB,
      transA,
      static_cast<int>(helper.N()),
      static_cast<int>(helper.M()),
      static_cast<int>(helper.K()),
      &alpha,
      right_arrays.GpuPtr(),
      ldb,
      left_arrays.GpuPtr(),
      lda,
      &zero,
      output_arrays.GpuPtr(),
      ldc,
      static_cast<int>(helper.OutputOffsets().size()),
      device_prop));

  return Status::OK();
}

}  // namespace cuda
}  // namespace onnxruntime
