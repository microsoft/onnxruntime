// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/math/matmul.h"
#include "core/providers/cpu/math/matmul_helper.h"
#include "core/providers/cuda/shared_inc/fpgeneric.h"
#include "core/providers/cuda/cuda_allocator.h"

namespace onnxruntime {
namespace cuda {

#define REGISTER_KERNEL_TYPED(T)                                  \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                        \
      MatMul,                                                     \
      kOnnxDomain,                                                \
      1, 8,                                                       \
      T,                                                          \
      kCudaExecutionProvider,                                     \
      KernelDefBuilder()                                          \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      MatMul<T>);                                                 \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                        \
      MatMul,                                                     \
      kOnnxDomain,                                                \
      9, 12,                                                      \
      T,                                                          \
      kCudaExecutionProvider,                                     \
      KernelDefBuilder()                                          \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      MatMul<T>);                                                 \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      MatMul,                                                     \
      kOnnxDomain,                                                \
      13,                                                         \
      T,                                                          \
      kCudaExecutionProvider,                                     \
      KernelDefBuilder()                                          \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      MatMul<T>);

REGISTER_KERNEL_TYPED(float)
REGISTER_KERNEL_TYPED(double)
REGISTER_KERNEL_TYPED(MLFloat16)
#if defined(CUDA_VERSION) && CUDA_VERSION >= 11000
REGISTER_KERNEL_TYPED(BFloat16)
#endif

// StridedBatchedGemm can be used for the following GEMM computation
// C[pnm] = A[pnk]*B[km] or C[pnm] = A[pnk]*B[pkm]
static bool CanUseStridedBatchedGemm(const TensorShape& left_shape, const TensorShape& right_shape,
                                     bool transa, bool transb,
                                     int64_t& stride_A, int64_t& stride_B, int64_t& stride_C, int64_t& batch_count) {
  size_t left_num_dims = left_shape.NumDimensions();
  size_t right_num_dims = right_shape.NumDimensions();

  if (!(left_num_dims >= 3 && right_num_dims >= 2)) {
    return false;
  }

  int64_t left_p = left_shape.SizeToDimension(left_num_dims - 2);
  int64_t left_k = transa ? left_shape[left_num_dims - 2] : left_shape[left_num_dims - 1];

  if (right_num_dims >= 3) {
    int64_t right_p = right_shape.SizeToDimension(right_num_dims - 2);
    if (left_p != right_p) {
      return false;
    }
  }

  int64_t right_k = transb ? right_shape[right_num_dims - 1] : right_shape[right_num_dims - 2];
  if (left_k != right_k) {
    return false;
  }

  int64_t n = transa ? left_shape[left_num_dims - 1] : left_shape[left_num_dims - 2];
  int64_t m = transb ? right_shape[right_num_dims - 2] : right_shape[right_num_dims - 1];
  stride_A = n * left_k;
  stride_B = right_num_dims == 2 ? 0 : right_k * m;
  stride_C = n * m;
  batch_count = left_p;
  return true;
}

#ifdef USE_CUSPARSELT

struct Sparse2x4WeightDescriptor {
  int input_idx_;
  TensorShape shape_;
  int64_t K_;
  int64_t N_;
  cusparseLtMatDescriptor_t mat_desc_;
};

template <typename T>
Status MatMul<T>::PrePack(const Tensor& tensor, const PrepackParam& param, bool& is_packed) {
  is_packed = false;
  // We only pack Matrix B just like CPU version
  // only if it is 2:4 pruned and only if A100 available
  // However, we will feed this to cuSparseLT as the first argument.
  // cuSparseLt only handles 2 -D matrices
  if (IsAmpereAvaiable()) {
    if (param.input_idx == 1 && param.Is2x4Format()) {
      if (!tensor.IsDataType<T>()) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Wrong type for the initializer");
      }

      std::unique_ptr<Sparse2x4WeightDescriptor> mat_B = onnxruntime::make_unique<Sparse2x4WeightDescriptor>();
      mat_B->input_idx_ = param.input_idx;
      const auto& shape = tensor.Shape();
      mat_B->shape_ = shape; // Save the original shape for later computation broadcasting

      const size_t num_dims = shape.NumDimensions();
      int64_t K, N;
      if (num_dims > 2) {
        // We need to flatten it for cuSparseLT.
        K = shape.SizeToDimension(num_dims - 1);
        N = shape[num_dims - 1];
      } else if (num_dims == 1) {
        K = shape[0];
        N = 1;
      }

      constexpr size_t data_type_size = sizeof(T);
      constexpr auto cuda_type = ToCudaTypeEnum<T>::type;
      const cusparseLtHandle_t* handle = CusparseLightHandle();
      // We say it is a column order and feed this as matrix A
      CUSPARSELT_RETURN_IF_ERROR(cusparseLtStructuredDescriptorInit(handle, &mat_B->mat_desc_, K, N,
                                                                    K, data_type_size, cuda_type,
                                                                    CUSPARSE_ORDER_COL, CUSPARSELT_SPARSITY_50_PERCENT));
      mat_B->K_ = K;
      mat_B->N_ = N;

      is_packed = true;
    }
  }
  return Status::OK();
}
#endif

template <typename T>
Status MatMul<T>::ComputeInternal(OpKernelContext* ctx) const {
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

  MatMulComputeHelper helper;
  ORT_RETURN_IF_ERROR(helper.Compute(left_X->Shape(), right_X->Shape(), transa, transb));

  Tensor* Y = ctx->Output(0, helper.OutputShape());

  // Bail out early if the output is going to be empty
  if (Y->Shape().Size() == 0)
    return Status::OK();

  const CudaT alpha = ToCudaType<T>::FromFloat(alpha_);
  const CudaT zero = ToCudaType<T>::FromFloat(0.0f);

  cublasOperation_t transA = transa ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t transB = transb ? CUBLAS_OP_T : CUBLAS_OP_N;
  const int lda = transa ? static_cast<int>(helper.M()) : static_cast<int>(helper.K());
  const int ldb = transb ? static_cast<int>(helper.K()) : static_cast<int>(helper.N());
  const int ldc = static_cast<int>(helper.N());
  int64_t stride_A, stride_B, stride_C, batch_count;
  auto& device_prop = GetDeviceProp();
  if (helper.OutputOffsets().size() == 1) {
    CUBLAS_RETURN_IF_ERROR(cublasGemmHelper(
        Base::CublasHandle(),
        transB,
        transA,
        static_cast<int>(helper.N()),
        static_cast<int>(helper.M()),
        static_cast<int>(helper.K()),
        &alpha,
        reinterpret_cast<const CudaT*>(right_X->template Data<T>()),
        ldb,
        reinterpret_cast<const CudaT*>(left_X->template Data<T>()),
        lda,
        &zero,
        reinterpret_cast<CudaT*>(Y->template MutableData<T>()),
        ldc,
        device_prop));
    return Status::OK();
  } else if (CanUseStridedBatchedGemm(left_X->Shape(), right_X->Shape(),
                                      transa, transb, stride_A, stride_B, stride_C, batch_count)) {
    CUBLAS_RETURN_IF_ERROR(cublasGemmStridedBatchedHelper(Base::CublasHandle(),
                                                          transB,
                                                          transA,
                                                          static_cast<int>(helper.N()),
                                                          static_cast<int>(helper.M()),
                                                          static_cast<int>(helper.K()),
                                                          &alpha,
                                                          reinterpret_cast<const CudaT*>(right_X->template Data<T>()),
                                                          ldb,
                                                          stride_B,
                                                          reinterpret_cast<const CudaT*>(left_X->template Data<T>()),
                                                          lda,
                                                          stride_A,
                                                          &zero,
                                                          reinterpret_cast<CudaT*>(Y->template MutableData<T>()),
                                                          ldc,
                                                          stride_C,
                                                          static_cast<int>(batch_count),
                                                          device_prop));

    return Status::OK();
  }

  CudaAsyncBuffer<const CudaT*> left_arrays(this, helper.LeftOffsets().size());
  CudaAsyncBuffer<const CudaT*> right_arrays(this, helper.RightOffsets().size());
  CudaAsyncBuffer<CudaT*> output_arrays(this, helper.OutputOffsets().size());
  MatMulComputeHelper::OffsetToArrays(reinterpret_cast<const CudaT*>(left_X->template Data<T>()), helper.LeftOffsets(), left_arrays.CpuSpan());
  MatMulComputeHelper::OffsetToArrays(reinterpret_cast<const CudaT*>(right_X->template Data<T>()), helper.RightOffsets(), right_arrays.CpuSpan());
  MatMulComputeHelper::OffsetToArrays(reinterpret_cast<CudaT*>(Y->template MutableData<T>()), helper.OutputOffsets(), output_arrays.CpuSpan());
  ORT_RETURN_IF_ERROR(left_arrays.CopyToGpu());
  ORT_RETURN_IF_ERROR(right_arrays.CopyToGpu());
  ORT_RETURN_IF_ERROR(output_arrays.CopyToGpu());

  // note that onnxruntime OrtValue is row major, while cublas is column major,
  // so swap left/right operands
  CUBLAS_RETURN_IF_ERROR(cublasGemmBatchedHelper(
      Base::CublasHandle(),
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
