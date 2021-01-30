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

class Sparse2x4WeightData {
  int input_idx_;
  TensorShape right_shape_;  // Save the original shape for later computation broadcasting
  int64_t K_;
  int64_t N_;
  cusparseLtMatDescriptor_t sparse_desc_;

 public:

   Sparse2x4WeightData() = default;
  ~Sparse2x4WeightData() = default;

  template <class T>
  Status PrePack(const CudaKernel* kernel, const Tensor& tensor, const OpKernel::PrepackParam& param, bool transA, bool transB, bool& is_packed) {
    if (!tensor.IsDataType<T>()) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, param.name + " : wrong data type for the constant initializer");
    }

    input_idx_ = param.input_idx;
    right_shape_ = tensor.Shape();

    // To verify the 2:4 format and then to compress, create fake descriptors for A(B) and C.
    int64_t M = 2;
    MatMulComputeHelper helper;
    const auto right_num_dims = right_shape_.NumDimensions();
    TensorShape left_shape({M, right_shape_.SizeToDimension(right_shape_[right_num_dims - 1])});
    ORT_RETURN_IF_ERROR(helper.Compute(left_shape, right_shape_, transA, transB));

    // We are expecting those to match with whatever will be computed at Compute
    M = helper.M();
    K_ = helper.K();
    N_ = helper.N();

    constexpr size_t data_type_size = sizeof(T);
    constexpr auto cuda_type = ToCudaTypeEnum<T>::type;
    const cusparseLtHandle_t* handle = kernel->CusparseLightHandle();
    // We say it is a column order and feed this as matrix A
    CUSPARSELT_RETURN_IF_ERROR(cusparseLtStructuredDescriptorInit(handle, &sparse_desc_, K_, N_,
                                                                  K_, data_type_size, cuda_type,
                                                                  CUSPARSE_ORDER_COL, CUSPARSELT_SPARSITY_50_PERCENT));

    // Descriptors A, C and D are fake and are only needed for purning verification and compression.
    // The should not be needed. https://github.com/NVIDIA/CUDALibrarySamples/issues/19
    cusparseLtMatDescriptor_t mat_A_desc;
    CUSPARSELT_RETURN_IF_ERROR(cusparseLtDenseDescriptorInit(handle, &mat_A_desc, M, K_,
                                                             M, data_type_size, cuda_type,
                                                             CUSPARSE_ORDER_COL));

    cusparseLtMatDescriptor_t mat_C_desc;
    CUSPARSELT_RETURN_IF_ERROR(cusparseLtDenseDescriptorInit(handle, &mat_C_desc, M, N_,
                                                             M, data_type_size, cuda_type,
                                                             CUSPARSE_ORDER_COL));

    // Swapping A and B
    cusparseLtMatmulDescriptor_t mat_mul_desc;
    CUSPARSELT_RETURN_IF_ERROR(cusparseLtMatmulDescriptorInit(handle, &mat_mul_desc,
                                           (transB) ? CUSPARSE_OPERATION_TRANSPOSE : CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           (transA) ? CUSPARSE_OPERATION_TRANSPOSE : CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           &sparse_desc_,
                                           &mat_A_desc,
                                           &mat_C_desc,
                                           &mat_C_desc,
                                           ToCudaTypeEnum<T>::at_least_precision));

    auto valid_buf = kernel->GetScratchBuffer<int>(1);
    CUSPARSELT_RETURN_IF_ERROR(cusparseLtSpMMAPruneCheck(handle, &mat_mul_desc,
                                                         tensor.DataRaw(), 
                                                         valid_buf.get(),
                                                         static_cast<cudaStream_t>(0)));

    int valid = 0;
    CUDA_RETURN_IF_ERROR(cudaMemcpy(&valid, valid_buf.get(), sizeof(int), cudaMemcpyDeviceToHost));
    if (valid == 0) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, param.name + " : 2:4 data format validation failed");
    }

    cusparseLtMatmulAlgSelection_t alg_selection;
    CUSPARSELT_RETURN_IF_ERROR(cusparseLtMatmulAlgSelectionInit(handle, &alg_selection, &mat_mul_desc,
                                       CUSPARSELT_MATMUL_ALG_DEFAULT));

    // Now copy this to GPU and remember the pointer to it
    //size_t compressed_size;
    //cusparseLtSpMMACompressedSize(handle, )

    is_packed = true;
    return Status::OK();
  }
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
      std::unique_ptr<Sparse2x4WeightData> data = onnxruntime::make_unique<Sparse2x4WeightData>();
      ORT_RETURN_IF_ERROR(data->template PrePack<T>(this, tensor, param, trans_A_, trans_B_, is_packed));
      // assign this to a member to save PrePack() generated data
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
