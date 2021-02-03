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

template <typename T>
struct MatMul<T>::SparseInfo {
  OpKernel::PrepackParam param;
  explicit SparseInfo(const OpKernel::PrepackParam p) : param(p) {}
};

/// <summary>
/// This class contains helper methods to deal with 2:4 sparse data by means
/// of cuSparseLT library.
/// </summary>

template <typename T>
class Sparse2x4ComputeHelper {
  cusparseLtMatDescriptor_t mat_A_desc_;
  cusparseLtMatDescriptor_t mat_B_desc_;
  cusparseLtMatDescriptor_t mat_C_desc_;
  cusparseLtMatmulDescriptor_t mat_mul_desc_;
  cusparseLtHandle_t alg_selection_;
  onnxruntime::optional<cusparseLtMatmulPlan_t> plan_;
  IAllocatorUniquePtr<T> compressed_buffer_;
  int64_t m, k, n;

 public:
  Sparse2x4ComputeHelper() = default;
  ~Sparse2x4ComputeHelper() {
    if (plan_.has_value()) {
      cusparseLtMatmulPlanDestroy(&*plan_);
    }
  }
  Sparse2x4ComputeHelper(const Sparse2x4ComputeHelper&) = delete;
  Sparse2x4ComputeHelper& operator=(const Sparse2x4ComputeHelper&) = delete;

  /// <summary>
  /// Creates necessary descriptors and copies right tensor data to GPU and compressed it
  /// Prepack() has already verfiied that this data is a valid 2:4 format
  /// </summary>
  /// <param name="helper"></param>
  /// <param name="kernel"></param>
  /// <param name="transa"></param>
  /// <param name="transb"></param>
  /// <param name="right">2:4 initializer data</param>
  /// <returns>status</returns>
  Status Initialize(const CudaKernel* kernel, const MatMulComputeHelper& helper,
                    bool transa, bool transb, const Tensor* right) const {
    cusparseOperation_t transA = transa ? CUSPARSE_OPERATION_TRANSPOSE : CUSPARSE_OPERATION_NON_TRANSPOSE;
    cusparseOperation_t transB = transb ? CUSPARSE_OPERATION_TRANSPOSE : CUSPARSE_OPERATION_NON_TRANSPOSE;

    constexpr size_t data_type_size = sizeof(T);
    constexpr auto cuda_type = ToCudaTypeEnum<T>::type;
    const cusparseLtHandle_t* handle = kernel->CusparseLightHandle();

    m_ = helper.M();
    k_ = helper.K();
    n_ = helper.N();

    // Switch K and M as we are feeding them swapped
    CUSPARSELT_RETURN_IF_ERROR(cusparseLtDenseDescriptorInit(handle, &mat_A_desc_, k_, m_,
                                                             (transa) ? m_ : k_,
                                                             data_type_size, cuda_type,
                                                             CUSPARSE_ORDER_COL));

    const int64_t sparse_size = right->Shape().Size();
    ORT_ENFORCE(sparse_size == n_ * k_, "Sparse initializer shape size does not match computed K*N");
    CUSPARSELT_RETURN_IF_ERROR(cusparseLtStructuredDescriptorInit(handle, &mat_B_desc_, n_, k_,
                                                                  (transb) ? n_ : k_,
                                                                  data_type_size, cuda_type,
                                                                  CUSPARSE_ORDER_COL, CUSPARSELT_SPARSITY_50_PERCENT));

    CUSPARSELT_RETURN_IF_ERROR(cusparseLtDenseDescriptorInit(handle, &mat_C_desc_, n_, m_,
                                                             n_, data_type_size, cuda_type,
                                                             CUSPARSE_ORDER_COL));

    CUSPARSELT_RETURN_IF_ERROR(cusparseLtMatmulDescriptorInit(handle, &mat_mul_desc_,
                                                              transB,
                                                              transA,
                                                              &mat_B_desc_,
                                                              &mat_A_desc_,
                                                              &mat_C_desc_,
                                                              &mat_C_desc_,
                                                              ToCudaTypeEnum<T>::at_least_precision));

    CUSPARSELT_RETURN_IF_ERROR(cusparseLtMatmulAlgSelectionInit(handle, &alg_selection_, &mat_mul_desc_, CUSPARSELT_MATMUL_ALG_DEFAULT));

    int alg_id = 0;  // set algorithm ID
    CUSPARSELT_RETURN_IF_ERROR(cusparseLtMatmulAlgSetAttribute(handle, &alg_selection_,
                                                               CUSPARSELT_MATMUL_ALG_CONFIG_ID,
                                                               &alg_id, sizeof(alg_id)));

    size_t workspace_size;
    CUSPARSELT_RETURN_IF_ERROR(cusparseLtMatmulGetWorkspace(handle, &alg_selection_, &workspace_size));

    cusparseLtMatmulPlan_t plan;
    CUSPARSELT_RETURN_IF_ERROR(cusparseLtMatmulPlanInit(handle, &plan_, &mat_mul_desc_, &alg_selection_, workspace_size));
    plan_ = onnxruntime::make_optional<cusparseLtMatmulPlan_t>(plan);

    size_t compressed_size;  // bytes
    CUSPARSELT_RETURN_IF_ERROR(cusparseLtSpMMACompressedSize(handle, &*plan_, &compressed_size));

    size_t num_compressed_elements = compressed_size / data_type_size;
    if ((num_compressed_elements * data_type_size) < compressed_size) {
      num_compressed_elements++;
    }
    // Copy to GPU for compression
    auto dense_buffer = kernel->GetScratchBuffer<T>(sparse_size);
    cudaMemcpy(dense_buffer.get(), right->DataRaw(), sparse_size * data_type_size, cudaMemcpyHostToDevice);
    compressed_buffer_ = kernel->GetScratchBuffer<T>(num_compressed_elements);
    CUSPARSELT_RETURN_IF_ERROR(cusparseLtSpMMACompress(handle, &*plan_, dense_buffer.get(),
                                                       compressed_buffer_.get(), nullptr /* default stream */));

    return Status::OK();
  }

  /// <summary>
  /// Determine if we have batches or strides and multiply repeatedly if necessary.
  /// </summary>
  /// <param name="left">Left Tensor</param>
  /// <param name="Y">Output</param>
  /// <returns>Status</returns>
  Status Compute(const CudaKernel* kernel, const Tensor* left, const Tensor* right, Tensor* Y,
                 float alpha, bool transa, bool transb) {

    int64_t stride_A, stride_B, stride_C;
    int64_t batch_count;

    // We only support the case when the initializer is 2 - D.
    // Otherwise, we would have to create a batch of compressed initializers
    if (CanUseStridedBatchedGemm(left->Shape(), right->Shape(),
                                 transa, transb, stride_A, stride_B, stride_C, batch_count)) {
      ORT_ENFORCE(strideB == 0, "Expecting initializer to be 2 - D, no batches");
      const T* left_data = left->Data<T>();
      const T* right_data = compressed_buffer_.get();
      // XXX: explore TP options
      for (int64_t batch = 0; batch < batch_count; ++batch) {
      }
    }
  }

  /// <summary>
  /// This method validates constant initializer to be a valid 2:4 sparse data
  /// It creates fake A and C descriptors, Matmul descriptor and calls cusparseLtSpMMAPruneCheck()
  ///to validate the initializer. If the initializer has more than 2 dimensions, it is flattened.
  /// If it has only one dimension, a is appended to its shape.
  ///  See https://github.com/NVIDIA/CUDALibrarySamples/issues/19
  /// </summary>
  static Status PrePack(const CudaKernel* kernel, const Tensor& tensor, const OpKernel::PrepackParam& param, bool transA, bool transB, bool& is_packed) {
    if (!tensor.IsDataType<T>()) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, param.name + " : wrong data type for the constant initializer");
    }

    const auto& right_shape = tensor.Shape();

    // To verify the 2:4 format create fake descriptors for A, B and C.
    int64_t M = 2;
    const auto right_num_dims = right_shape.NumDimensions();
    int64_t K = 0;
    int64_t N = 0;
    if (right_num_dims >= 2) {
      // Flatten the initializer to 2 - D
      K = right_shape.SizeToDimension(right_shape[right_num_dims - 1]);
      N = right_shape[right_num_dims - 1];
    } else {
      K = right_shape[0];
      N = 1;
    }

    TensorShape left_shape({M, K});

    constexpr size_t data_type_size = sizeof(T);
    constexpr auto cuda_type = ToCudaTypeEnum<T>::type;
    const cusparseLtHandle_t* handle = kernel->CusparseLightHandle();

    cusparseLtMatDescriptor_t mat_B_desc;
    CUSPARSELT_RETURN_IF_ERROR(cusparseLtStructuredDescriptorInit(handle, &mat_B_desc, N, K,
                                                                  (transB) ? K : N,
                                                                  data_type_size, cuda_type,
                                                                  CUSPARSE_ORDER_COL, CUSPARSELT_SPARSITY_50_PERCENT));

    // Descriptors A, C and D are fake and are only needed for pruning verification and compression.
    // The should not be needed. https://github.com/NVIDIA/CUDALibrarySamples/issues/19
    cusparseLtMatDescriptor_t mat_A_desc;
    CUSPARSELT_RETURN_IF_ERROR(cusparseLtDenseDescriptorInit(handle, &mat_A_desc, K, M,
                                                             (transA) ? M : K,
                                                             data_type_size, cuda_type,
                                                             CUSPARSE_ORDER_COL));

    cusparseLtMatDescriptor_t mat_C_desc;
    CUSPARSELT_RETURN_IF_ERROR(cusparseLtDenseDescriptorInit(handle, &mat_C_desc, N, M,
                                                             N, data_type_size, cuda_type,
                                                             CUSPARSE_ORDER_COL));

    // Swapping A and B
    cusparseLtMatmulDescriptor_t mat_mul_desc;
    CUSPARSELT_RETURN_IF_ERROR(cusparseLtMatmulDescriptorInit(handle, &mat_mul_desc,
                                                              (transB) ? CUSPARSE_OPERATION_TRANSPOSE : CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                              (transA) ? CUSPARSE_OPERATION_TRANSPOSE : CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                              &mat_B_desc,
                                                              &mat_A_desc,
                                                              &mat_C_desc,
                                                              &mat_C_desc,
                                                              ToCudaTypeEnum<T>::at_least_precision));

    // Initializer tensors are stored on host, copy them for validation
    const auto data_size = right_shape.Size();
    auto validation_buffer = kernel->GetScratchBuffer<T>(data_size);
    CUDA_RETURN_IF_ERROR(cudaMemcpy(validation_buffer.get(), tensor.DataRaw(), data_size * data_type_size, cudaMemcpyHostToDevice));

    int valid = 1;
    CUSPARSELT_RETURN_IF_ERROR(cusparseLtSpMMAPruneCheck(handle, &mat_mul_desc,
                                                         validation_buffer.get(),
                                                         &valid,
                                                         static_cast<cudaStream_t>(0)));

    if (valid == 1) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, param.name + " : 2:4 data format validation failed");
    }

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
  // cuSparseLt only handles 2-D matrices
  if (IsAmpereAvaiable()) {
    if (param.input_idx == 1 && param.Is2x4Format()) {
      ORT_RETURN_IF_ERROR(Sparse2x4ComputeHelper<T>::PrePack(this, tensor, param, trans_A_, trans_B_, is_packed));
      sparse_info_ = onnxruntime::make_unique<SparseInfo>(param);
      // We leave is_packed false as we do not copy the data into the kernel
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

  //if (sparse_info_) {
  //  if (sparse_info_->param.Is2x4Format()) {
  //  }
  //  ORT_ENFORCE(false, "SparseInfo present but format is not supported");
  //}

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
