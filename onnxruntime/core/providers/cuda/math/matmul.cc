// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "matmul_utils.cuh"

#include "core/providers/cuda/math/matmul.h"
#include "core/framework/float8.h"
#include "core/framework/ort_value.h"

#include "core/providers/cuda/shared_inc/fpgeneric.h"
#include "core/providers/cuda/cuda_allocator.h"
#include "core/providers/cuda/tensor/transpose.h"
#include "core/providers/cuda/tunable/math/matmul.h"

#include <memory.h>

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

template <typename U>
float ComputeStandardDeviation(const U* elems, const int32_t size)
{
  if (size == 0 || elems == nullptr) {
    return 0.0f;
  }

  // Calculate the mean
  float sum = 0.0f;
  for (int32_t i = 0; i < size; i ++)
    sum += float(elems[i]);
  float mean = sum / size;

  // Calculate the sum of squared differences from the mean
  float squaredSum = 0.0f;
  for (int32_t i = 0; i < size; i ++) {
    float diff = float(elems[i]) - mean;
    squaredSum += diff * diff;
  }

  // Calculate the variance
  float variance = squaredSum / size;

  // Return the square root of variance as standard deviation
  return std::sqrt(variance);
}

Status ComputeScale(cudaStream_t& stream, const Tensor* tensor, const float std_quant, float& scale)
{
  const int32_t num_coef = tensor->Shape().Size();
  MLFloat16* scale_coef = (MLFloat16*)malloc(num_coef * sizeof(MLFloat16));
  auto status = ComputeStdDevCoefficientsForScale(stream, tensor, num_coef, scale_coef);
  if (! status.IsOK())
    return status;

  float std_coef = ComputeStandardDeviation(scale_coef, num_coef);
  free(scale_coef);

  // If the standard deviation is 0, just use a scale of 1
  scale = fabs(std_coef) < 1e-5 ? 1.0f : std_quant / std_coef;

  return status;
}

void NoOpDeleter(void* [[maybe_unused]] ptr) {
  (void)(ptr);
}

Status TransposeAndConvertToFp8(cudaStream_t& cuda_stream, const Tensor* right_X, IAllocatorUniquePtr<void>& right_X_fp8,
  const cudaDeviceProp& device_prop, const AllocatorPtr& allocator, Stream* stream)
{
  const auto shape_size = right_X->Shape().GetDims().size();
  const int num_elems = right_X->SizeInBytes() / sizeof(MLFloat16);

  cublasHandle_t cublas_handle;
  CUBLAS_RETURN_IF_ERROR(cublasCreate(&cublas_handle));
  cublasSetStream(cublas_handle, cuda_stream);

  right_X_fp8 = IAllocator::MakeUniquePtr<void>(allocator, num_elems * sizeof(Float8E4M3FN), false, stream);

  Status return_status;
  if (shape_size > 1)
  {
    InlinedVector<size_t> perm;
    perm.push_back(1);
    perm.push_back(0);
    for (size_t i = 2; i < shape_size; i++) {
      perm.push_back(i);
    }
    gsl::span<size_t> permutation(perm.data(), shape_size);

    TensorShapeVector dims;
    for (size_t i = 0; i < shape_size; i++) {
      dims.push_back(right_X->Shape()[perm[i]]);
    }

    std::unique_ptr<Tensor> right_X_transposed = Tensor::Create(right_X->DataType(), TensorShape(dims), allocator);

    auto status = cuda::Transpose::DoTranspose(device_prop, cuda_stream, cublas_handle, permutation, *right_X, *right_X_transposed);
    if (!status.IsOK()) {
      return status;
    }
    return_status = MLFloat16ToFloat8E4M3FN(cuda_stream, right_X_transposed.get(), right_X_fp8.get());
  }
  else
  {
    return_status = MLFloat16ToFloat8E4M3FN(cuda_stream, right_X, right_X_fp8.get());
  }

  CUBLAS_RETURN_IF_ERROR(cublasDestroy(cublas_handle));
  return return_status;
}

Status ComputeUsingFp8(OpKernelContext* ctx, MatMulComputeHelper& helper,  cudaStream_t& stream,
  cublasLtHandle_t& cublasLt, const cudaDeviceProp& device_prop, const cublasLtEpilogue_t& epilogue, float alpha,
  IAllocatorUniquePtr<void>& right_X_fp8)
{
  CUDA_RETURN_IF_ERROR(cudaStreamSynchronize(stream));

  const Tensor* left_X = ctx->Input<Tensor>(0);
  const Tensor* right_X = ctx->Input<Tensor>(1);
  Tensor* Y = ctx->Output(0, helper.OutputShape());

  // onnxruntime is row_major, but cublas is col_major, so we need to use right_X as A and left_X as B.
  const TensorShape& shape_A = right_X->Shape();
  const TensorShape& shape_B = left_X->Shape();
  const TensorShape& shape_Y = Y->Shape();

  // https://docs.nvidia.com/cuda/cublas/index.html?highlight=cublasStatus_t#cublasltmatmul
  // cublasLtMatmul only works if A is transposed and B is not.
  cublasOperation_t transA = CUBLAS_OP_T;
  cublasOperation_t transB = CUBLAS_OP_N;

  const int M = static_cast<int>(helper.M());
  const int K = static_cast<int>(helper.K());
  const int N = static_cast<int>(helper.N());

  const int lda = helper.Ldb(transA); // right_X transposed is N,K in row_major, so leading dimension is K in col_major
  const int ldb = helper.Lda(transB); // left_X is M,K in row_major, so leading dimension is K in col_major
  const int ldc = N; // result in col major (cublas) has the N,M shape
  const int ldy = ldc;

  // Create new tensors on the device which hold fp8 (__nv_fp8_e4m3) equivalents of left_X and right_X.
  AllocatorPtr allocator;
  ORT_RETURN_IF_ERROR(ctx->GetTempSpaceAllocator(&allocator));

  size_t C_size = M * N * sizeof(MLFloat16);
  IAllocatorUniquePtr<void> C = IAllocator::MakeUniquePtr<void>(allocator, C_size, false, ctx->GetComputeStream());

  auto left_X_fp8 = IAllocator::MakeUniquePtr<void>(allocator, M * K * sizeof(Float8E4M3FN), false, ctx->GetComputeStream());
  auto status = MLFloat16ToFloat8E4M3FN(stream, left_X, left_X_fp8.get());
  if (! status.IsOK())
    return status;

  // If right_X is nullptr, it means that its value was constant and this computation was already done inside PrePack.
  if (right_X != nullptr) {
    auto status = TransposeAndConvertToFp8(stream, right_X, right_X_fp8, device_prop, allocator, ctx->GetComputeStream());
    if (! status.IsOK()) {
      return status;
    }
  }

  // `cublasltmatmul` computes the following formula: D = alpha*(A*B) + beta*(C).
  // Our matrix multiplication doesn't use a beta * C bias, so we just set the beta to 0 when calling the API.
  // https://docs.nvidia.com/cuda/cublas/index.html?highlight=cublasLtMatmul#cublasltmatmul
  float beta = 0;

  const void* p_input_a = right_X_fp8.get();
  const void* p_input_b = left_X_fp8.get();
  const void* p_input_c = C.get();
  void* p_output_y = Y->MutableDataRaw();

  // Create matrix descriptors. Not setting any extra attributes.
  // ORT is row_major, but cublas is col_major, so we need to use right_X as A and left_X as B.
  int32_t dtype_A = right_X->GetElementType();
  int32_t dtype_B = left_X->GetElementType();
  int32_t dtype_C = Y->GetElementType();
  int32_t dtype_Y = Y->GetElementType();
  cudaDataType_t a_cuda_type = onnxruntime::cuda::ToCudaDataType(dtype_A);
  cudaDataType_t b_cuda_type = onnxruntime::cuda::ToCudaDataType(dtype_B);
  cudaDataType_t c_cuda_type = onnxruntime::cuda::ToCudaDataType(dtype_C);
  cudaDataType_t y_cuda_type = onnxruntime::cuda::ToCudaDataType(dtype_Y);

  // https://docs.nvidia.com/cuda/cublas/index.html?highlight=cublasStatus_t#cublasltmatmul
  // "The compute type must be CUBLAS_COMPUTE_32F."
  // "The scale type must be CUDA_R_32F."
  cudaDataType_t scale_cuda_type = CUDA_R_32F;
  cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F;

  cublasLtMatmulDesc_t operationDesc = nullptr;
  CUBLAS_RETURN_IF_ERROR(cublasLtMatmulDescCreate(&operationDesc, compute_type, scale_cuda_type));
  CUBLAS_RETURN_IF_ERROR(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transA, sizeof(transA)));
  CUBLAS_RETURN_IF_ERROR(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transB, sizeof(transB)));

  cublasLtMatrixLayout_t Adesc = nullptr, Bdesc = nullptr, Cdesc = nullptr, Ydesc = nullptr;
  CUBLAS_RETURN_IF_ERROR(cublasLtMatrixLayoutCreate(&Adesc, a_cuda_type, K, N, lda)); // right_X
  CUBLAS_RETURN_IF_ERROR(cublasLtMatrixLayoutCreate(&Bdesc, b_cuda_type, K, M, ldb)); // left_X
  CUBLAS_RETURN_IF_ERROR(cublasLtMatrixLayoutCreate(&Ydesc, y_cuda_type, N, M, ldc)); // cublas is col major

  int64_t sm_count_ = device_prop.multiProcessorCount;
#if CUDA_VERSION >= 11060
  // CUBLASLT_MATMUL_DESC_SM_COUNT_TARGET exists from https://docs.nvidia.com/cuda/archive/11.6.0/pdf/CUBLAS_Library.pdf
  if (sm_count_ != 0) {
    int math_sm_count = static_cast<int>(sm_count_);
    CUBLAS_RETURN_IF_ERROR(cublasLtMatmulDescSetAttribute(
        operationDesc, CUBLASLT_MATMUL_DESC_SM_COUNT_TARGET, &math_sm_count,
        sizeof(math_sm_count)));
  }
#endif

  // matmul float 8
#if CUDA_VERSION >= 11080
  float* quant_float = (float*)malloc(256 * sizeof(float));
  for (int i = 0; i < 256; i ++) {
    quant_float[i] = i;
  }
  float std_quant = ComputeStandardDeviation(quant_float, 256);
  free(quant_float);

  // Get the weights of the model
  float scale_a, scale_b;
  status = ComputeScale(stream, right_X, std_quant, scale_a);
  if (! status.IsOK())
    return status;
  status = ComputeScale(stream, left_X, std_quant, scale_b);
  if (! status.IsOK())
    return status;
  float scale_y = 1.0f;

  // Create the on device scale values.
  IAllocatorUniquePtr<void> p_scale_a = IAllocator::MakeUniquePtr<void>(allocator, sizeof(float), false, ctx->GetComputeStream());
  IAllocatorUniquePtr<void> p_scale_b = IAllocator::MakeUniquePtr<void>(allocator, sizeof(float), false, ctx->GetComputeStream());
  IAllocatorUniquePtr<void> p_scale_y = IAllocator::MakeUniquePtr<void>(allocator, sizeof(float), false, ctx->GetComputeStream());

  void* sa = p_scale_a.get();
  void* sb = p_scale_b.get();
  void* sy = p_scale_y.get();
  CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(sa, &scale_a, sizeof(float), cudaMemcpyHostToDevice, stream));
  CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(sb, &scale_b, sizeof(float), cudaMemcpyHostToDevice, stream));
  CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(sy, &scale_y, sizeof(float), cudaMemcpyHostToDevice, stream));

  // TODO why does this segfault?
  // If CUBLASLT_MATMUL_DESC_A_SCALE_POINTER is NULL or not set, the corresponding scale will be 1.
  // CUBLAS_RETURN_IF_ERROR(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, sa, sizeof(sa)));
  // CUBLAS_RETURN_IF_ERROR(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, sb, sizeof(sb)));
  // CUBLAS_RETURN_IF_ERROR(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_C_SCALE_POINTER, sy, sizeof(sy)));
  // CUBLAS_RETURN_IF_ERROR(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_D_SCALE_POINTER, sy, sizeof(sy)));
#endif

// float 8
#if !defined(DISABLE_FLOAT8_TYPES)
// For E4M3FN FP8 output, cuBLAS requires C_type to be same as bias_type
CUBLAS_RETURN_IF_ERROR(cublasLtMatrixLayoutCreate(&Cdesc, c_cuda_type, N, M, ldc));
// CUBLAS_RETURN_IF_ERROR(cublasLtMatmulDescSetAttribute(
//     operationDesc, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &c_cuda_type, sizeof(c_cuda_type)));
#else
CUBLAS_RETURN_IF_ERROR(cublasLtMatrixLayoutCreate(&Cdesc, y_cuda_type, N, M, ldc));
#endif

  cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue));

  // https://docs.nvidia.com/cuda/cublas/index.html?highlight=cublasLtMatmulPreferenceAttributes_t#cublasltmatmulpreferenceattributes-t
  // The workspace should be allocated once from OpKernelContext assuming only one cuda function
  // is running at a time (which is not necessarily true with H100).
  size_t workspaceSize = static_cast<size_t>(1 << 25);  // suggested fixed value 32Mb
  cublasLtMatmulPreference_t preference = nullptr;
  CUBLAS_RETURN_IF_ERROR(cublasLtMatmulPreferenceCreate(&preference));
  CUBLAS_RETURN_IF_ERROR(cublasLtMatmulPreferenceSetAttribute(
    preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize, sizeof(workspaceSize)));

  // https://docs.nvidia.com/cuda/cublas/index.html?highlight=cublasLtMatmulAlgoGetHeuristic#cublasltmatmulalgogetheuristic
  cublasLtMatmulHeuristicResult_t heuristicResult = {};
  int returnedResults = 0;
  cublasStatus_t cuda_status = cublasLtMatmulAlgoGetHeuristic(
      cublasLt, operationDesc, Adesc, Bdesc, Cdesc, Ydesc, preference, 1, &heuristicResult, &returnedResults);

  int n_inputs = ctx->InputCount();
  ORT_ENFORCE(
      returnedResults > 0 && cuda_status == CUBLAS_STATUS_SUCCESS,
      " Unable to find any suitable algorithm due to ", onnxruntime::cuda::cublasGetErrorEnum(cuda_status),
      ", returnedResults=", returnedResults, ", alpha=", alpha, ", beta=", beta, ", n_inputs=", n_inputs,
      ", A_type=", onnxruntime::cuda::CudaDataTypeToString(a_cuda_type),
      ", B_type=", onnxruntime::cuda::CudaDataTypeToString(b_cuda_type),
      ", bias_type=", onnxruntime::cuda::CudaDataTypeToString(c_cuda_type),
      ", result_type=", onnxruntime::cuda::CudaDataTypeToString(y_cuda_type),
      ", scale_type=", onnxruntime::cuda::CudaDataTypeToString(scale_cuda_type),
      ", computeType=", onnxruntime::cuda::CublasComputeTypeToString(compute_type),
      ", epilogue=", epilogue, ", smCount=", sm_count_, ", transA=", transA, ", transB=", transB,
      ", fastAccumulationMode=", 1,
      ", shape_A=", shape_A[0], "x", shape_A[1],
      ", shape_B=", shape_B[0], "x", shape_B[1],
      ", shape_Y=", (shape_Y.NumDimensions() > 0 ? shape_Y[0] : 0), "x", (shape_Y.NumDimensions() > 1 ? shape_Y[1] : 0),
      ", M=", M, ", N=", N, ", K=", K, ", lda=", lda, ", ldb=", ldb, ", ldy=", ldy, ", workspaceSize=", workspaceSize,
      ". Check NVIDIA documentation to see what combination is valid: ",
      "https://docs.nvidia.com/cuda/cublas/index.html?highlight=cublasLtMatmulAlgoGetHeuristic#cublasltmatmulalgogetheuristic"
      ". CUDA>=11.8 is required to use float 8 types.");

  void* workspace = nullptr;
  if (workspaceSize > 0) {
    CUDA_RETURN_IF_ERROR(cudaMalloc(reinterpret_cast<void**>(&workspace), workspaceSize));
  }

  // https://docs.nvidia.com/cuda/cublas/index.html?highlight=cublasLtMatmul#cublasltmatmul
  cuda_status = cublasLtMatmul(
      cublasLt,
      operationDesc,
      static_cast<const void*>(&alpha),         /* alpha */
      p_input_a,                                /* A */
      Adesc,
      p_input_b,                                /* B */
      Bdesc,
      static_cast<const void*>(&beta),          /* beta */
      p_input_c,                                /* C */
      Cdesc,
      p_output_y,                               /* Y */
      Ydesc,
      &heuristicResult.algo,                    /* algo */
      workspace,                                /* workspace */
      workspaceSize,
      stream);                                  /* stream */

  ORT_ENFORCE(
      cuda_status == CUBLAS_STATUS_SUCCESS,
      " Unable to run cublasLtMatmul due to ", onnxruntime::cuda::cublasGetErrorEnum(cuda_status),
      ", returnedResults=", returnedResults, ", alpha=", alpha, ", n_inputs=", n_inputs,
      ", A_type=", onnxruntime::cuda::CudaDataTypeToString(a_cuda_type),
      ", B_type=", onnxruntime::cuda::CudaDataTypeToString(b_cuda_type),
      ", bias_type=", onnxruntime::cuda::CudaDataTypeToString(c_cuda_type),
      ", result_type=", onnxruntime::cuda::CudaDataTypeToString(y_cuda_type),
      ", scale_type=", onnxruntime::cuda::CudaDataTypeToString(scale_cuda_type),
      ", computeType=", onnxruntime::cuda::CublasComputeTypeToString(compute_type),
      ", epilogue=", epilogue, ", smCount=", sm_count_, ", transA=", transA, ", transB=", transB,
      ", fastAccumulationMode=", 1,
      ", shape_A=", shape_A[0], "x", shape_A[1],
      ", shape_B=", shape_B[0], "x", shape_B[1],
      ", M=", M, ", N=", N, ", K=", K, ", lda=", lda, ", ldb=", ldb, ", ldy=", ldy, ", workspaceSize=", workspaceSize,
      ". CUDA>=11.8 is required to use float 8 types.");

  if (workspaceSize > 0) {
    CUDA_RETURN_IF_ERROR(cudaFree(workspace));
  }

  CUBLAS_RETURN_IF_ERROR(cublasLtMatmulPreferenceDestroy(preference));
  CUBLAS_RETURN_IF_ERROR(cublasLtMatrixLayoutDestroy(Ydesc));
  CUBLAS_RETURN_IF_ERROR(cublasLtMatrixLayoutDestroy(Cdesc));
  CUBLAS_RETURN_IF_ERROR(cublasLtMatrixLayoutDestroy(Bdesc));
  CUBLAS_RETURN_IF_ERROR(cublasLtMatrixLayoutDestroy(Adesc));
  CUBLAS_RETURN_IF_ERROR(cublasLtMatmulDescDestroy(operationDesc));
  CUBLAS_RETURN_IF_ERROR(cublasLtDestroy(cublasLt));
  return Status::OK();
}

template <typename T>
Status MatMul<T>::PrePack(const Tensor& tensor, int input_idx, AllocatorPtr alloc, bool& is_packed,
                          PrePackedWeights* prepacked_weights) {
  ORT_UNUSED_PARAMETER(prepacked_weights);
  is_packed = false;

  if (input_idx == 1) { // right_X
    const cudaDeviceProp& device_prop = GetDeviceProp();

    // Create an ONNX Runtime Stream
    cudaStream_t default_cuda_stream = DefaultCudaStream();
    OrtDevice ort_device(OrtDevice::GPU, OrtDevice::MemType::DEFAULT, 0);
    onnxruntime::Stream stream(default_cuda_stream, ort_device);

    auto status = TransposeAndConvertToFp8(default_cuda_stream, &tensor, right_X_fp8_, device_prop, alloc, &stream);

    if (! status.IsOK()) {
      return status;
    }
    is_packed = true;
  }

  return Status::OK();
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
  const auto output_size = Y->Shape().Size();
  if (output_size == 0) return Status::OK();

  if (helper.K() == 0) {
    // When we have (M, 0, N) then the inputs are empty, but the output should
    // be filled out with zeros.
    using CudaT = typename ToCudaType<T>::MappedType;
    Fill<CudaT>(Stream(ctx), reinterpret_cast<CudaT*>(Y->MutableData<T>()), CudaT(0.f), narrow<int64_t>(output_size));
    return Status::OK();
  }

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
        device_prop,
        cuda_kernel->UseTF32()));
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
                                                          device_prop,
                                                          cuda_kernel->UseTF32()));

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
  // It requires Ampere or newer GPU, and pointers of matrices shall be aligned (ideal alignment is 16-byte).
  // Assume that start memory of input/output tensor is aligned, we only check offsets of sub-matrix per batch here.
  bool use_tf32 = std::is_same<T, float>::value &&
                  cuda_kernel->UseTF32() &&
                  device_prop.major >= 8 &&
                  helper.IsBatchedGemmAligned();

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
      device_prop,
      use_tf32));
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


template <>
Status MatMul<MLFloat16>::ComputeDefault(OpKernelContext* ctx, MatMulComputeHelper& helper) const {
  if (use_fp8_) {
    cudaStream_t stream = Stream(ctx);
    auto& device_prop = GetDeviceProp();
    cublasLtHandle_t cublasLt = CublasLtHandle();
    return ComputeUsingFp8(ctx, helper, stream, cublasLt, device_prop, epilogue_, alpha_, right_X_fp8_);
  }

  return ComputeDefaultImpl(ctx, helper);
}

template <typename T>
Status MatMul<T>::ComputeDefault(OpKernelContext* ctx, MatMulComputeHelper& helper) const {
  return ComputeDefaultImpl(ctx, helper);
}

template <typename T>
Status MatMul<T>::ComputeDefaultImpl(OpKernelContext* ctx, MatMulComputeHelper& helper) const {
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
        device_prop,
        UseTF32()));
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
                                                          device_prop,
                                                          UseTF32()));
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
  // It requires Ampere or newer GPU, and pointers of matrices shall be aligned (ideal alignment is 16-byte).
  // Assume that start memory of input/output tensor is aligned, we only check offsets of sub-matrix per batch here.
  bool use_tf32 = std::is_same<T, float>::value &&
                  this->UseTF32() &&
                  device_prop.major >= 8 &&
                  helper.IsBatchedGemmAligned();

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
      device_prop,
      use_tf32));

  return Status::OK();
}

}  // namespace cuda
}  // namespace onnxruntime
