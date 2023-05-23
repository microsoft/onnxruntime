// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gemm_float8_impl.cuh"
#include "core/providers/cuda/cu_inc/common.cuh"
#include "core/providers/cuda/shared_inc/cuda_utils.h"
#include <algorithm>
#include <cuda_runtime.h>

namespace onnxruntime {
namespace contrib {
namespace cuda {

template <>
cudaDataType ToCudaDataType<float>() { return CUDA_R_32F; }

template <>
cudaDataType ToCudaDataType<MLFloat16>() { return CUDA_R_16F; }

template <>
cudaDataType ToCudaDataType<BFloat16>() { return CUDA_R_16BF; }

template <>
cudaDataType ToCudaDataType<Float8E4M3FN>() { return CUDA_R_8F_E4M3; }

template <>
cudaDataType ToCudaDataType<Float8E5M2>() { return CUDA_R_8F_E5M2; }

void GemmFloat8_Impl::set(int M, int N, int K, int& lda, int& ldb, int& ldd) const {
  if (trans_A_ && !trans_B_) {  // TN
    lda = K;
    ldb = K;
    ldd = M;
  } else if (!trans_A_ && !trans_B_) {  // NN
    lda = M;
    ldb = K;
    ldd = M;
  } else if (!trans_A_ && trans_B_) {  // NT
    lda = M;
    ldb = N;
    ldd = M;
  } else {  // TT
    ORT_THROW("trans_A_ == true && trans_B_ == true not allowed.");
  }
}

template <typename AType, typename BType, typename CType, typename DType, typename BiasType>
void GemmFloat8_Impl::CudaCompute<AType, BType, CType, DType, BiasType>(cudaStream_t stream, cublasLtHandle_t handle,
                                                                        const Tensor* A, const Tensor* B, const Tensor* C,
                                                                        Tensor* D, BiasType* relu_bias,
                                                                        int M, int N, int K) const {
  typedef typename onnxruntime::cuda::ToCudaType<AType>::MappedType CudaAType;
  typedef typename onnxruntime::cuda::ToCudaType<BType>::MappedType CudaBType;
  typedef typename onnxruntime::cuda::ToCudaType<CType>::MappedType CudaCType;
  typedef typename onnxruntime::cuda::ToCudaType<DType>::MappedType CudaDType;
  typedef typename onnxruntime::cuda::ToCudaType<BiasType>::MappedType CudaBiasType;

  int lda, ldb, ldd;
  DType alpha_cast, beta_cast;

  set(M, N, K, lda, ldb, ldd);
  alpha_cast = onnxruntime::cuda::ToCudaType<DType>::FromFloat(alpha_);
  beta_cast = onnxruntime::cuda::ToCudaType<DType>::FromFloat(beta_);

  // broadcast bias if needed and is present
  if (beta_ != 0 && C != nullptr) {
    auto& a_shape = A->Shape();
    auto& b_shape = B->Shape();
    auto& c_shape = C->Shape();
    const CudaCType* b_data = reinterpret_cast<const CudaCType*>(C->Data<CudaCType>());
    if (c_shape.Size() == 1) {
      // if C is (), (1,) or (1, 1), broadcast the scalar
      ORT_THROW("Broadcasting is not implemented in GemmFloatByte.");
    } else if (c_shape.NumDimensions() == 1 || c_shape[0] == 1) {
      // C is (N,) or (1, N), broadcast using Y(N,M) = 1 * C(N,1) x ones(1,M) + 0 * C
      ORT_THROW("Broadcasting is not implemented in GemmFloatByte.");
    } else if (b_shape.NumDimensions() == 2 && b_shape[1] == 1) {
      // B is (M, 1), broadcast using Y(N,M) = 1 * ones(N,1) x B(1,M) + 0 * C
      ORT_THROW("Broadcasting is not implemented in GemmFloatByte.");
    } else {
      // C is (M, N), no broadcast needed.
      /*
      constexpr bool same_type = std::same_type<DType, BiasType>::value;
      if (same_type) {
        CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(out_data, b_data, static_cast<size_t>(M) * N * sizeof(T), cudaMemcpyDeviceToDevice, Stream(ctx)));
      }
      */
    }
  }

  // Gemm, note that CUDA assumes col-major, so Y(N,M) = alpha * op(B) x op(A) + beta * C

  constexpr auto A_type = ToCudaDataType<AType>();
  constexpr auto B_type = ToCudaDataType<BType>();
  constexpr auto C_type = ToCudaDataType<CType>();
  constexpr auto D_type = ToCudaDataType<DType>();
  constexpr auto bias_type = ToCudaDataType<BiasType>();

  cublasLtMatmulDesc_t operationDesc = nullptr;
  cublasLtMatrixLayout_t Adesc = nullptr, Bdesc = nullptr, Cdesc = nullptr, Ddesc = nullptr;
  cublasLtMatmulPreference_t preference = nullptr;
  cublasLtMatmulHeuristicResult_t heuristicResult = {};
  cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_DEFAULT;

  // Create matrix descriptors. Not setting any extra attributes.
  CUBLAS_RETURN_IF_ERROR(cublasLtMatrixLayoutCreate(&Adesc, A_type, trans_A_ ? M : K, trans_A_ ? K : M, lda));
  CUBLAS_RETURN_IF_ERROR(cublasLtMatrixLayoutCreate(&Bdesc, B_type, trans_B_ ? K : N, trans_B_ ? N : K, ldb));
  CUBLAS_RETURN_IF_ERROR(cublasLtMatrixLayoutCreate(&Ddesc, D_type, M, N, ldd));

  // CUDA_R_32F is the scale type for the time being since it is not used.
  cublasLtMatmulDescCreate(&operationDesc, compute_type_, CUDA_R_32F);
  cublasOperation_t transa = trans_A_ ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t transb = trans_B_ ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa));
  cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb));
  const int8_t ifast_accumulation_mode = fast_accumulation_mode_ ? 0 : 1;
  cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_FAST_ACCUM, &ifast_accumulation_mode, sizeof(ifast_accumulation_mode));

  /*
  // TODO add inputs for the scales.
  // No scale for the time being so no need to set.
  CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
  CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
  CUBLASLT_MATMUL_DESC_C_SCALE_POINTER
  CUBLASLT_MATMUL_DESC_D_SCALE_POINTER
  CUBLASLT_MATMUL_DESC_AMAX_D_POINTER
  */

  if (sm_count_ != 0) {
    int math_sm_count = sm_count_;
    cublasLtMatmulDescSetAttribute(
        operationDesc, CUBLASLT_MATMUL_DESC_SM_COUNT_TARGET,
        &math_sm_count, sizeof(math_sm_count));
  }

  CUBLAS_RETURN_IF_ERROR(cublasLtMatrixLayoutCreate(&Cdesc, C_type, M, N, ldd));
  if (relu_bias) {
    CUBLAS_RETURN_IF_ERROR(cublasLtMatmulDescSetAttribute(operationDesc,
                                                          CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE,
                                                          &bias_type, sizeof(bias_type)));
    epilogue = CUBLASLT_EPILOGUE_BIAS;
    CUBLAS_RETURN_IF_ERROR(cublasLtMatmulDescSetAttribute(operationDesc,
                                                          CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                                                          &relu_bias, sizeof(*relu_bias)));
  }

  cublasLtMatmulDescSetAttribute(operationDesc,
                                 CUBLASLT_MATMUL_DESC_EPILOGUE,
                                 &epilogue, sizeof(epilogue));

  cublasLtMatmulPreferenceCreate(&preference);

  // See https://docs.nvidia.com/cuda/cublas/index.html?highlight=cublasLtMatmulPreferenceAttributes_t#cublasltmatmulpreferenceattributes-t
  // The workspace should be allocated once from OpKernelContext assuming
  // only one cuda function is running at a time (which is not necessarily true with H100).
  constexpr size_t type_size = std::max(std::max(sizeof(AType), sizeof(BType)), std::max(std::max(sizeof(CType), sizeof(DType)), sizeof(BiasType)));
  size_t workspaceSize = std::max(K * M, K * N) * type_size;  // suggested fixed value 24Mb
  cublasLtMatmulPreferenceSetAttribute(
      preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
      &workspaceSize, sizeof(workspaceSize));

  int returnedResults = 0;
  cublasLtMatmulAlgoGetHeuristic(handle, operationDesc, Adesc, Bdesc, Cdesc,
                                 Ddesc, preference, 1, &heuristicResult,
                                 &returnedResults);
  ORT_ENFORCE(returnedResults > 0, "Unable to find any suitable algorithm.");
  void* workspace = nullptr;
  CUDA_CALL_THROW(cudaMalloc((void**)&workspace, workspaceSize));
  // void* workspace = cudaMalloc(workspaceSize);
  cublasLtMatmul(handle,
                 operationDesc,
                 static_cast<const void*>(&alpha_cast), /* alpha */
                 A,                                     /* A */
                 Adesc,
                 B, /* B */
                 Bdesc,
                 static_cast<const void*>(&beta_cast), /* beta */
                 C,                                    /* C */
                 Cdesc,
                 D, /* D */
                 Ddesc,
                 &heuristicResult.algo, /* algo */
                 workspace,             /* workspace */
                 workspaceSize,
                 stream); /* stream */
  cudaFree(workspace);

  cublasLtMatmulPreferenceDestroy(preference);
  cublasLtMatrixLayoutDestroy(Ddesc);
  cublasLtMatrixLayoutDestroy(Cdesc);
  cublasLtMatrixLayoutDestroy(Bdesc);
  cublasLtMatrixLayoutDestroy(Adesc);
  cublasLtMatmulDescDestroy(operationDesc);
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
