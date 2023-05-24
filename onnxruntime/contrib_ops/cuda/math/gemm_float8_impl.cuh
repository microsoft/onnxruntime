// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/cuda_common.h"

// see https://docs.nvidia.com/cuda/cublas/index.html#cublasltmatmul
// D = alpha*(A*B) + beta*(C)

namespace onnxruntime {
namespace contrib {
namespace cuda {

// It must exist somewhere already.
inline cudaDataType ToCudaDataType(int32_t element_type) {
  switch (element_type) {
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:
      return CUDA_R_32F;
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT16:
      return CUDA_R_16F;
    case ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16:
      return CUDA_R_16BF;
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT8E4M3FN:
      return CUDA_R_8F_E4M3;
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT8E5M2:
      return CUDA_R_8F_E5M2;
    default:
      ORT_THROW("Unexpected element_type=", element_type, ".");
  }
}

// It must exist somewhere already.
inline int32_t TypeSize(int32_t element_type) {
  switch (element_type) {
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:
      return 4;
    case ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16:
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT16:
      return 2;
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT8E4M3FN:
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT8E5M2:
      return 1;
    default:
      ORT_THROW("Unexpected element_type=", element_type, ".");
  }
}

struct GemmFloat8_Impl {
  // see https://docs.nvidia.com/cuda/cublas/index.html?highlight=cublasLtMatmulDescAttributes_t#cublasltmatmuldescattributes-t
  bool fast_accumulation_mode_;
  // see https://docs.nvidia.com/cuda/cublas/index.html?highlight=cublasComputeType_t#cublascomputetype-t
  cublasComputeType_t compute_type_;
  cudaDataType_t scale_type_;
  int64_t sm_count_;
  bool trans_A_;
  bool trans_B_;
  float alpha_;
  float beta_;

  void set(int M, int N, int K, int& lda, int& ldb, int& ldd) const;

  onnxruntime::Status CudaCompute(const int32_t* dtypes, cudaStream_t stream, cublasLtHandle_t handle,
                                  const Tensor* A, const Tensor* B, const Tensor* C, Tensor* D,
                                  int M, int N, int K) const;
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
