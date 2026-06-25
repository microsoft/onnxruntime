#pragma once

#ifndef CUFFT_RETURN_IF_ERROR
#define CUFFT_RETURN_IF_ERROR(expr) ORT_RETURN_IF_ERROR(CUFFT_CALL(expr))
#endif

namespace onnxruntime {
namespace cuda {

inline const char* CudaDataTypeToString(cudaDataType_t dt) {
  switch (dt) {
    case CUDA_R_16F:
      return "CUDA_R_16F";
    case CUDA_R_16BF:
      return "CUDA_R_16BF";
    case CUDA_R_32F:
      return "CUDA_R_32F";
#if !defined(DISABLE_FLOAT8_TYPES)
    case CUDA_R_8F_E4M3:
      return "CUDA_R_8F_E4M3";
    case CUDA_R_8F_E5M2:
      return "CUDA_R_8F_E5M2";
#endif
    default:
      return "<unknown>";
  }
}

#ifndef USE_CUDA_MINIMAL
inline const char* cublasGetErrorEnum(cublasStatus_t error) {
  switch (error) {
    case CUBLAS_STATUS_SUCCESS:
      return "CUBLAS_STATUS_SUCCESS";
    case CUBLAS_STATUS_NOT_INITIALIZED:
      return "CUBLAS_STATUS_NOT_INITIALIZED";
    case CUBLAS_STATUS_ALLOC_FAILED:
      return "CUBLAS_STATUS_ALLOC_FAILED";
    case CUBLAS_STATUS_INVALID_VALUE:
      return "CUBLAS_STATUS_INVALID_VALUE";
    case CUBLAS_STATUS_ARCH_MISMATCH:
      return "CUBLAS_STATUS_ARCH_MISMATCH";
    case CUBLAS_STATUS_MAPPING_ERROR:
      return "CUBLAS_STATUS_MAPPING_ERROR";
    case CUBLAS_STATUS_EXECUTION_FAILED:
      return "CUBLAS_STATUS_EXECUTION_FAILED";
    case CUBLAS_STATUS_INTERNAL_ERROR:
      return "CUBLAS_STATUS_INTERNAL_ERROR";
    case CUBLAS_STATUS_NOT_SUPPORTED:
      return "CUBLAS_STATUS_NOT_SUPPORTED";
    case CUBLAS_STATUS_LICENSE_ERROR:
      return "CUBLAS_STATUS_LICENSE_ERROR";
    default:
      return "<unknown>";
  }
}

inline const char* CublasComputeTypeToString(cublasComputeType_t ct) {
  switch (ct) {
    case CUBLAS_COMPUTE_16F:
      return "CUBLAS_COMPUTE_16F";
    case CUBLAS_COMPUTE_32F:
      return "CUBLAS_COMPUTE_32F";
    case CUBLAS_COMPUTE_32F_FAST_16F:
      return "CUBLAS_COMPUTE_32F_FAST_16F";
    case CUBLAS_COMPUTE_32F_FAST_16BF:
      return "CUBLAS_COMPUTE_32F_FAST_16BF";
    case CUBLAS_COMPUTE_32F_FAST_TF32:
      return "CUBLAS_COMPUTE_32F_FAST_TF32";
    case CUBLAS_COMPUTE_64F:
      return "CUBLAS_COMPUTE_64F";
    default:
      return "<unknown>";
  }
}
#endif

inline cudaDataType_t ToCudaDataType(int32_t element_type) {
  switch (element_type) {
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:
      return CUDA_R_32F;
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT16:
      return CUDA_R_16F;
    case ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16:
      return CUDA_R_16BF;
#if !defined(DISABLE_FLOAT8_TYPES)
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT8E4M3FN:
      return CUDA_R_8F_E4M3;
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT8E5M2:
      return CUDA_R_8F_E5M2;
#endif
    default:
      ORT_THROW("Unexpected element_type=", element_type, ".");
  }
}

}  // namespace cuda
}  // namespace onnxruntime
