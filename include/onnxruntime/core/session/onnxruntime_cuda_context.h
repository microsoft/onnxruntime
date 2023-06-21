// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef ORT_CUDA_CTX
#define ORT_CUDA_CTX

#include <cuda_runtime.h>
#include <cublas.h>
//#include <cudnn.h>

namespace Ort {

namespace Custom {

struct OrtCudaContext {
  void* raw_stream = {};

  cudaStream_t cuda_stream = {};
  // cudnnHandle_t cudnn_handle = {};
  cublasHandle_t cublas_handle = {};

  void Init(const OrtKernelContext& kernel_ctx) {
    void* stream = {};
    const auto& ort_api = GetApi();
    void* resource = {};
    OrtStatus* status = nullptr;

    status = ort_api.KernelContext_GetStream(&kernel_ctx, &raw_stream);
    if (status) {
      ORT_CXX_API_THROW("failed to fetch raw stream", OrtErrorCode::ORT_RUNTIME_EXCEPTION);
    }

    status = ort_api.Stream_GetResource(raw_stream, "cuda_stream", &resource);
    if (status) {
      ORT_CXX_API_THROW("failed to fetch cuda stream", OrtErrorCode::ORT_RUNTIME_EXCEPTION);
    }
    cuda_stream = reinterpret_cast<cudaStream_t>(resource);

    //resource = {};
    //status = ort_api.Stream_GetResource(raw_stream, "cudnn_handle", &resource);
    //if (status) {
    //  ORT_CXX_API_THROW("failed to fetch cudnn stream", OrtErrorCode::ORT_RUNTIME_EXCEPTION);
    //}
    //cudnn_handle = reinterpret_cast<cudnnHandle_t>(resource);

    resource = {};
    status = ort_api.Stream_GetResource(raw_stream, "cublas_handle", &resource);
    if (status) {
      ORT_CXX_API_THROW("failed to fetch cublas stream", OrtErrorCode::ORT_RUNTIME_EXCEPTION);
    }
    cublas_handle = reinterpret_cast<cublasHandle_t>(resource);
  }
};

}  // namespace Custom
}  // namespace Ort
#endif