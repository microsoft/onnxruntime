// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#define ORT_ROCM_CTX

#include "rocm_resource.h"
#include "core/providers/custom_op_context.h"
#include <hip/hip_runtime.h>
#include <miopen/miopen.h>
#include <rocblas/rocblas.h>

namespace Ort {

namespace Custom {

struct RocmContext : public CustomOpContext {
  hipStream_t hip_stream = {};
  miopenHandle_t miopen_handle = {};
  rocblas_handle rblas_handle = {};

  void Init(const OrtKernelContext& kernel_ctx) override {
    const auto& ort_api = Ort::GetApi();
    void* resource = {};
    OrtStatus* status = nullptr;

    status = ort_api.KernelContext_GetResource(&kernel_ctx, ORT_ROCM_RESOUCE_VERSION, RocmResource::hip_stream_t, &resource);
    if (status) {
      ORT_CXX_API_THROW("failed to fetch hip stream", OrtErrorCode::ORT_RUNTIME_EXCEPTION);
    }
    hip_stream = reinterpret_cast<hipStream_t>(resource);

    resource = {};
    status = ort_api.KernelContext_GetResource(&kernel_ctx, ORT_ROCM_RESOUCE_VERSION, RocmResource::miopen_handle_t, &resource);
    if (status) {
      ORT_CXX_API_THROW("failed to fetch miopen handle", OrtErrorCode::ORT_RUNTIME_EXCEPTION);
    }
    miopen_handle = reinterpret_cast<miopenHandle_t>(resource);

    resource = {};
    status = ort_api.KernelContext_GetResource(&kernel_ctx, ORT_ROCM_RESOUCE_VERSION, RocmResource::rocblas_handle_t, &resource);
    if (status) {
      ORT_CXX_API_THROW("failed to fetch rocblas handle", OrtErrorCode::ORT_RUNTIME_EXCEPTION);
    }
    rblas_handle = reinterpret_cast<rocblas_handle>(resource);
  }
};

}  // namespace Custom
}  // namespace Ort
