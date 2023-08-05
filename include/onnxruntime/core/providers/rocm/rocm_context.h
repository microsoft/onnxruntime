// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#define ORT_ROCM_CTX

#include "rocm_resource.h"
#include "core/providers/context.h"

namespace Ort {

namespace Custom {

struct RocmContext : public Context {

  hip_stream_t hip_stream = {};
  miopen_handle_t miopen_handle = {};
  rocblas_handle_t rocblas_handle = {};

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
    rblas_handle = reinterpret_cast<rocblasHandle_t>(resource);
  }
};

}  // namespace Csutom
}  // namespace Ort

