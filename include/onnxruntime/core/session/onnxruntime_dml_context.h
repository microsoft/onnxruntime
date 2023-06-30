// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef ORT_DML_CTX
#define ORT_DML_CTX

#include "onnxruntime_ep_resource.h"
#include <DirectML.h>
#include <d3d12.h>

namespace Ort {

namespace Custom {

struct OrtDmlContext {
  IDMLDevice* m_dmlDevice = {};
  ID3D12Device* m_d3d12Device = {};

  void Init(const OrtKernelContext& kernel_ctx) {
    const auto& ort_api = GetApi();
    void* resource = {};
    OrtStatus* status = nullptr;

    status = ort_api.KernelContext_GetResource(&kernel_ctx, ORT_DML_RESOUCE_VERSION, DmlResource::dml_device_t, &resource);
    if (status) {
      ORT_CXX_API_THROW("failed to fetch dml device", OrtErrorCode::ORT_RUNTIME_EXCEPTION);
    }
    m_dmlDevice = reinterpret_cast<IDMLDevice*>(resource);

    resource = {};
    status = ort_api.KernelContext_GetResource(&kernel_ctx, ORT_DML_RESOUCE_VERSION, DmlResource::d3d12_device_t, &resource);
    if (status) {
      ORT_CXX_API_THROW("failed to fetch dml d3d12 device", OrtErrorCode::ORT_RUNTIME_EXCEPTION);
    }
    m_d3d12Device = reinterpret_cast<ID3D12Device*>(resource);
  }
};

}  // namespace Custom
}  // namespace Ort
#endif