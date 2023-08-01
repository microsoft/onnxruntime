// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#define ORT_DML_CTX

#include "dml_resource.h"
#include "core/providers/context.h"
#include <DirectML.h>
#include <d3d12.h>

namespace Ort {

namespace Custom {

struct DmlContext : public Context {
  IDMLDevice* dml_device = {};
  ID3D12Device* d3d12_device = {};
  ID3D12GraphicsCommandList* cmd_list = {};
  IDMLCommandRecorder* cmd_recorder = {};

  void Init(const OrtKernelContext& kernel_ctx) override {
    const auto& ort_api = Ort::GetApi();
    void* resource = {};
    OrtStatus* status = nullptr;

    status = ort_api.KernelContext_GetResource(&kernel_ctx, ORT_DML_RESOUCE_VERSION, DmlResource::dml_device_t, &resource);
    if (status) {
      ORT_CXX_API_THROW("failed to fetch dml device", OrtErrorCode::ORT_RUNTIME_EXCEPTION);
    }
    dml_device = reinterpret_cast<IDMLDevice*>(resource);

    resource = {};
    status = ort_api.KernelContext_GetResource(&kernel_ctx, ORT_DML_RESOUCE_VERSION, DmlResource::d3d12_device_t, &resource);
    if (status) {
      ORT_CXX_API_THROW("failed to fetch dml d3d12 device", OrtErrorCode::ORT_RUNTIME_EXCEPTION);
    }
    d3d12_device = reinterpret_cast<ID3D12Device*>(resource);

    resource = {};
    status = ort_api.KernelContext_GetResource(&kernel_ctx, ORT_DML_RESOUCE_VERSION, DmlResource::cmd_list_t, &resource);
    if (status) {
      ORT_CXX_API_THROW("failed to fetch command list", OrtErrorCode::ORT_RUNTIME_EXCEPTION);
    }
    cmd_list = reinterpret_cast<ID3D12GraphicsCommandList*>(resource);

    resource = {};
    status = ort_api.KernelContext_GetResource(&kernel_ctx, ORT_DML_RESOUCE_VERSION, DmlResource::cmd_recorder_t, &resource);
    if (status) {
      ORT_CXX_API_THROW("failed to fetch command recorder", OrtErrorCode::ORT_RUNTIME_EXCEPTION);
    }
    cmd_recorder = reinterpret_cast<IDMLCommandRecorder*>(resource);
  }
};

}  // namespace Custom
}  // namespace Ort
