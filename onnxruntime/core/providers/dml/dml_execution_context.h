// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef ORT_DML_CTX
#define ORT_DML_CTX

#include "core/session/onnxruntime_cxx_api.h"
#include <DirectML.h>
#include <d3d12.h>

#define ORT_DML_RESOUCE_VERSION 1

enum DmlResource : int {
  dml_device_t = 0,
  d3d12_device_t,
  cmd_list_t,
  cmd_recorder_t
};

namespace Ort {

namespace Custom {

struct DmlContext {
  IDMLDevice* dml_device = {};
  ID3D12Device* d3d12_device = {};
  ID3D12GraphicsCommandList* cmd_list = {};
  IDMLCommandRecorder* cmd_recorder = {};

  void Init(const OrtKernelContext& kernel_ctx) {
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

} // Custom
} // Ort
#endif