// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>

#include <wrl/client.h>
#include <d3d12.h>
#include "core/providers/providers.h"
#include "core/providers/dml/dml_provider_factory.h"

namespace onnxruntime {

struct DMLProviderFactoryCreator {
  static std::shared_ptr<IExecutionProviderFactory> Create(int device_id);
  static std::shared_ptr<IExecutionProviderFactory> Create(int device_id, bool skip_software_device_check);
  static Microsoft::WRL::ComPtr<ID3D12Device> CreateD3D12Device(int device_id, bool skip_software_device_check);
  static Microsoft::WRL::ComPtr<IDMLDevice> CreateDMLDevice(ID3D12Device* d3d12_device);
};
}  // namespace onnxruntime
