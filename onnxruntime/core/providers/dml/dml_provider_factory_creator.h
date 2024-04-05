// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>

#include <wrl/client.h>
#include "directx/d3d12.h"
#include "core/framework/provider_options.h"
#include "core/providers/providers.h"
#include "core/providers/dml/dml_provider_factory.h"
#include "core/framework/config_options.h"

#include <directx/dxcore.h>
#include <vector>

constexpr GUID dml_command_queue_guid = {0x9270ce8c, 0x7150, 0x4944, {0xa1, 0xda, 0x5c, 0x07, 0x60, 0xd5, 0x68, 0x10}};

namespace onnxruntime {

struct DMLProviderFactoryCreator {
  static std::shared_ptr<IExecutionProviderFactory> Create(
    const ConfigOptions& config_options,
    int device_id,
    bool skip_software_device_check,
    bool disable_metacommands,
    bool python_api = false);

  static std::shared_ptr<IExecutionProviderFactory> CreateFromProviderOptions(
    const ConfigOptions& config_options,
    const ProviderOptions& provider_options_map,
    bool python_api = false);

  static std::shared_ptr<IExecutionProviderFactory> CreateFromDeviceOptions(
    const ConfigOptions& config_options,
    const OrtDmlDeviceOptions* device_options,
    bool disable_metacommands,
    bool python_api = false);

  static std::shared_ptr<IExecutionProviderFactory> CreateFromAdapterList(
    const ConfigOptions& config_options,
    std::vector<Microsoft::WRL::ComPtr<IDXCoreAdapter>>&& dxcore_devices,
    bool disable_metacommands,
    bool python_api = false);

  static Microsoft::WRL::ComPtr<ID3D12Device> CreateD3D12Device(int device_id, bool skip_software_device_check);
  static Microsoft::WRL::ComPtr<IDMLDevice> CreateDMLDevice(ID3D12Device* d3d12_device);
};
}  // namespace onnxruntime
