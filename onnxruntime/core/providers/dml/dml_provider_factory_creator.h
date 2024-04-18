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

constexpr GUID dml_execution_context_guid = {0xe270a9ef, 0x41ab, 0x430f, {0xb6, 0xd4, 0xaf, 0x50, 0x1e, 0xb6, 0x7e, 0xb7}};
constexpr GUID dml_device_guid = {0x92389a9f, 0x372c, 0x43c1, {0x90, 0xe2, 0xb0, 0xa3, 0xc9, 0x00, 0xd3, 0x0b}};

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
