// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <dxgi1_6.h>
#include <initguid.h>
#include <d3d11.h>

#if __has_include("dxcore.h")
#define ENABLE_DXCORE 1
#endif
#ifdef ENABLE_DXCORE
#include <dxcore.h>
#endif

namespace _winml {

HRESULT CreateD3D11On12Device(ID3D12Device* device12, ID3D11Device** device11);

#ifdef ENABLE_DXCORE
HRESULT GetDXCoreHardwareAdapterWithPreference(DXGI_GPU_PREFERENCE preference, _COM_Outptr_ IDXCoreAdapter** ppAdapter);
#endif

HRESULT GetDXGIHardwareAdapterWithPreference(DXGI_GPU_PREFERENCE preference, _COM_Outptr_ IDXGIAdapter1** adapter);

HRESULT GetGPUPreference(winml::LearningModelDeviceKind deviceKind, DXGI_GPU_PREFERENCE* preference) noexcept;

}  // namespace _winml
