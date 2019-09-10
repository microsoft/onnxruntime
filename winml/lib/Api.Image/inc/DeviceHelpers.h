//-----------------------------------------------------------------------------
//
//  Copyright (c) Microsoft Corporation. All rights reserved.
//
//-----------------------------------------------------------------------------

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

//
// Exception information
//
#ifndef FACILITY_VISUALCPP
#define FACILITY_VISUALCPP  ((LONG)0x6d)
#endif

#define VcppException(sev,err)  ((sev) | (FACILITY_VISUALCPP<<16) | err)

namespace DeviceHelpers
{
    struct AdapterEnumerationSupport 
    {
        bool hasDxgi;
        bool hasDxcore;
    };

    HRESULT GetAdapterEnumerationSupport(AdapterEnumerationSupport* support);
    bool IsFloat16Supported(ID3D12Device* device);
    bool IsFloat16Supported(const winrt::Windows::AI::MachineLearning::LearningModelDevice& device);
    HRESULT CreateD3D11On12Device(ID3D12Device* device12, ID3D11Device** device11);
#ifdef ENABLE_DXCORE
    HRESULT GetDXCoreHardwareAdapterWithPreference(DXGI_GPU_PREFERENCE preference, _COM_Outptr_ IDXCoreAdapter **ppAdapter);
#endif
    HRESULT GetDXGIHardwareAdapterWithPreference(DXGI_GPU_PREFERENCE preference, _COM_Outptr_ IDXGIAdapter1 **ppAdapter);
    HRESULT GetGPUPreference(winrt::Windows::AI::MachineLearning::LearningModelDeviceKind deviceKind, DXGI_GPU_PREFERENCE* preference) noexcept;
}