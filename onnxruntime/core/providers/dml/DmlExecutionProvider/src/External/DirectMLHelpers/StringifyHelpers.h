//  Copyright (c) Microsoft Corporation.  All rights reserved.
#pragma once

#pragma warning(push)
#pragma warning(disable:4499)

// This should only include ToString functions for enums which are not part of DirectMLSchema.json
namespace StringifyHelpers
{
    template <typename T>
    static const char* ToString(T value)
    {
#ifndef WAI_BUILD_LINUX
        // Clang will instantiate this template even if it isn't used,
        // so this static_assert will always fire and break the build.
        static_assert(false, "Not implemented for this type");
#endif
    }

    template<>
    static const char* ToString(DmlStage value)
    {
        switch (value)
        {
        case DmlStage::Input:      return "input";
        case DmlStage::Output:     return "output";
        case DmlStage::Persistent: return "the persistent resource";
        case DmlStage::GlobalMemoryTemporary:  return "the global memory temporary resource";
        case DmlStage::FixedFunctionCacheMemoryTemporary:  return "the fixed function cache memory temporary resource";
        case DmlStage::DspCacheMemoryTemporary:  return "the DSP cache memory temporary resource";

        default:
            return "<unknown>";
        }
    }

    template<>
    static const char* ToString(D3D12_RESOURCE_DIMENSION value)
    {
        switch (value)
        {
        case D3D12_RESOURCE_DIMENSION_BUFFER: return "D3D12_RESOURCE_DIMENSION_BUFFER";
        case D3D12_RESOURCE_DIMENSION_TEXTURE1D: return "D3D12_RESOURCE_DIMENSION_TEXTURE1D";
        case D3D12_RESOURCE_DIMENSION_TEXTURE2D: return "D3D12_RESOURCE_DIMENSION_TEXTURE2D";
        case D3D12_RESOURCE_DIMENSION_TEXTURE3D: return "D3D12_RESOURCE_DIMENSION_TEXTURE3D";
        case D3D12_RESOURCE_DIMENSION_UNKNOWN: return "D3D12_RESOURCE_DIMENSION_UNKNOWN";

        default:
            return "<unknown>";
        }
    }
}

#pragma warning(pop)