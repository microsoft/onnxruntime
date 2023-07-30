// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "precomp.h"
#include "DmlExternalGpuAllocator.h"
#include "DmlResourceWrapper.h"
#include "DmlCommittedResourceWrapper.h"
#include "DmlAllocationInfo.h"

namespace Dml
{
    DmlExternalGpuAllocator::DmlExternalGpuAllocator(ID3D12Device* device)
    : onnxruntime::IAllocator(
        OrtMemoryInfo(
            onnxruntime::DML,
            OrtAllocatorType::OrtDeviceAllocator,
            OrtDevice(OrtDevice::GPU, OrtDevice::MemType::DML_EXTERNAL, 0),
            -1
        )),
        m_device(device)
    {
    }

    void* DmlExternalGpuAllocator::Alloc(size_t size_in_bytes)
    {
        Microsoft::WRL::ComPtr<ID3D12Resource> resource;
        auto buffer = CD3DX12_RESOURCE_DESC::Buffer(size_in_bytes, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        auto props = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);
        ORT_THROW_IF_FAILED(m_device->CreateCommittedResource(
            &props,
            D3D12_HEAP_FLAG_NONE,
            &buffer,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            nullptr,
            IID_GRAPHICS_PPV_ARGS(resource.GetAddressOf())
        ));

        const uint64_t resourceWidth = resource->GetDesc().Width;
        constexpr uint64_t pooledResourceId = 0; // Not a pooled resource

        Microsoft::WRL::ComPtr<DmlResourceWrapper> resourceWrapper;
        wil::MakeOrThrow<DmlCommittedResourceWrapper>(std::move(resource)).As(&resourceWrapper);

        Microsoft::WRL::ComPtr<AllocationInfo> allocInfo = wil::MakeOrThrow<AllocationInfo>(
            nullptr,
            0,
            pooledResourceId,
            resourceWrapper.Get(),
            static_cast<size_t>(resourceWidth));

        return allocInfo.Detach();
    }

    void DmlExternalGpuAllocator::Free(void* ptr)
    {
        Microsoft::WRL::ComPtr<AllocationInfo> resource;
        resource.Attach(static_cast<AllocationInfo*>(ptr));
    }

} // namespace Dml
