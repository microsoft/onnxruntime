// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <wrl/client.h>
#include "directx/d3d12.h"
#include <wil/wrl.h>
#include <wil/result_macros.h>
#include "External/D3DX12/d3dx12.h"
#include "core/framework/allocator.h"
#include "core/providers/dml/dml_provider_factory_creator.h"
#include "AllocationInfo.h"
#include "GraphicsUnknownHelper.h"
#include "ErrorHandling.h"
#include "DmlCommittedResourceWrapper.h"

namespace Dml
{
    class DmlInputBufferAllocator : public onnxruntime::IAllocator
    {
    public:
        DmlInputBufferAllocator(ID3D12Device* d3d12Device) : onnxruntime::IAllocator(
            OrtMemoryInfo(
                "DML",
                OrtAllocatorType::OrtDeviceAllocator,
                OrtDevice(OrtDevice::GPU, OrtDevice::MemType::DML_INPUT, 0)
            )),
            m_d3d12Device(d3d12Device)
        {
        }

        void* Alloc(size_t size) final
        {
            Microsoft::WRL::ComPtr<ID3D12Resource> resource;
            auto buffer = CD3DX12_RESOURCE_DESC::Buffer(size, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
            auto props = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);
            ORT_THROW_IF_FAILED(m_d3d12Device->CreateCommittedResource(
                &props,
                D3D12_HEAP_FLAG_NONE,
                &buffer,
                D3D12_RESOURCE_STATE_COMMON,
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

        void Free(void* ptr) final
        {
            Microsoft::WRL::ComPtr<AllocationInfo> resource;
            resource.Attach(static_cast<AllocationInfo*>(ptr));
        }

    private:
        Microsoft::WRL::ComPtr<ID3D12Device> m_d3d12Device;
    };
}
