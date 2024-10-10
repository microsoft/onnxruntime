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
    class DmlUnpooledBufferAllocator : public onnxruntime::IAllocator, public IDmlBufferAllocator, public std::enable_shared_from_this<DmlUnpooledBufferAllocator>
    {
    public:
        DmlUnpooledBufferAllocator(ID3D12Device* d3d12Device, ExecutionContext* context, OrtDevice::MemoryType memoryType) : onnxruntime::IAllocator(
            OrtMemoryInfo(
                "DML",
                OrtAllocatorType::OrtDeviceAllocator,
                OrtDevice(OrtDevice::GPU, memoryType, 0)
            )),
            m_d3d12Device(d3d12Device),
            m_context(context)
        {
        }

        virtual ~DmlUnpooledBufferAllocator() = default;

        void* Alloc(size_t size) final
        {
            size = (size + 3) & ~3;

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
                shared_from_this(),
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

        void FreeResource(void* p, uint64_t) final
        {
            AllocationInfo *allocInfo = static_cast<AllocationInfo*>(p);

            assert(allocInfo != nullptr); // Can't free nullptr

            if (allocInfo->GetOwner() != this)
            {
                // This allocation doesn't belong to this allocator!
                ORT_THROW_HR(E_INVALIDARG);
            }

            // Free the resource to the pool if its size matches a bucket size
            if (!m_context->IsClosed())
            {
                // Free the underlying allocation once queued work has completed.
#ifdef _GAMING_XBOX
                m_context->QueueReference(WRAP_GRAPHICS_UNKNOWN(allocInfo->GetResource()).Get());
#else
                m_context->QueueReference(allocInfo->GetResource());
#endif
            }

            allocInfo->DetachResourceWrapper();
        }

    private:
        Microsoft::WRL::ComPtr<ID3D12Device> m_d3d12Device;
        Microsoft::WRL::ComPtr<ExecutionContext> m_context;
    };
}
