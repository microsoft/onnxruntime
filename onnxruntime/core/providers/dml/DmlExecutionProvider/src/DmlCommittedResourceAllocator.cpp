// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "precomp.h"
#include "DmlCommittedResourceAllocator.h"
#include "DmlResourceWrapper.h"
#include "DmlCommittedResourceWrapper.h"

namespace Dml
{
    ComPtr<DmlResourceWrapper> DmlCommittedResourceAllocator::Alloc(size_t size)
    {
        ComPtr<ID3D12Resource> resource;
        auto buffer = CD3DX12_RESOURCE_DESC::Buffer(size, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        ORT_THROW_IF_FAILED(m_device->CreateCommittedResource(
            &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT),
            D3D12_HEAP_FLAG_NONE,
            &buffer,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            nullptr,
            IID_GRAPHICS_PPV_ARGS(resource.GetAddressOf())
        ));

        ComPtr<DmlResourceWrapper> resourceWrapper;
        wil::MakeOrThrow<DmlCommittedResourceWrapper>(std::move(resource)).As(&resourceWrapper);
        return resourceWrapper;
    }
}
