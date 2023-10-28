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
            unmove_ptr(CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT)),
            D3D12_HEAP_FLAG_NONE,
            &buffer,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            nullptr,
            IID_GRAPHICS_PPV_ARGS(resource.GetAddressOf())
        ));

        ID3D12Pageable* pageable;
        ORT_THROW_IF_FAILED(resource->QueryInterface(&pageable));
        m_resources.push_back(pageable);

        ComPtr<DmlResourceWrapper> resourceWrapper;
        wil::MakeOrThrow<DmlCommittedResourceWrapper>(std::move(resource)).As(&resourceWrapper);
        resourceWrapper->AddReleaseCallback(&DmlCommittedResourceAllocator::OnResourceRelease, this);

        return resourceWrapper;
    }

    DmlCommittedResourceAllocator::~DmlCommittedResourceAllocator()
    {
        for (auto& item : m_resources)
        {
            item->Release();
        }
    }

    void DmlCommittedResourceAllocator::SetResidency(bool value)
    {
        if (m_isResident == value) return;

        if (value)
        {
            ORT_THROW_IF_FAILED(m_device->MakeResident(UINT(m_resources.size()), m_resources.data()));
        }
        else
        {
            ORT_THROW_IF_FAILED(m_device->Evict(UINT(m_resources.size()), m_resources.data()));
        }

        m_isResident = value;
    }

    void DmlCommittedResourceAllocator::OnResourceRelease(void * context, ID3D12Resource * resource)
    {
        auto that = static_cast<DmlCommittedResourceAllocator*>(context);

        ComPtr<ID3D12Pageable> pageable;
        ORT_THROW_IF_FAILED(resource->QueryInterface(pageable.GetAddressOf()));

        for (auto& item : that->m_resources)
        {
            if (item == resource)
            {
                item->Release();

                std::swap(item, that->m_resources.back());
                that->m_resources.pop_back();
                break;
            }
        }
    }
}
