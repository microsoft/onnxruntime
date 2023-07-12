// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "DmlReservedResourceWrapper.h"

namespace Dml
{
    class DmlReservedResourceSubAllocator;

    class AllocationInfo : public Microsoft::WRL::RuntimeClass<
        Microsoft::WRL::RuntimeClassFlags<Microsoft::WRL::ClassicCom>, IUnknown>
    {
    public:
        AllocationInfo(
            DmlReservedResourceSubAllocator* owner,
            size_t id,
            DmlResourceWrapper* resourceWrapper,
            size_t requestedSize)
            : m_owner(owner)
            , m_allocationId(id)
            , m_resourceWrapper(resourceWrapper)
            , m_requestedSize(requestedSize)
        {}

        ~AllocationInfo();

        DmlReservedResourceSubAllocator* GetOwner() const
        {
            return m_owner;
        }

        ID3D12Resource* GetUavResource() const
        {
            return m_resourceWrapper->GetUavResource();
        }

        ID3D12Resource* GetCopySrcResource() const
        {
            return m_resourceWrapper->GetCopySrcResource();
        }

        ID3D12Resource* GetCopyDstResource() const
        {
            return m_resourceWrapper->GetCopyDstResource();
        }

        ComPtr<DmlResourceWrapper> DetachResourceWrapper() const
        {
            return std::move(m_resourceWrapper);
        }

        size_t GetRequestedSize() const
        {
            return m_requestedSize;
        }

        size_t GetId() const
        {
            return m_allocationId;
        }

    private:
        DmlReservedResourceSubAllocator* m_owner;
        size_t m_allocationId; // For debugging purposes
        Microsoft::WRL::ComPtr<DmlResourceWrapper> m_resourceWrapper;

        // The size requested during Alloc(), which may be smaller than the physical resource size
        size_t m_requestedSize;
    };
} // namespace Dml
