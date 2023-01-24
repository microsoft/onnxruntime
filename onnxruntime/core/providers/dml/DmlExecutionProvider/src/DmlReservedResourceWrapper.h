// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "DmlResourceWrapper.h"
#include "DmlHeapAllocation.h"
#include "DmlTaggedPointer.h"

namespace Dml
{
    class DmlReservedResourceWrapper : public Microsoft::WRL::RuntimeClass<Microsoft::WRL::RuntimeClassFlags<Microsoft::WRL::ClassicCom>, DmlResourceWrapper>
    {
    public:
        DmlReservedResourceWrapper(DmlHeapAllocation&& allocation)
            : m_allocation(std::move(allocation))
        {
        }

        ID3D12Resource* GetUavResource() const final { return m_allocation.resource_uav_state.Get(); }
        ID3D12Resource* GetCopySrcResource() const final { return m_allocation.resource_copy_src_state.Get(); }
        ID3D12Resource* GetCopyDstResource() const final { return m_allocation.resource_copy_dst_state.Get(); }

        D3D12_RESOURCE_STATES GetDefaultUavState() const final { return D3D12_RESOURCE_STATE_UNORDERED_ACCESS; }
        D3D12_RESOURCE_STATES GetDefaultCopySrcState() const final { return D3D12_RESOURCE_STATE_COPY_SOURCE; }
        D3D12_RESOURCE_STATES GetDefaultCopyDstState() const final { return D3D12_RESOURCE_STATE_COPY_DEST; }

    private:
        DmlHeapAllocation m_allocation;
    };
}
