// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "DmlResourceWrapper.h"
#include "DmlBufferRegion.h"
#include "DmlHeapAllocator.h"

namespace Dml
{
    class DmlReservedResourceWrapper : public Microsoft::WRL::RuntimeClass<Microsoft::WRL::RuntimeClassFlags<Microsoft::WRL::ClassicCom>, DmlResourceWrapper>
    {
    public:
        DmlReservedResourceWrapper(Allocation&& allocation) : m_allocation(std::move(allocation)) {}
        ID3D12Resource* GetResourceInUavState() const final { return m_allocation.resource_uav_state.Get(); }
        ID3D12Resource* GetResourceInCopySrcState() const final { return m_allocation.resource_copy_src_state.Get(); }
        ID3D12Resource* GetResourceInCopyDstState() const final { return m_allocation.resource_copy_dst_state.Get(); }

    private:
        Allocation m_allocation;
    };
}
