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

        ID3D12Resource* GetD3D12Resource() const final { return m_allocation.resourceUavState.Get(); }

    private:
        DmlHeapAllocation m_allocation;
    };
}
