// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "DmlBufferRegion.h"
#include "DmlAllocationInfo.h"

namespace Dml
{
    class DmlManagedBufferRegion : public Microsoft::WRL::RuntimeClass<Microsoft::WRL::RuntimeClassFlags<Microsoft::WRL::ClassicCom>, IUnknown>
    {
    public:
        DmlManagedBufferRegion(Microsoft::WRL::ComPtr<AllocationInfo> allocation, D3D12BufferRegion&& bufferRegion)
            : m_allocation(std::move(allocation)),
              m_bufferRegion(std::move(bufferRegion))
        {
        }

        const D3D12BufferRegion& GetBufferRegion() const { return m_bufferRegion; }

    private:
        Microsoft::WRL::ComPtr<AllocationInfo> m_allocation;
        D3D12BufferRegion m_bufferRegion;
    };
}
