// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "precomp.h"
#include "DmlBufferRegion.h"

namespace Dml
{

    D3D12BufferRegion::D3D12BufferRegion(uint64_t offset, uint64_t sizeInBytes, ID3D12Resource* resource)
        : m_resource(resource),
        m_offset(offset),
        m_sizeInBytes(sizeInBytes)
    {
        ORT_THROW_HR_IF(E_INVALIDARG, m_resource == nullptr);

        // Regions cannot be empty.
        ORT_THROW_HR_IF(E_INVALIDARG, m_sizeInBytes == 0);

        // Regions cannot extend beyond the size of the resource.
        uint64_t bufferSize = m_resource->GetDesc().Width;
        ORT_THROW_HR_IF(E_INVALIDARG, m_offset >= bufferSize);
        ORT_THROW_HR_IF(E_INVALIDARG, m_sizeInBytes > bufferSize - offset);

        // All three resources, if provided, must be identical aside from state.
        assert(m_resource->GetDesc().Dimension == D3D12_RESOURCE_DIMENSION_BUFFER);
        assert(m_resource->GetDesc().Width == bufferSize);
    }

    D3D12BufferRegion::D3D12BufferRegion(D3D12BufferRegion&& that) noexcept
    {
        std::swap(this->m_resource, that.m_resource);
        std::swap(this->m_offset, that.m_offset);
        std::swap(this->m_sizeInBytes, that.m_sizeInBytes);
    }

    D3D12BufferRegion& D3D12BufferRegion::operator=(D3D12BufferRegion&& that) noexcept
    {
        std::swap(this->m_resource, that.m_resource);
        std::swap(this->m_offset, that.m_offset);
        std::swap(this->m_sizeInBytes, that.m_sizeInBytes);
        return *this;
    }

    ID3D12Resource* D3D12BufferRegion::GetD3D12Resource() const
    {
        return m_resource;
    }

    uint64_t D3D12BufferRegion::Offset() const
    {
        return m_resource ? m_offset : 0;
    }

    uint64_t D3D12BufferRegion::SizeInBytes() const
    {
        return m_resource ? m_sizeInBytes : 0;
    }

    DML_BUFFER_BINDING D3D12BufferRegion::GetBufferBinding() const
    {
        if (!m_resource)
        {
            return DML_BUFFER_BINDING{};
        }

        return DML_BUFFER_BINDING{m_resource, m_offset, m_sizeInBytes};
    }

} // namespace Dml
