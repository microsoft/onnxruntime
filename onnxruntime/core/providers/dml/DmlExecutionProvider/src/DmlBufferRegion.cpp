// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "precomp.h"
#include "DmlBufferRegion.h"

namespace Dml
{

    D3D12BufferRegion::D3D12BufferRegion(uint64_t offset, uint64_t size_in_bytes, ID3D12Resource* resource)
        : m_resource(resource),
        offset_(offset),
        size_in_bytes_(size_in_bytes)
    {
        ORT_THROW_HR_IF(E_INVALIDARG, m_resource == nullptr);

        // Regions cannot be empty.
        ORT_THROW_HR_IF(E_INVALIDARG, size_in_bytes_ == 0);

        // Regions cannot extend beyond the size of the resource.
        uint64_t buffer_size = m_resource->GetDesc().Width;
        ORT_THROW_HR_IF(E_INVALIDARG, offset_ >= buffer_size);
        ORT_THROW_HR_IF(E_INVALIDARG, size_in_bytes_ > buffer_size - offset);

        // All three resources, if provided, must be identical aside from state.
        assert(m_resource->GetDesc().Dimension == D3D12_RESOURCE_DIMENSION_BUFFER);
        assert(m_resource->GetDesc().Width == buffer_size);
    }

    D3D12BufferRegion::D3D12BufferRegion(D3D12BufferRegion&& that)
    {
        std::swap(this->m_resource, that.m_resource);
        std::swap(this->offset_, that.offset_);
        std::swap(this->size_in_bytes_, that.size_in_bytes_);
    }

    D3D12BufferRegion& D3D12BufferRegion::operator=(D3D12BufferRegion&& that)
    {
        std::swap(this->m_resource, that.m_resource);
        std::swap(this->offset_, that.offset_);
        std::swap(this->size_in_bytes_, that.size_in_bytes_);
        return *this;
    }

    ID3D12Resource* D3D12BufferRegion::ResourceInUavState() const
    {
        return m_resource;
    }

    uint64_t D3D12BufferRegion::Offset() const
    {
        return m_resource ? offset_ : 0;
    }

    uint64_t D3D12BufferRegion::SizeInBytes() const
    {
        return m_resource ? size_in_bytes_ : 0;
    }

    DML_BUFFER_BINDING D3D12BufferRegion::GetBufferBinding() const
    {
        if (!m_resource)
        {
            return DML_BUFFER_BINDING{};
        }

        return DML_BUFFER_BINDING{m_resource, offset_, size_in_bytes_};
    }

} // namespace Dml
