// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "precomp.h"
#include "DmlBufferRegion.h"

namespace Dml
{

    D3D12BufferRegion::D3D12BufferRegion(
        uint64_t offset,
        uint64_t size_in_bytes,
        ID3D12Resource* resource_uav_state,
        ID3D12Resource* resource_copy_src_state,
        ID3D12Resource* resource_copy_dst_state)
        : resource_uav_state_(resource_uav_state),
        resource_copy_src_state_(resource_copy_src_state),
        resource_copy_dst_state_(resource_copy_dst_state),
        offset_(offset),
        size_in_bytes_(size_in_bytes)
    {
        // Get a raw pointer to the first non-null resource passed in. At least one
        // resource must be provided.
        first_valid_resource_ = resource_uav_state_;
        if (!first_valid_resource_)
        {
            first_valid_resource_ = resource_copy_src_state_;
        }
        if (!first_valid_resource_)
        {
            first_valid_resource_ = resource_copy_dst_state_;
        }
        ORT_THROW_HR_IF(E_INVALIDARG, first_valid_resource_ == nullptr);

        // Regions cannot be empty.
        ORT_THROW_HR_IF(E_INVALIDARG, size_in_bytes_ == 0);

        // Regions cannot extend beyond the size of the resource.
        uint64_t buffer_size = first_valid_resource_->GetDesc().Width;
        ORT_THROW_HR_IF(E_INVALIDARG, offset_ >= buffer_size);
        ORT_THROW_HR_IF(E_INVALIDARG, size_in_bytes_ > buffer_size - offset);

        // All three resources, if provided, must be identical aside from state.
        assert(
            first_valid_resource_->GetDesc().Dimension ==
            D3D12_RESOURCE_DIMENSION_BUFFER);
        assert(
            !resource_uav_state ||
            (resource_uav_state->GetDesc().Dimension ==
                D3D12_RESOURCE_DIMENSION_BUFFER &&
            resource_uav_state->GetDesc().Width == buffer_size));
        assert(
            !resource_copy_src_state_ ||
            (resource_copy_src_state_->GetDesc().Dimension ==
                D3D12_RESOURCE_DIMENSION_BUFFER &&
            resource_copy_src_state_->GetDesc().Width == buffer_size));
        assert(
            !resource_copy_dst_state_ ||
            (resource_copy_dst_state_->GetDesc().Dimension ==
                D3D12_RESOURCE_DIMENSION_BUFFER &&
            resource_copy_dst_state_->GetDesc().Width == buffer_size));
    }

    D3D12BufferRegion::D3D12BufferRegion(D3D12BufferRegion&& that)
    {
        std::swap(this->resource_uav_state_, that.resource_uav_state_);
        std::swap(this->resource_copy_src_state_, that.resource_copy_src_state_);
        std::swap(this->resource_copy_dst_state_, that.resource_copy_dst_state_);
        std::swap(this->offset_, that.offset_);
        std::swap(this->size_in_bytes_, that.size_in_bytes_);
        std::swap(this->first_valid_resource_, that.first_valid_resource_);
    }

    D3D12BufferRegion& D3D12BufferRegion::operator=(D3D12BufferRegion&& that)
    {
        std::swap(this->resource_uav_state_, that.resource_uav_state_);
        std::swap(this->resource_copy_src_state_, that.resource_copy_src_state_);
        std::swap(this->resource_copy_dst_state_, that.resource_copy_dst_state_);
        std::swap(this->offset_, that.offset_);
        std::swap(this->size_in_bytes_, that.size_in_bytes_);
        std::swap(this->first_valid_resource_, that.first_valid_resource_);
        return *this;
    }

    ID3D12Resource* D3D12BufferRegion::ResourceInUavState() const
    {
        return resource_uav_state_;
    }

    ID3D12Resource* D3D12BufferRegion::ResourceInCopySrcState() const
    {
        return resource_copy_src_state_;
    }

    ID3D12Resource* D3D12BufferRegion::ResourceInCopyDstState() const
    {
        return resource_copy_dst_state_;
    }

    uint64_t D3D12BufferRegion::Offset() const
    {
        return first_valid_resource_ ? offset_ : 0;
    }

    uint64_t D3D12BufferRegion::SizeInBytes() const
    {
        return first_valid_resource_ ? size_in_bytes_ : 0;
    }

    DML_BUFFER_BINDING D3D12BufferRegion::GetBufferBinding() const
    {
        if (!resource_uav_state_)
        {
            return DML_BUFFER_BINDING{};
        }

        return DML_BUFFER_BINDING{resource_uav_state_, offset_, size_in_bytes_};
    }

} // namespace Dml
