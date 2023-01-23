// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "precomp.h"
#include "DmlBufferRegion.h"
#include "DmlHeapAllocator.h"

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
    ORT_THROW_HR_IF(E_UNEXPECTED, first_valid_resource_ == nullptr);

    // Regions cannot be empty.
    ORT_THROW_HR_IF(E_UNEXPECTED, size_in_bytes_ == 0);

    // Regions cannot extend beyond the size of the resource.
    uint64_t buffer_size = first_valid_resource_->GetDesc().Width;
    ORT_THROW_HR_IF(E_UNEXPECTED, offset_ >= buffer_size);
    ORT_THROW_HR_IF(E_UNEXPECTED, size_in_bytes_ > buffer_size - offset);

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

ID3D12Resource* D3D12BufferRegion::GetResourceInUavState() const
{
    return resource_uav_state_;
}

ID3D12Resource* D3D12BufferRegion::GetResourceInCopySrcState() const
{
    return resource_copy_src_state_;
}

ID3D12Resource* D3D12BufferRegion::GetResourceInCopyDstState() const
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

D3D12BufferRegion GetBufferForOpaqueData(
    D3D12HeapAllocator* allocator,
    const void* opaque_data,
    uint64_t unaligned_size_in_bytes)
{
    // DML always requires at least 4 byte alignment in all cases, so both the
    // offset and size must certainly be divisible by 4.
    constexpr uint64_t DML_ALIGNMENT = 4;

    // The offset and size of the region must be aligned to DirectML's
    // requirement. Each tensor has two sizes:
    //
    // - TotalBytes: num_elements * sizeof_element. This may be too small if the
    // tensor has elements smaller than 4 bytes (e.g. 3x float16 is 6 bytes, but
    // DML needs an 8 byte region).
    //
    // - AllocatedBytes: the size of allocation backing the tensor. This is
    // often larger than TotalBytes since the smallest DML allocation size is
    // 256 bytes.
    //
    // While AllocatedBytes is guaranteed to meet DML's requirement, tensor
    // buffers may be offset within an individual allocation (see
    // Tensor::Slice). Using AllocatedBytes directly can result in a region that
    // extends beyond the bounds of the allocation. Instead we round the total
    // bytes up to an aligned value, which should always fit within the
    // allocated bytes.
    uint64_t size_in_bytes =
        (1 + (unaligned_size_in_bytes - 1) / DML_ALIGNMENT) * DML_ALIGNMENT;

    auto region = allocator->CreateBufferRegion(opaque_data, size_in_bytes);

    // DML always requires at least 4 byte alignment in all cases, so both the
    // offset and size must certainly be divisible by 4
    assert(region.Offset() % DML_ALIGNMENT == 0);
    assert(region.SizeInBytes() % DML_ALIGNMENT == 0);

    return region;
}

}
