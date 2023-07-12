// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

namespace Dml
{
    // Represents a region of a D3D12 buffer resource. A buffer region has an
    // underlying ID3D12Resource* (of D3D12_RESOURCE_DIMENSION_BUFFER), an offset in
    // bytes from the beginning of that buffer, and a size in bytes of the region.
    class D3D12BufferRegion
    {
    public:
        D3D12BufferRegion() = default;

        // References a region of a buffer. The respective ID3D12Resource objects
        // must be in the appropriate states. Each resource is optional, but if more
        // than one are provided they must map to the same region of memory.
        D3D12BufferRegion(
            uint64_t offset,
            uint64_t size_in_bytes,
            ID3D12Resource* resource_uav_state,
            ID3D12Resource* resource_copy_src_state,
            ID3D12Resource* resource_copy_dst_state);

        // Move-only
        D3D12BufferRegion(const D3D12BufferRegion&) = default;
        D3D12BufferRegion& operator=(const D3D12BufferRegion&) = default;
        D3D12BufferRegion(D3D12BufferRegion&&);
        D3D12BufferRegion& operator=(D3D12BufferRegion&&);

        ID3D12Resource* ResourceInUavState() const;

        // NOTE: may be any state that is valid as a copy source (COPY_SRC,
        // GENERIC_READ, or COMMON).
        ID3D12Resource* ResourceInCopySrcState() const;

        ID3D12Resource* ResourceInCopyDstState() const;

        uint64_t Offset() const;
        uint64_t SizeInBytes() const;

        DML_BUFFER_BINDING GetBufferBinding() const;

        explicit operator bool() const { return first_valid_resource_ != nullptr; }

        // Creates a subregion at an offset from the start of this region. If no
        // size is provided the region runs to the end of the current region.
        inline D3D12BufferRegion Subregion(
            uint64_t offset,
            uint64_t size_in_bytes = 0) const
        {
            // start of subregion must be within current region
            ORT_THROW_HR_IF(E_INVALIDARG, offset >= size_in_bytes_);
            size_in_bytes =
                size_in_bytes == 0 ? size_in_bytes_ - offset : size_in_bytes;
            // end of subregion must be within current region
            ORT_THROW_HR_IF(E_INVALIDARG, size_in_bytes > size_in_bytes_ - offset);

            return D3D12BufferRegion(
                offset_ + offset,
                size_in_bytes,
                resource_uav_state_,
                resource_copy_src_state_,
                resource_copy_dst_state_);
        }

    private:
        ID3D12Resource* resource_uav_state_ = nullptr;
        ID3D12Resource* resource_copy_src_state_ = nullptr;
        ID3D12Resource* resource_copy_dst_state_ = nullptr;
        uint64_t offset_ = 0;
        uint64_t size_in_bytes_ = 0;

        // Pointer to the first resource above that isn't null.
        ID3D12Resource* first_valid_resource_ = nullptr;
    };

} // namespace Dml
