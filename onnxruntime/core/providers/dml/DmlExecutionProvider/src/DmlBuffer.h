// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "DmlBufferRegion.h"

namespace Dml
{

class DmlGpuAllocator;
class OpKernelContext;

// Owns a D3D12 default heap buffer allocated using the DML device's
// allocator. This is essentially a convenience wrapper over a device memory
// allocation as well as the buffer region that spans it. When this object is
// destructed, the device memory is freed to the allocator.
class DmlBuffer
{
  public:
    explicit DmlBuffer(DmlGpuAllocator* allocator, uint64_t size_in_bytes);
    ~DmlBuffer();

    // Move-only
    DmlBuffer(const DmlBuffer&) = delete;
    DmlBuffer& operator=(const DmlBuffer&) = delete;
    DmlBuffer(DmlBuffer&&);
    DmlBuffer& operator=(DmlBuffer&&);

    ID3D12Resource* ResourceInUavState() const;
    ID3D12Resource* ResourceInCopySrcState() const;
    ID3D12Resource* ResourceInCopyDstState() const;
    uint64_t Offset() const;
    uint64_t SizeInBytes() const;
    const D3D12BufferRegion& Region() const { return buffer_region_; }

    DML_BUFFER_BINDING GetBufferBinding() const;

    explicit operator bool() const { return !!buffer_region_; }

  private:
    DmlGpuAllocator* allocator_;
    D3D12BufferRegion buffer_region_;
    void* m_opaqueData;
};

} // namespace tfdml
