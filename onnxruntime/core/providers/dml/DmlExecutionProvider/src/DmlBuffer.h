// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "DmlBufferRegion.h"
#include "DmlAllocatorRoundingMode.h"

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
    explicit DmlBuffer(DmlGpuAllocator* allocator, uint64_t sizeInBytes, AllocatorRoundingMode roundingMode);
    ~DmlBuffer();

    // Move-only
    DmlBuffer(const DmlBuffer&) = delete;
    DmlBuffer& operator=(const DmlBuffer&) = delete;
    DmlBuffer(DmlBuffer&&) noexcept;
    DmlBuffer& operator=(DmlBuffer&&) noexcept;

    ID3D12Resource* GetD3D12Resource() const;
    uint64_t Offset() const;
    uint64_t SizeInBytes() const;
    const D3D12BufferRegion& Region() const { return m_bufferRegion; }
    DML_BUFFER_BINDING GetBufferBinding() const;

    explicit operator bool() const { return !!m_bufferRegion; }

  private:
    DmlGpuAllocator* m_allocator;
    D3D12BufferRegion m_bufferRegion;
    void* m_opaqueData;
};

} // namespace tfdml
