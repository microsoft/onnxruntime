// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "precomp.h"
#include "DmlBuffer.h"
#include "DmlGpuAllocator.h"

namespace Dml
{

/*explicit*/ DmlBuffer::DmlBuffer(DmlGpuAllocator* allocator, uint64_t size_in_bytes)
    : allocator_(allocator)
{
    m_opaqueData = allocator_->Alloc(size_in_bytes);
    ORT_THROW_HR_IF(E_OUTOFMEMORY, m_opaqueData == nullptr);

    buffer_region_ = allocator_->CreateBufferRegion(m_opaqueData, size_in_bytes);
}

DmlBuffer::~DmlBuffer()
{
    if (m_opaqueData != nullptr)
    {
        allocator_->Free(m_opaqueData);
    }
}

DmlBuffer::DmlBuffer(DmlBuffer&& other) noexcept
{
    m_opaqueData = other.m_opaqueData;
    allocator_ = other.allocator_;
    buffer_region_ = std::move(other.buffer_region_);
    other.m_opaqueData = nullptr;
}

DmlBuffer& DmlBuffer::operator=(DmlBuffer&& other) noexcept
{
    m_opaqueData = other.m_opaqueData;
    allocator_ = other.allocator_;
    buffer_region_ = std::move(other.buffer_region_);
    other.m_opaqueData = nullptr;
    return *this;
}

ID3D12Resource* DmlBuffer::GetD3D12Resource() const
{
    return buffer_region_.GetD3D12Resource();
}

uint64_t DmlBuffer::Offset() const
{
    return buffer_region_ ? buffer_region_.Offset() : 0;
}

uint64_t DmlBuffer::SizeInBytes() const
{
    return buffer_region_ ? buffer_region_.SizeInBytes() : 0;
}

DML_BUFFER_BINDING DmlBuffer::GetBufferBinding() const
{
    return buffer_region_ ? buffer_region_.GetBufferBinding()
                          : DML_BUFFER_BINDING{};
}

} // namespace Dml
