// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "precomp.h"
#include "DmlBuffer.h"
#include "DmlGpuAllocator.h"

namespace Dml
{

/*explicit*/ DmlBuffer::DmlBuffer(DmlGpuAllocator* allocator, uint64_t sizeInBytes)
    : m_allocator(allocator)
{
    m_opaqueData = m_allocator->Alloc(sizeInBytes);
    ORT_THROW_HR_IF(E_OUTOFMEMORY, m_opaqueData == nullptr);

    m_bufferRegion = m_allocator->CreateBufferRegion(m_opaqueData, sizeInBytes);
}

DmlBuffer::~DmlBuffer()
{
    if (m_opaqueData != nullptr)
    {
        m_allocator->Free(m_opaqueData);
    }
}

DmlBuffer::DmlBuffer(DmlBuffer&& other) noexcept
{
    m_opaqueData = other.m_opaqueData;
    m_allocator = other.m_allocator;
    m_bufferRegion = std::move(other.m_bufferRegion);
    other.m_opaqueData = nullptr;
}

DmlBuffer& DmlBuffer::operator=(DmlBuffer&& other) noexcept
{
    m_opaqueData = other.m_opaqueData;
    m_allocator = other.m_allocator;
    m_bufferRegion = std::move(other.m_bufferRegion);
    other.m_opaqueData = nullptr;
    return *this;
}

ID3D12Resource* DmlBuffer::GetD3D12Resource() const
{
    return m_bufferRegion.GetD3D12Resource();
}

uint64_t DmlBuffer::Offset() const
{
    return m_bufferRegion ? m_bufferRegion.Offset() : 0;
}

uint64_t DmlBuffer::SizeInBytes() const
{
    return m_bufferRegion ? m_bufferRegion.SizeInBytes() : 0;
}

DML_BUFFER_BINDING DmlBuffer::GetBufferBinding() const
{
    return m_bufferRegion ? m_bufferRegion.GetBufferBinding()
                          : DML_BUFFER_BINDING{};
}

} // namespace Dml
