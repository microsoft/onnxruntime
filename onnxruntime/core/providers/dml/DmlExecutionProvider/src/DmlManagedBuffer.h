// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "DmlBuffer.h"

namespace Dml
{
    // Light wrapper around DmlBuffer used with CommandQueue::QueueReference to keep a reference on the buffer until GPU work is completed
    class DmlManagedBuffer : public Microsoft::WRL::RuntimeClass<Microsoft::WRL::RuntimeClassFlags<Microsoft::WRL::ClassicCom>, IUnknown>
    {
    public:
        DmlManagedBuffer(DmlBuffer&& buffer) : m_buffer(std::move(buffer)) {}
        uint64_t SizeInBytes() const { return m_buffer.SizeInBytes(); }

    private:
        DmlBuffer m_buffer;
    };
}
