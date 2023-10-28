// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "DmlSubAllocator.h"

namespace Dml
{
    struct DmlResourceWrapper;

    class DmlCommittedResourceAllocator : public DmlSubAllocator
    {
    public:
        DmlCommittedResourceAllocator(ID3D12Device* device) : m_device(device) {}
        Microsoft::WRL::ComPtr<DmlResourceWrapper> Alloc(size_t size) final;

        ~DmlCommittedResourceAllocator();

        void SetResidency(bool value) final;

    private:
        ID3D12Device* m_device = nullptr;
        std::vector<ID3D12Pageable*> m_resources;
        bool m_isResident = true;

        static void OnResourceRelease(void* context, ID3D12Resource* resource);
    };
}
