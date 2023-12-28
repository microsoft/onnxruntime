// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

namespace Dml
{
    struct DmlResourceWrapper;

    class DmlSubAllocator
    {
    public:
        virtual Microsoft::WRL::ComPtr<DmlResourceWrapper> Alloc(size_t size) = 0;
        virtual void SetResidency(bool value) = 0;
        virtual ~DmlSubAllocator(){}
    };
}
