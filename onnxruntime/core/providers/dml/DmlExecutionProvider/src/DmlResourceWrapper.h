// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

struct ID3D12Resource;

namespace Dml
{
    interface __declspec(uuid("d430f6f1-5c43-48d1-97e6-f080cc7fa0c5"))
    DmlResourceWrapper : public IUnknown
    {
    public:
        virtual ID3D12Resource* GetResourceInUavState() const = 0;
        virtual ID3D12Resource* GetResourceInCopySrcState() const = 0;
        virtual ID3D12Resource* GetResourceInCopyDstState() const = 0;
        virtual ~DmlResourceWrapper(){}
    };
}
