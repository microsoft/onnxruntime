// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

struct ID3D12Resource;

namespace Dml
{
    typedef void (*ResourceCallback)(void* context, ID3D12Resource* resource);

    interface __declspec(uuid("d430f6f1-5c43-48d1-97e6-f080cc7fa0c5"))
    DmlResourceWrapper : public IUnknown
    {
    public:
        virtual ID3D12Resource* GetD3D12Resource() const = 0;
        virtual void AddReleaseCallback(ResourceCallback callback, void* context = nullptr) = 0;
        virtual ~DmlResourceWrapper(){}
    };
}
