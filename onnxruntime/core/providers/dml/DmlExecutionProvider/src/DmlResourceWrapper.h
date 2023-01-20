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
        virtual ID3D12Resource* GetUavResource() const = 0;
        virtual ID3D12Resource* GetCopySrcResource() const = 0;
        virtual ID3D12Resource* GetCopyDstResource() const = 0;
        virtual D3D12_RESOURCE_STATES GetDefaultUavState() const = 0;
        virtual D3D12_RESOURCE_STATES GetDefaultCopySrcState() const = 0;
        virtual D3D12_RESOURCE_STATES GetDefaultCopyDstState() const = 0;
        virtual ~DmlResourceWrapper(){}
    };
}
