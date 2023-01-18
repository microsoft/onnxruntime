// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "DmlResourceWrapper.h"

namespace Dml
{
    class DmlCommittedResourceWrapper : public Microsoft::WRL::RuntimeClass<Microsoft::WRL::RuntimeClassFlags<Microsoft::WRL::ClassicCom>, DmlResourceWrapper>
    {
    public:
        DmlCommittedResourceWrapper(ComPtr<ID3D12Resource>&& d3d12Resource) : m_d3d12Resource(std::move(d3d12Resource)) {}
        ID3D12Resource* GetResourceInUavState() const final { return m_d3d12Resource.Get(); }
        ID3D12Resource* GetResourceInCopySrcState() const final { return nullptr; }
        ID3D12Resource* GetResourceInCopyDstState() const final { return nullptr; }

    private:
        ComPtr<ID3D12Resource> m_d3d12Resource;
    };
}
