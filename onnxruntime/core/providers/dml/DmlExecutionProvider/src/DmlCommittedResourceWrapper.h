// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "DmlResourceWrapper.h"

namespace Dml
{
    class DmlCommittedResourceWrapper : public Microsoft::WRL::RuntimeClass<Microsoft::WRL::RuntimeClassFlags<Microsoft::WRL::ClassicCom>, DmlResourceWrapper>
    {
    public:
        DmlCommittedResourceWrapper(ComPtr<ID3D12Resource>&& d3d12Resource) : m_d3d12Resource(std::move(d3d12Resource)) {}

        // Committed resources use the same resource for all states and use barriers to transition between states
        ID3D12Resource* GetUavResource() const final { return m_d3d12Resource.Get(); }
        ID3D12Resource* GetCopySrcResource() const final { return m_d3d12Resource.Get(); }
        ID3D12Resource* GetCopyDstResource() const final { return m_d3d12Resource.Get(); }

        D3D12_RESOURCE_STATES GetDefaultUavState() const final { return D3D12_RESOURCE_STATE_UNORDERED_ACCESS; }
        D3D12_RESOURCE_STATES GetDefaultCopySrcState() const final { return D3D12_RESOURCE_STATE_UNORDERED_ACCESS; }
        D3D12_RESOURCE_STATES GetDefaultCopyDstState() const final { return D3D12_RESOURCE_STATE_UNORDERED_ACCESS; }

    private:
        ComPtr<ID3D12Resource> m_d3d12Resource;
    };
}
