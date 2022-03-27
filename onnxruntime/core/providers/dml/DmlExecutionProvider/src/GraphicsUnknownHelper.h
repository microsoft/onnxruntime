// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

namespace Dml
{
#ifdef _GAMING_XBOX
// Helper to pass IGraphicsUnknown objects as IUnknown
class GraphicsUnknownWrapper : public IUnknown
{
public:
    GraphicsUnknownWrapper(IGraphicsUnknown* gfxUnk)
        : m_inner(gfxUnk)
    {
    }

    void Assign(IGraphicsUnknown* gfxUnk)
    {
        m_inner = gfxUnk;
    }

    ULONG AddRef()
    {
        return m_inner->AddRef();
    }

    ULONG Release()
    {
        return m_inner->Release();
    }

    virtual HRESULT STDMETHODCALLTYPE QueryInterface(
        REFIID riid,
        _COM_Outptr_ void __RPC_FAR* __RPC_FAR* ppvObject)
    {
        return m_inner->QueryInterface(riid, ppvObject);
    }

private:
    Microsoft::WRL::ComPtr<IGraphicsUnknown> m_inner;
};

IUnknown* WrapGraphicsUnknown(IGraphicsUnknown* graphicsUnknown)
{
    // TODO: this is a hack to build but will leak and needs to be replaced.
    Microsoft::WRL::ComPtr<GraphicsUnknownWrapper> wrapper{ new GraphicsUnknownWrapper(graphicsUnknown) };
    return wrapper.Detach();
}

#else // !_GAMING_XBOX

#define IGraphicsUnknown IUnknown
#define IID_GRAPHICS_PPV_ARGS IID_PPV_ARGS

IUnknown* WrapGraphicsUnknown(IGraphicsUnknown* graphicsUnknown)
{
    return graphicsUnknown;
}

#endif

Microsoft::WRL::ComPtr<ID3D12Device> GetDeviceFromDeviceChild(ID3D12DeviceChild* child)
{
    Microsoft::WRL::ComPtr<ID3D12Device> device;
#ifdef _GAMING_XBOX
    // Return type of ID3D12DeviceChild::GetDevice is void
    child->GetDevice(IID_GRAPHICS_PPV_ARGS(device.ReleaseAndGetAddressOf()));
#else
    // Return type of ID3D12DeviceChild::GetDevice is HRESULT
    ORT_THROW_IF_FAILED(child->GetDevice(IID_PPV_ARGS(device.ReleaseAndGetAddressOf())));
#endif
    return device;
}

} // namespace Dml