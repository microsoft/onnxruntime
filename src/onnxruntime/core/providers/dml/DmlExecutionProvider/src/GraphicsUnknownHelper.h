// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

namespace Dml
{
// D3D12.x interfaces inherit from IGraphicsUnknown, which does not inherit from IUnknown. This
// wrapper exists to pass IGraphicsUnknown-inheriting objects to functions with IUnknown parameters.
#ifdef _GAMING_XBOX
class GraphicsUnknownWrapper : public Microsoft::WRL::RuntimeClass<Microsoft::WRL::RuntimeClassFlags<Microsoft::WRL::ClassicCom>, IUnknown>
{
public:
    explicit GraphicsUnknownWrapper(Microsoft::WRL::ComPtr<IGraphicsUnknown> graphicsUnknown) : m_graphicsUnknown(std::move(graphicsUnknown)) {}
    explicit GraphicsUnknownWrapper(IGraphicsUnknown* graphicsUnknown) : m_graphicsUnknown(graphicsUnknown) {}

    HRESULT __stdcall QueryInterface(const IID& iid, void** object) noexcept final
    {
        if (iid == __uuidof(IUnknown))
        {
            return RuntimeClass::QueryInterface(iid, object);
        }

        return m_graphicsUnknown->QueryInterface(iid, object);
    }

private:
    Microsoft::WRL::ComPtr<IGraphicsUnknown> m_graphicsUnknown;
};
#endif

// Convenience macro for functions that take an IUnknown* parameter and are typically
// called with a D3D-typed ComPtr argument. Example: 
// m_winmlProvider->QueueReference(WRAP_GRAPHICS_UNKNOWN(resource).Get());
// The macro is a no-op for PC (arg is convertible to IUnknown), and it will wrap 
// any IGraphicsUnknown for _GAMING_XBOX platforms.
#ifdef _GAMING_XBOX
#define WRAP_GRAPHICS_UNKNOWN(expr) Microsoft::WRL::Make<GraphicsUnknownWrapper>(expr)
#else
#define WRAP_GRAPHICS_UNKNOWN(expr) (expr)
#endif

// IID_GRAPHICS_PPV_ARGS is to IGraphicsUnknown as IID_PPV_ARGS is to IUnknown.
// There is no IGraphicsUnknown in stock D3D, so this is the same as IID_PPV_ARGS.
// This macro should be used anywhere an interface is represented in both D3D12.x and
// D3D12.
#ifndef _GAMING_XBOX
#define IID_GRAPHICS_PPV_ARGS IID_PPV_ARGS
#endif

// APIs like ID3D12DeviceChild::GetDevice return void in D3D12.x, but return HRESULT in D3D12.
#ifdef _GAMING_XBOX
#define GRAPHICS_THROW_IF_FAILED(hr) (hr)
#else
#define GRAPHICS_THROW_IF_FAILED(hr) ORT_THROW_IF_FAILED(hr)
#endif
} // namespace Dml