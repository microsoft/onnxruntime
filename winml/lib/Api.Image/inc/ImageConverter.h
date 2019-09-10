//
//  Copyright (c) Microsoft Corporation.  All rights reserved.
//

#pragma once

#include <wrl/implements.h>
#include "WinML_Lock.h"
#include "ImageConversionHelpers.h"

// Assign a name to the object to aid with debugging.
#if defined(_DEBUG)
inline void SetName(ID3D12Object* pObject, LPCWSTR name)
{
    pObject->SetName(name);
}
inline void SetNameIndexed(ID3D12Object* pObject, LPCWSTR name, UINT index)
{
    WCHAR fullName[50];
    if (swprintf_s(fullName, L"%s[%u]", name, index) > 0)
    {
        pObject->SetName(fullName);
    }
}
#else
inline void SetName(ID3D12Object*, LPCWSTR)
{
}
inline void SetNameIndexed(ID3D12Object*, LPCWSTR, UINT)
{
}
#endif

// Forward declaration
namespace winrt::Windows::AI::MachineLearning::implementation
{
    class D3DDeviceCache;
}

namespace Windows::AI::MachineLearning::Internal
{
    struct ConstantBufferCS
    {
        UINT Height;
        UINT Width;
    };

    class ImageConverter
    {
    public:
        ImageConverter() :
            _spConvertedVideoFrame(nullptr),
            _sharedHandle(nullptr) {}
        HRESULT ResetAllocator();

    protected:
        // Indices of shader resources in the descriptor heap.
        enum DescriptorHeapIndex : UINT32
        {
            SrvBufferIdx = 0,
            UavBufferIdx = SrvBufferIdx + 1,
            DescriptorCount = UavBufferIdx + 1
        };

        HANDLE _sharedHandle;
        Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList> _spCommandList;
        Microsoft::WRL::ComPtr<ID3D12CommandAllocator> _spCommandAllocator;
        Microsoft::WRL::ComPtr<ID3D12RootSignature> _spRootSignature;
        Microsoft::WRL::ComPtr<ID3D12PipelineState> _spPipelineState;
        Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> _spDescriptorHeap;
        Microsoft::WRL::ComPtr<ID3D11Texture2D> _spD3D11CachedTexture;
        winrt::Windows::Media::VideoFrame _spConvertedVideoFrame;
        CWinML_Lock _Lock;

        HRESULT SyncD3D11ToD3D12(_In_ winrt::Windows::AI::MachineLearning::implementation::D3DDeviceCache& deviceCache, _In_ ID3D11Texture2D* pD3D11Texture);
        HRESULT SyncD3D12ToD3D11(_In_ winrt::Windows::AI::MachineLearning::implementation::D3DDeviceCache& deviceCache, _In_ ID3D11Texture2D* spTexture);
        HRESULT ResetCommandList(_In_ winrt::Windows::AI::MachineLearning::implementation::D3DDeviceCache& deviceCache);
        HRESULT FetchOrCreateFenceOnDevice(_In_ winrt::Windows::AI::MachineLearning::implementation::D3DDeviceCache& deviceCache, _In_ ID3D11Device* pD3D11Device, _Out_ ID3D11Fence** ppFence);
        HRESULT ShareD3D11Texture(_In_ ID3D11Texture2D* pTexture, _In_ ID3D12Device* pDevice, _Outptr_ ID3D12Resource** ppResource);

        HRESULT CreateTextureFromUnsupportedColorFormat(
            _In_ const winrt::Windows::Media::IVideoFrame& videoFrame,
            _In_ const winrt::Windows::Graphics::Imaging::BitmapBounds& inputBounds,
            _In_ const winrt::Windows::Graphics::Imaging::BitmapBounds& outputBounds,
            _In_ winrt::Windows::Graphics::DirectX::DirectXPixelFormat newFormat,
            _Out_ ID3D11Texture2D** ppTexture
        );

        static HRESULT CopyTextureIntoTexture(
            _In_ ID3D11Texture2D* pTextureFrom,
            _In_ const winrt::Windows::Graphics::Imaging::BitmapBounds& inputBounds,
            _Inout_ ID3D11Texture2D* pTextureTo
        );
    };
}