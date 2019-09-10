//
//  Copyright (c) Microsoft Corporation.  All rights reserved.
//

#pragma once

#include <D3d11_4.h>
#include "ImageConversionTypes.h"

namespace Windows::AI::MachineLearning::Internal
{
    class ImageConversionHelpers
    {
    public:
        // This API that takes a video frame and converts it to a video frame of desired format (DXGI_FORMAT_B8G8R8X8_UNORM/BitmapPixelFormat::Bgra8) and size (after any scale/crop operations).
        // This should also cover any DX adapter hop (if needed in a multi GPU scenario) and CPU->GPU / GPU->CPU conversion 
        static HRESULT ConvertVideoFrameToVideoFrame(
            _In_ const winrt::Windows::Media::IVideoFrame& inputVideoFrame,
            _In_ const winrt::Windows::Graphics::Imaging::BitmapBounds& inputBounds,
            _In_ UINT32 outputWidth,
            _In_ UINT32 outputHeight,
            _Inout_ winrt::Windows::Media::VideoFrame& pOutputVideoFrame
        );

        // This helper method uses the input parameters do determine if a conversion is necessary 
        // A conversion is not necessary if 
        // 1. input bounds cover the entire input bitmap/surface
        // 2. desired output size is equal to input size
        // 3. (mapping softwarebitmap to softwarebitmap) OR (mapping from d3dsurface to d3dsurface AND the two surfaces are on the same device)
        // 4. the input is already in the desired format (BGRA8/B8G8R8X8UIntNormalized)
        static bool NeedsVideoFrameConversion(
            _In_ const winrt::Windows::Media::IVideoFrame& inputVideoFrame,
            _In_ LUID outputLuid,
            _In_ const winrt::Windows::Graphics::Imaging::BitmapBounds& inputBounds,
            _In_ UINT32 outputWidth,
            _In_ UINT32 outputHeight
        );

        static bool SoftwareBitmapFormatSupported(_In_ const winrt::Windows::Graphics::Imaging::SoftwareBitmap& softwareBitmap);
        static bool DirectXPixelFormatSupported(_In_ winrt::Windows::Graphics::DirectX::DirectXPixelFormat format);
        static bool FormatSupportedForUAV(_In_ ID3D12Device1* device, _In_ DXGI_FORMAT format);
        static IMG_TENSOR_CHANNEL_TYPE GetChannelTypeFromSoftwareBitmap(_In_ const winrt::Windows::Graphics::Imaging::SoftwareBitmap& softwareBitmap);
        static IMG_TENSOR_CHANNEL_TYPE GetChannelTypeFromDirect3DSurface(_In_ const winrt::Windows::Graphics::DirectX::Direct3D11::IDirect3DSurface& direct3DSurface);
        static winrt::Windows::Graphics::Imaging::BitmapPixelFormat GetBitmapPixelFormatFromChannelType(_In_ IMG_TENSOR_CHANNEL_TYPE channelType);
        static winrt::Windows::Graphics::DirectX::DirectXPixelFormat GetDirectXPixelFormatFromDXGIFormat(_In_ DXGI_FORMAT dxgiFormat);
        static DXGI_FORMAT GetDXGIFormatFromDirectXPixelFormat(_In_ winrt::Windows::Graphics::DirectX::DirectXPixelFormat directXPixelFormat);
        static winrt::Windows::Graphics::DirectX::DirectXPixelFormat GetDirectXPixelFormatFromChannelType(_In_ IMG_TENSOR_CHANNEL_TYPE channelType);
        static HRESULT GetLUIDFromDirect3DSurface(_In_ const winrt::Windows::Graphics::DirectX::Direct3D11::IDirect3DSurface& surface, _Out_ LUID &luid);
        static HRESULT GetTextureFromDirect3DSurface(_In_ const winrt::Windows::Graphics::DirectX::Direct3D11::IDirect3DSurface& d3dSurface, ID3D11Texture2D** ppD3D11Device);
        static bool TexturesHaveSameDevice(_In_ ID3D11Texture2D* pTexture1, _In_ ID3D11Texture2D* pTexture2);
        static bool TextureIsOnDevice(_In_ ID3D11Texture2D* pTexture, _In_ ID3D11Device* pDevice);
        static bool VideoFramesHaveSameDimensions(_In_ const winrt::Windows::Media::IVideoFrame& videoFrame1, _In_ const winrt::Windows::Media::IVideoFrame& videoFrame2);
        static bool VideoFramesHaveSameDevice(_In_ const winrt::Windows::Media::IVideoFrame& videoFrame1, _In_ const winrt::Windows::Media::IVideoFrame& videoFrame2);

        static HRESULT GetDeviceFromDirect3DSurface(
            _In_ const winrt::Windows::Graphics::DirectX::Direct3D11::IDirect3DSurface& d3dSurface,
            _Inout_ winrt::Windows::Graphics::DirectX::Direct3D11::IDirect3DDevice& d3dDevice
        );

        static constexpr std::array<DXGI_FORMAT, 3> supportedWinMLFormats = {
            DXGI_FORMAT_R8G8B8A8_UNORM,
            DXGI_FORMAT_B8G8R8A8_UNORM,
            DXGI_FORMAT_B8G8R8X8_UNORM
        };

    private:
        static HRESULT GetVideoFrameInfo(
            _In_ const winrt::Windows::Media::IVideoFrame& inputVideoFrame,
            _Out_ DWORD& format,
            _Out_ int& width,
            _Out_ int& height,
            _Out_ LUID& luid
        );
    };
}