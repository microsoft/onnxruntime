// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <D3d11_4.h>
#include "ImageConversionTypes.h"

namespace Windows::AI::MachineLearning::Internal::ImageConversionHelpers {
  // This API that takes a video frame and converts it to a video frame of desired format (DXGI_FORMAT_B8G8R8X8_UNORM/BitmapPixelFormat::Bgra8) and size (after any scale/crop operations).
  // This should also cover any DX adapter hop (if needed in a multi GPU scenario) and CPU->GPU / GPU->CPU conversion
  void ConvertVideoFrameToVideoFrame(
      _In_ const winrt::Windows::Media::IVideoFrame& input_video_frame,
      _In_ const winrt::Windows::Graphics::Imaging::BitmapBounds& input_bounds,
      _In_ UINT32 output_width,
      _In_ UINT32 output_height,
      _Inout_ winrt::Windows::Media::VideoFrame& output_video_frame);

  // This helper method uses the input parameters do determine if a conversion is necessary
  // A conversion is not necessary if
  // 1. input bounds cover the entire input bitmap/surface
  // 2. desired output size is equal to input size
  // 3. (mapping softwarebitmap to softwarebitmap) OR (mapping from d3dsurface to d3dsurface AND the two surfaces are on the same device)
  // 4. the input is already in the desired format (BGRA8/B8G8R8X8UIntNormalized)
  bool NeedsVideoFrameConversion(
      _In_ const winrt::Windows::Media::IVideoFrame& input_video_frame,
      _In_ LUID output_luid,
      _In_ const winrt::Windows::Graphics::Imaging::BitmapBounds& input_bounds,
      _In_ UINT32 output_width,
      _In_ UINT32 output_height);

  bool SoftwareBitmapFormatSupported(const winrt::Windows::Graphics::Imaging::SoftwareBitmap& software_bitmap);
  bool DirectXPixelFormatSupported(winrt::Windows::Graphics::DirectX::DirectXPixelFormat format);
  bool FormatSupportedForUAV(_In_ ID3D12Device1* device, _In_ DXGI_FORMAT format);
  ImageTensorChannelType GetChannelTypeFromSoftwareBitmap(const winrt::Windows::Graphics::Imaging::SoftwareBitmap& software_bitmap);
  ImageTensorChannelType GetChannelTypeFromDirect3DSurface(const winrt::Windows::Graphics::DirectX::Direct3D11::IDirect3DSurface& direct3D_surface);
  winrt::Windows::Graphics::Imaging::BitmapPixelFormat GetBitmapPixelFormatFromChannelType(ImageTensorChannelType channel_type);
  winrt::Windows::Graphics::DirectX::DirectXPixelFormat GetDirectXPixelFormatFromDXGIFormat(DXGI_FORMAT dxgi_format);
  DXGI_FORMAT GetDXGIFormatFromDirectXPixelFormat(_In_ winrt::Windows::Graphics::DirectX::DirectXPixelFormat directX_pixel_format);
  winrt::Windows::Graphics::DirectX::DirectXPixelFormat GetDirectXPixelFormatFromChannelType(_In_ ImageTensorChannelType channel_type);
  Microsoft::WRL::ComPtr<ID3D11Texture2D> GetTextureFromDirect3DSurface(const winrt::Windows::Graphics::DirectX::Direct3D11::IDirect3DSurface& d3d_surface);
  bool TexturesHaveSameDevice(_In_ ID3D11Texture2D* pTexture1, _In_ ID3D11Texture2D* texture2d);
  bool TextureIsOnDevice(_In_ ID3D11Texture2D* pTexture, _In_ ID3D11Device* device);
  bool VideoFramesHaveSameDimensions(const winrt::Windows::Media::IVideoFrame& video_frame_1, const winrt::Windows::Media::IVideoFrame& video_frame_2);
  bool VideoFramesHaveSameDevice(const winrt::Windows::Media::IVideoFrame& video_frame_1, const winrt::Windows::Media::IVideoFrame& video_frame_2);

  winrt::Windows::Graphics::DirectX::Direct3D11::IDirect3DDevice GetDeviceFromDirect3DSurface(
      const winrt::Windows::Graphics::DirectX::Direct3D11::IDirect3DSurface& d3dSurface);

  constexpr std::array<DXGI_FORMAT, 3> supportedWinMLFormats = {
      DXGI_FORMAT_R8G8B8A8_UNORM,
      DXGI_FORMAT_B8G8R8A8_UNORM,
      DXGI_FORMAT_B8G8R8X8_UNORM};
}  // namespace Windows::AI::MachineLearning::Internal::ImageConversionHelpers