// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <D3d11_4.h>
#include "ImageConversionTypes.h"

namespace _winml::Imaging {

// This API that takes a video frame and converts it to a video frame of desired format (DXGI_FORMAT_B8G8R8X8_UNORM/BitmapPixelFormat::Bgra8) and size (after any scale/crop operations).
// This should also cover any DX adapter hop (if needed in a multi GPU scenario) and CPU->GPU / GPU->CPU conversion
void ConvertVideoFrameToVideoFrame(
  _In_ const wm::IVideoFrame& input_video_frame,
  _In_ const wgi::BitmapBounds& input_bounds,
  _In_ UINT32 output_width,
  _In_ UINT32 output_height,
  _Inout_ wm::VideoFrame& output_video_frame
);

// This helper method uses the input parameters do determine if a conversion is necessary
// A conversion is not necessary if
// 1. input bounds cover the entire input bitmap/surface
// 2. desired output size is equal to input size
// 3. (mapping softwarebitmap to softwarebitmap) OR (mapping from d3dsurface to d3dsurface AND the two surfaces are on the same device)
// 4. the input is already in the desired format (BGRA8/B8G8R8X8UIntNormalized)
bool NeedsVideoFrameConversion(
  _In_ const wm::IVideoFrame& input_video_frame,
  _In_ LUID output_luid,
  _In_ const wgi::BitmapBounds& input_bounds,
  _In_ UINT32 output_width,
  _In_ UINT32 output_height
);

bool SoftwareBitmapFormatSupported(const wgi::SoftwareBitmap& software_bitmap);
bool DirectXPixelFormatSupported(wgdx::DirectXPixelFormat format);
bool FormatSupportedForUAV(_In_ ID3D12Device1* device, _In_ DXGI_FORMAT format);
ImageTensorChannelType GetChannelTypeFromSoftwareBitmap(const wgi::SoftwareBitmap& software_bitmap);
ImageTensorChannelType GetChannelTypeFromDirect3DSurface(const wgdx::Direct3D11::IDirect3DSurface& direct3D_surface);
wgi::BitmapPixelFormat GetBitmapPixelFormatFromChannelType(ImageTensorChannelType channel_type);
wgdx::DirectXPixelFormat GetDirectXPixelFormatFromDXGIFormat(DXGI_FORMAT dxgi_format);
DXGI_FORMAT GetDXGIFormatFromDirectXPixelFormat(_In_ wgdx::DirectXPixelFormat directX_pixel_format);
wgdx::DirectXPixelFormat GetDirectXPixelFormatFromChannelType(_In_ ImageTensorChannelType channel_type);
Microsoft::WRL::ComPtr<ID3D11Texture2D> GetTextureFromDirect3DSurface(
  const wgdx::Direct3D11::IDirect3DSurface& d3d_surface
);
bool TexturesHaveSameDevice(_In_ ID3D11Texture2D* pTexture1, _In_ ID3D11Texture2D* texture2d);
bool TextureIsOnDevice(_In_ ID3D11Texture2D* pTexture, _In_ ID3D11Device* device);
bool VideoFramesHaveSameDimensions(const wm::IVideoFrame& video_frame_1, const wm::IVideoFrame& video_frame_2);
bool VideoFramesHaveSameDevice(const wm::IVideoFrame& video_frame_1, const wm::IVideoFrame& video_frame_2);

wgdx::Direct3D11::IDirect3DDevice GetDeviceFromDirect3DSurface(const wgdx::Direct3D11::IDirect3DSurface& d3dSurface);

constexpr std::array<DXGI_FORMAT, 3> supportedWinMLFormats = {
  DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_FORMAT_B8G8R8A8_UNORM, DXGI_FORMAT_B8G8R8X8_UNORM
};
}  // namespace _winml::Imaging
