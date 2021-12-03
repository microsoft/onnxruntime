// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "lib/Api.Image/pch.h"
#include "inc/ImageConversionHelpers.h"

#include <winrt/Windows.Graphics.DirectX.Direct3D11.h>

using namespace Microsoft::WRL;
using namespace Windows::Graphics::DirectX::Direct3D11;

static LUID GetLUIDFromDirect3DSurface(const wgdx::Direct3D11::IDirect3DSurface& surface) {
  ComPtr<ID3D11Device> spDx11Device;
  ComPtr<IDirect3DDxgiInterfaceAccess> spDxgiInterfaceAccess;
  ComPtr<ID3D11Texture2D> spDx11Texture2D;
  ComPtr<IDXGIDevice> spDXGIDevice;
  ComPtr<IDXGIAdapter> spDXGIAdapter;
  DXGI_ADAPTER_DESC adapterDesc = {0};

  spDxgiInterfaceAccess = surface.as<IDirect3DDxgiInterfaceAccess>().get();
  WINML_THROW_IF_FAILED(spDxgiInterfaceAccess->GetInterface(IID_PPV_ARGS(&spDx11Texture2D)));
  spDx11Texture2D->GetDevice(&spDx11Device);
  WINML_THROW_IF_FAILED(spDx11Device->QueryInterface(IID_PPV_ARGS(&spDXGIDevice)));
  WINML_THROW_IF_FAILED(spDXGIDevice->GetAdapter(&spDXGIAdapter));
  WINML_THROW_IF_FAILED(spDXGIAdapter->GetDesc(&adapterDesc));

  return adapterDesc.AdapterLuid;
}

static HRESULT GetVideoFrameInfo(
    _In_ const wm::IVideoFrame& inputVideoFrame,
    _Out_ DWORD& format,
    _Out_ int& width,
    _Out_ int& height,
    _Out_ LUID& luid) {
  wgdx::Direct3D11::IDirect3DSurface spInputSurface = inputVideoFrame.Direct3DSurface();
  if (spInputSurface != nullptr) {
    wgdx::Direct3D11::Direct3DSurfaceDescription description;
    description = spInputSurface.Description();
    format = (DWORD)description.Format;
    width = description.Width;
    height = description.Height;
    luid = GetLUIDFromDirect3DSurface(spInputSurface);
  } else {
    wgi::SoftwareBitmap spInputSoftwareBitmap = inputVideoFrame.SoftwareBitmap();
    if (spInputSoftwareBitmap != nullptr) {
      format = (DWORD)spInputSoftwareBitmap.BitmapPixelFormat();
      height = spInputSoftwareBitmap.PixelHeight();
      width = spInputSoftwareBitmap.PixelWidth();
      luid.HighPart = luid.LowPart = 0;
    } else {
      return E_INVALIDARG;
    }
  }
  return S_OK;
}

void _winmli::ConvertVideoFrameToVideoFrame(
    _In_ const wm::IVideoFrame& inputVideoFrame,
    _In_ const wgi::BitmapBounds& inputBounds,
    _In_ UINT32 outputWidth,
    _In_ UINT32 outputHeight,
    _Inout_ wm::VideoFrame& pOutputVideoFrame) {
  wgi::BitmapBounds outputBounds = {
      0,
      0,
      outputWidth,
      outputHeight};

  wgi::SoftwareBitmap spInputSoftwareBitmap = inputVideoFrame.SoftwareBitmap();
  wgdx::Direct3D11::IDirect3DSurface spInputDirect3DSurface = inputVideoFrame.Direct3DSurface();

  // only one of softwarebitmap or direct3Dsurface should be non-null
  if ((spInputSoftwareBitmap == nullptr && spInputDirect3DSurface == nullptr) || (spInputSoftwareBitmap != nullptr && spInputDirect3DSurface != nullptr)) {
    WINML_THROW_HR(E_INVALIDARG);
  }

  auto pInputVideoFrame2 = inputVideoFrame.as<wm::IVideoFrame2>();
  pInputVideoFrame2.CopyToAsync(pOutputVideoFrame, inputBounds, outputBounds).get();
}

bool _winmli::SoftwareBitmapFormatSupported(const wgi::SoftwareBitmap& softwareBitmap) {
  assert(softwareBitmap != nullptr);

  switch (softwareBitmap.BitmapPixelFormat()) {
    case wgi::BitmapPixelFormat::Bgra8:
    case wgi::BitmapPixelFormat::Rgba8:
    case wgi::BitmapPixelFormat::Gray8:
      return true;
  }

  return false;
}

bool _winmli::DirectXPixelFormatSupported(wgdx::DirectXPixelFormat format) {
  switch (format) {
    case wgdx::DirectXPixelFormat::B8G8R8X8UIntNormalized:
    case wgdx::DirectXPixelFormat::B8G8R8A8UIntNormalized:
    case wgdx::DirectXPixelFormat::R8G8B8A8UIntNormalized:
    case wgdx::DirectXPixelFormat::R8UIntNormalized:
      return true;
  }

  return false;
}

bool _winmli::FormatSupportedForUAV(_In_ ID3D12Device1* device, _In_ DXGI_FORMAT format) {
  assert(device != nullptr);

  D3D12_FEATURE_DATA_FORMAT_SUPPORT formatSupport = {format};
  HRESULT hr = device->CheckFeatureSupport(D3D12_FEATURE_FORMAT_SUPPORT, &formatSupport, sizeof(formatSupport));

  return SUCCEEDED(hr) && (formatSupport.Support1 & D3D12_FORMAT_SUPPORT1_TYPED_UNORDERED_ACCESS_VIEW);
}

// This helper method uses the input parameters do determine if a conversion is necessary
// A conversion is not necessary if
// 1. input bounds cover the entire input bitmap/surface (else we are cropping)
// 2. desired output size is equal to input size (else we are resizing)
// 3. (mapping softwarebitmap to softwarebitmap) OR (mapping from d3dsurface to d3dsurface AND the two surfaces are on the same device)
// 4. the input is already in the desired format (BGRA8/B8G8R8X8UIntNormalized)
bool _winmli::NeedsVideoFrameConversion(
    _In_ const wm::IVideoFrame& inputVideoFrame,
    _In_ LUID outputLuid,
    _In_ const wgi::BitmapBounds& inputBounds,
    _In_ UINT32 outputWidth,
    _In_ UINT32 outputHeight) {
  bool bNeedConversion = false;
  HRESULT hr = S_OK;

  DWORD format = 0;
  int width = 0, height = 0;
  LUID luid;

  if (FAILED((hr = GetVideoFrameInfo(inputVideoFrame, format, width, height, luid)))) {
    bNeedConversion = true;
  } else if (((int)inputBounds.Width != outputWidth) ||
            (inputBounds.X != 0) ||
            ((int)inputBounds.Height != outputHeight) ||
            (inputBounds.Y != 0) ||
            (inputVideoFrame == nullptr))  // Check crop
  {
    bNeedConversion = true;
  } else if (luid.HighPart != outputLuid.HighPart ||
            luid.LowPart != outputLuid.LowPart) {
    bNeedConversion = true;
  } else if (static_cast<uint32_t>(width) != outputWidth ||
            static_cast<uint32_t>(height) != outputHeight) {
    bNeedConversion = true;
  } else if (outputLuid.HighPart != 0 ||
            outputLuid.LowPart != 0) {
    if (format != (DWORD)wgdx::DirectXPixelFormat::B8G8R8X8UIntNormalized) {
      bNeedConversion = true;
    }
  } else {
    if (format != (DWORD)wgi::BitmapPixelFormat::Bgra8) {
      bNeedConversion = true;
    }
  }

  TraceLoggingWrite(
      winml_trace_logging_provider,
      "InputVideoFrame",
      TraceLoggingKeyword(WINML_PROVIDER_KEYWORD_DEFAULT),
      TraceLoggingBool(bNeedConversion, "Convert"),
      TraceLoggingHexInt32(hr, "HRESULT"),
      TraceLoggingInt32(width, "iWidth"),
      TraceLoggingInt32(outputWidth, "oWidth"),
      TraceLoggingInt32(height, "iHeight"),
      TraceLoggingInt32(outputWidth, "oHeight"),
      TraceLoggingHexInt64(*((ULONGLONG*)&luid), "iLuid"),
      TraceLoggingHexInt64(*((ULONGLONG*)&outputLuid), "oLuid"),
      TraceLoggingHexInt32(format, "iFormat"),
      TraceLoggingInt32(inputBounds.X, "rX"),
      TraceLoggingInt32(inputBounds.Y, "rY"),
      TraceLoggingInt32(inputBounds.Width, "rW"),
      TraceLoggingInt32(inputBounds.Height, "rH"));

  return bNeedConversion;
}

_winml::ImageTensorChannelType _winmli::GetChannelTypeFromSoftwareBitmap(const wgi::SoftwareBitmap& softwareBitmap) {
  assert(softwareBitmap != nullptr);

  switch (softwareBitmap.BitmapPixelFormat()) {
    case wgi::BitmapPixelFormat::Bgra8:
      return _winml::kImageTensorChannelTypeBGR8;
    case wgi::BitmapPixelFormat::Rgba8:
      return _winml::kImageTensorChannelTypeRGB8;
    case wgi::BitmapPixelFormat::Gray8:
      return _winml::kImageTensorChannelTypeGRAY8;
  }

  WINML_THROW_HR(E_INVALIDARG);
}

wgi::BitmapPixelFormat _winmli::GetBitmapPixelFormatFromChannelType(_winml::ImageTensorChannelType channelType) {
  switch (channelType) {
    case _winml::kImageTensorChannelTypeBGR8:
      return wgi::BitmapPixelFormat::Bgra8;
    case _winml::kImageTensorChannelTypeRGB8:
      return wgi::BitmapPixelFormat::Rgba8;
    case _winml::kImageTensorChannelTypeGRAY8:
      return wgi::BitmapPixelFormat::Gray8;
  }

  WINML_THROW_HR(E_INVALIDARG);
}

_winml::ImageTensorChannelType _winmli::GetChannelTypeFromDirect3DSurface(
    const wgdx::Direct3D11::IDirect3DSurface& direct3DSurface) {
  assert(direct3DSurface != nullptr);

  switch (direct3DSurface.Description().Format) {
    case wgdx::DirectXPixelFormat::B8G8R8A8UIntNormalized:
    case wgdx::DirectXPixelFormat::B8G8R8X8UIntNormalized:
      return _winml::kImageTensorChannelTypeBGR8;

    case wgdx::DirectXPixelFormat::R8G8B8A8UIntNormalized:
      return _winml::kImageTensorChannelTypeRGB8;

    case wgdx::DirectXPixelFormat::R8UIntNormalized:
      return _winml::kImageTensorChannelTypeGRAY8;
  }

  WINML_THROW_HR(E_INVALIDARG);
}

wgdx::DirectXPixelFormat _winmli::GetDirectXPixelFormatFromDXGIFormat(DXGI_FORMAT dxgiFormat) {
  switch (dxgiFormat) {
    case DXGI_FORMAT_A8P8: return wgdx::DirectXPixelFormat::A8P8;
    case DXGI_FORMAT_A8_UNORM: return wgdx::DirectXPixelFormat::A8UIntNormalized;
    case DXGI_FORMAT_AI44: return wgdx::DirectXPixelFormat::AI44;
    case DXGI_FORMAT_AYUV: return wgdx::DirectXPixelFormat::Ayuv;
    case DXGI_FORMAT_B4G4R4A4_UNORM: return wgdx::DirectXPixelFormat::B4G4R4A4UIntNormalized;
    case DXGI_FORMAT_B5G5R5A1_UNORM: return wgdx::DirectXPixelFormat::B5G5R5A1UIntNormalized;
    case DXGI_FORMAT_B5G6R5_UNORM: return wgdx::DirectXPixelFormat::B5G6R5UIntNormalized;
    case DXGI_FORMAT_B8G8R8A8_TYPELESS: return wgdx::DirectXPixelFormat::B8G8R8A8Typeless;
    case DXGI_FORMAT_B8G8R8A8_UNORM: return wgdx::DirectXPixelFormat::B8G8R8A8UIntNormalized;
    case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB: return wgdx::DirectXPixelFormat::B8G8R8A8UIntNormalizedSrgb;
    case DXGI_FORMAT_B8G8R8X8_TYPELESS: return wgdx::DirectXPixelFormat::B8G8R8X8Typeless;
    case DXGI_FORMAT_B8G8R8X8_UNORM: return wgdx::DirectXPixelFormat::B8G8R8X8UIntNormalized;
    case DXGI_FORMAT_B8G8R8X8_UNORM_SRGB: return wgdx::DirectXPixelFormat::B8G8R8X8UIntNormalizedSrgb;
    case DXGI_FORMAT_BC1_TYPELESS: return wgdx::DirectXPixelFormat::BC1Typeless;
    case DXGI_FORMAT_BC1_UNORM: return wgdx::DirectXPixelFormat::BC1UIntNormalized;
    case DXGI_FORMAT_BC1_UNORM_SRGB: return wgdx::DirectXPixelFormat::BC1UIntNormalizedSrgb;
    case DXGI_FORMAT_BC2_TYPELESS: return wgdx::DirectXPixelFormat::BC2Typeless;
    case DXGI_FORMAT_BC2_UNORM: return wgdx::DirectXPixelFormat::BC2UIntNormalized;
    case DXGI_FORMAT_BC2_UNORM_SRGB: return wgdx::DirectXPixelFormat::BC2UIntNormalizedSrgb;
    case DXGI_FORMAT_BC3_TYPELESS: return wgdx::DirectXPixelFormat::BC3Typeless;
    case DXGI_FORMAT_BC3_UNORM: return wgdx::DirectXPixelFormat::BC3UIntNormalized;
    case DXGI_FORMAT_BC3_UNORM_SRGB: return wgdx::DirectXPixelFormat::BC3UIntNormalizedSrgb;
    case DXGI_FORMAT_BC4_SNORM: return wgdx::DirectXPixelFormat::BC4IntNormalized;
    case DXGI_FORMAT_BC4_TYPELESS: return wgdx::DirectXPixelFormat::BC4Typeless;
    case DXGI_FORMAT_BC4_UNORM: return wgdx::DirectXPixelFormat::BC4UIntNormalized;
    case DXGI_FORMAT_BC5_SNORM: return wgdx::DirectXPixelFormat::BC5IntNormalized;
    case DXGI_FORMAT_BC5_TYPELESS: return wgdx::DirectXPixelFormat::BC5Typeless;
    case DXGI_FORMAT_BC5_UNORM: return wgdx::DirectXPixelFormat::BC5UIntNormalized;
    case DXGI_FORMAT_BC6H_SF16: return wgdx::DirectXPixelFormat::BC6H16Float;
    case DXGI_FORMAT_BC6H_UF16: return wgdx::DirectXPixelFormat::BC6H16UnsignedFloat;
    case DXGI_FORMAT_BC6H_TYPELESS: return wgdx::DirectXPixelFormat::BC6HTypeless;
    case DXGI_FORMAT_BC7_TYPELESS: return wgdx::DirectXPixelFormat::BC7Typeless;
    case DXGI_FORMAT_BC7_UNORM: return wgdx::DirectXPixelFormat::BC7UIntNormalized;
    case DXGI_FORMAT_BC7_UNORM_SRGB: return wgdx::DirectXPixelFormat::BC7UIntNormalizedSrgb;
    case DXGI_FORMAT_D16_UNORM: return wgdx::DirectXPixelFormat::D16UIntNormalized;
    case DXGI_FORMAT_D24_UNORM_S8_UINT: return wgdx::DirectXPixelFormat::D24UIntNormalizedS8UInt;
    case DXGI_FORMAT_D32_FLOAT: return wgdx::DirectXPixelFormat::D32Float;
    case DXGI_FORMAT_D32_FLOAT_S8X24_UINT: return wgdx::DirectXPixelFormat::D32FloatS8X24UInt;
    case DXGI_FORMAT_G8R8_G8B8_UNORM: return wgdx::DirectXPixelFormat::G8R8G8B8UIntNormalized;
    case DXGI_FORMAT_IA44: return wgdx::DirectXPixelFormat::IA44;
    case DXGI_FORMAT_NV11: return wgdx::DirectXPixelFormat::NV11;
    case DXGI_FORMAT_NV12: return wgdx::DirectXPixelFormat::NV12;
    case DXGI_FORMAT_420_OPAQUE: return wgdx::DirectXPixelFormat::Opaque420;
    case DXGI_FORMAT_P010: return wgdx::DirectXPixelFormat::P010;
    case DXGI_FORMAT_P016: return wgdx::DirectXPixelFormat::P016;
    case DXGI_FORMAT_P208: return wgdx::DirectXPixelFormat::P208;
    case DXGI_FORMAT_P8: return wgdx::DirectXPixelFormat::P8;
    case DXGI_FORMAT_R10G10B10A2_TYPELESS: return wgdx::DirectXPixelFormat::R10G10B10A2Typeless;
    case DXGI_FORMAT_R10G10B10A2_UINT: return wgdx::DirectXPixelFormat::R10G10B10A2UInt;
    case DXGI_FORMAT_R10G10B10A2_UNORM: return wgdx::DirectXPixelFormat::R10G10B10A2UIntNormalized;
    case DXGI_FORMAT_R10G10B10_XR_BIAS_A2_UNORM: return wgdx::DirectXPixelFormat::R10G10B10XRBiasA2UIntNormalized;
    case DXGI_FORMAT_R11G11B10_FLOAT: return wgdx::DirectXPixelFormat::R11G11B10Float;
    case DXGI_FORMAT_R16_FLOAT: return wgdx::DirectXPixelFormat::R16Float;
    case DXGI_FORMAT_R16G16B16A16_FLOAT: return wgdx::DirectXPixelFormat::R16G16B16A16Float;
    case DXGI_FORMAT_R16G16B16A16_SINT: return wgdx::DirectXPixelFormat::R16G16B16A16Int;
    case DXGI_FORMAT_R16G16B16A16_SNORM: return wgdx::DirectXPixelFormat::R16G16B16A16IntNormalized;
    case DXGI_FORMAT_R16G16B16A16_TYPELESS: return wgdx::DirectXPixelFormat::R16G16B16A16Typeless;
    case DXGI_FORMAT_R16G16B16A16_UINT: return wgdx::DirectXPixelFormat::R16G16B16A16UInt;
    case DXGI_FORMAT_R16G16B16A16_UNORM: return wgdx::DirectXPixelFormat::R16G16B16A16UIntNormalized;
    case DXGI_FORMAT_R16G16_FLOAT: return wgdx::DirectXPixelFormat::R16G16Float;
    case DXGI_FORMAT_R16G16_SINT: return wgdx::DirectXPixelFormat::R16G16Int;
    case DXGI_FORMAT_R16G16_SNORM: return wgdx::DirectXPixelFormat::R16G16IntNormalized;
    case DXGI_FORMAT_R16G16_TYPELESS: return wgdx::DirectXPixelFormat::R16G16Typeless;
    case DXGI_FORMAT_R16G16_UINT: return wgdx::DirectXPixelFormat::R16G16UInt;
    case DXGI_FORMAT_R16G16_UNORM: return wgdx::DirectXPixelFormat::R16G16UIntNormalized;
    case DXGI_FORMAT_R16_SINT: return wgdx::DirectXPixelFormat::R16Int;
    case DXGI_FORMAT_R16_SNORM: return wgdx::DirectXPixelFormat::R16IntNormalized;
    case DXGI_FORMAT_R16_TYPELESS: return wgdx::DirectXPixelFormat::R16Typeless;
    case DXGI_FORMAT_R16_UINT: return wgdx::DirectXPixelFormat::R16UInt;
    case DXGI_FORMAT_R16_UNORM: return wgdx::DirectXPixelFormat::R16UIntNormalized;
    case DXGI_FORMAT_R1_UNORM: return wgdx::DirectXPixelFormat::R1UIntNormalized;
    case DXGI_FORMAT_R24G8_TYPELESS: return wgdx::DirectXPixelFormat::R24G8Typeless;
    case DXGI_FORMAT_R24_UNORM_X8_TYPELESS: return wgdx::DirectXPixelFormat::R24UIntNormalizedX8Typeless;
    case DXGI_FORMAT_R32_FLOAT: return wgdx::DirectXPixelFormat::R32Float;
    case DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS: return wgdx::DirectXPixelFormat::R32FloatX8X24Typeless;
    case DXGI_FORMAT_R32G32B32A32_FLOAT: return wgdx::DirectXPixelFormat::R32G32B32A32Float;
    case DXGI_FORMAT_R32G32B32A32_SINT: return wgdx::DirectXPixelFormat::R32G32B32A32Int;
    case DXGI_FORMAT_R32G32B32A32_TYPELESS: return wgdx::DirectXPixelFormat::R32G32B32A32Typeless;
    case DXGI_FORMAT_R32G32B32A32_UINT: return wgdx::DirectXPixelFormat::R32G32B32A32UInt;
    case DXGI_FORMAT_R32G32B32_FLOAT: return wgdx::DirectXPixelFormat::R32G32B32Float;
    case DXGI_FORMAT_R32G32B32_SINT: return wgdx::DirectXPixelFormat::R32G32B32Int;
    case DXGI_FORMAT_R32G32B32_TYPELESS: return wgdx::DirectXPixelFormat::R32G32B32Typeless;
    case DXGI_FORMAT_R32G32B32_UINT: return wgdx::DirectXPixelFormat::R32G32B32UInt;
    case DXGI_FORMAT_R32G32_FLOAT: return wgdx::DirectXPixelFormat::R32G32Float;
    case DXGI_FORMAT_R32G32_SINT: return wgdx::DirectXPixelFormat::R32G32Int;
    case DXGI_FORMAT_R32G32_TYPELESS: return wgdx::DirectXPixelFormat::R32G32Typeless;
    case DXGI_FORMAT_R32G32_UINT: return wgdx::DirectXPixelFormat::R32G32UInt;
    case DXGI_FORMAT_R32G8X24_TYPELESS: return wgdx::DirectXPixelFormat::R32G8X24Typeless;
    case DXGI_FORMAT_R32_SINT: return wgdx::DirectXPixelFormat::R32Int;
    case DXGI_FORMAT_R32_TYPELESS: return wgdx::DirectXPixelFormat::R32Typeless;
    case DXGI_FORMAT_R32_UINT: return wgdx::DirectXPixelFormat::R32UInt;
    case DXGI_FORMAT_R8G8B8A8_SINT: return wgdx::DirectXPixelFormat::R8G8B8A8Int;
    case DXGI_FORMAT_R8G8B8A8_SNORM: return wgdx::DirectXPixelFormat::R8G8B8A8IntNormalized;
    case DXGI_FORMAT_R8G8B8A8_TYPELESS: return wgdx::DirectXPixelFormat::R8G8B8A8Typeless;
    case DXGI_FORMAT_R8G8B8A8_UINT: return wgdx::DirectXPixelFormat::R8G8B8A8UInt;
    case DXGI_FORMAT_R8G8B8A8_UNORM: return wgdx::DirectXPixelFormat::R8G8B8A8UIntNormalized;
    case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB: return wgdx::DirectXPixelFormat::R8G8B8A8UIntNormalizedSrgb;
    case DXGI_FORMAT_R8G8_B8G8_UNORM: return wgdx::DirectXPixelFormat::R8G8B8G8UIntNormalized;
    case DXGI_FORMAT_R8G8_SINT: return wgdx::DirectXPixelFormat::R8G8Int;
    case DXGI_FORMAT_R8G8_SNORM: return wgdx::DirectXPixelFormat::R8G8IntNormalized;
    case DXGI_FORMAT_R8G8_TYPELESS: return wgdx::DirectXPixelFormat::R8G8Typeless;
    case DXGI_FORMAT_R8G8_UINT: return wgdx::DirectXPixelFormat::R8G8UInt;
    case DXGI_FORMAT_R8G8_UNORM: return wgdx::DirectXPixelFormat::R8G8UIntNormalized;
    case DXGI_FORMAT_R8_SINT: return wgdx::DirectXPixelFormat::R8Int;
    case DXGI_FORMAT_R8_SNORM: return wgdx::DirectXPixelFormat::R8IntNormalized;
    case DXGI_FORMAT_R8_TYPELESS: return wgdx::DirectXPixelFormat::R8Typeless;
    case DXGI_FORMAT_R8_UINT: return wgdx::DirectXPixelFormat::R8UInt;
    case DXGI_FORMAT_R8_UNORM: return wgdx::DirectXPixelFormat::R8UIntNormalized;
    case DXGI_FORMAT_R9G9B9E5_SHAREDEXP: return wgdx::DirectXPixelFormat::R9G9B9E5SharedExponent;
    case DXGI_FORMAT_SAMPLER_FEEDBACK_MIN_MIP_OPAQUE: return wgdx::DirectXPixelFormat::SamplerFeedbackMinMipOpaque;
    case DXGI_FORMAT_SAMPLER_FEEDBACK_MIP_REGION_USED_OPAQUE: return wgdx::DirectXPixelFormat::SamplerFeedbackMipRegionUsedOpaque;
    case DXGI_FORMAT_UNKNOWN: return wgdx::DirectXPixelFormat::Unknown;
    case DXGI_FORMAT_V208: return wgdx::DirectXPixelFormat::V208;
    case DXGI_FORMAT_V408: return wgdx::DirectXPixelFormat::V408;
    case DXGI_FORMAT_X24_TYPELESS_G8_UINT: return wgdx::DirectXPixelFormat::X24TypelessG8UInt;
    case DXGI_FORMAT_X32_TYPELESS_G8X24_UINT: return wgdx::DirectXPixelFormat::X32TypelessG8X24UInt;
    case DXGI_FORMAT_Y210: return wgdx::DirectXPixelFormat::Y210;
    case DXGI_FORMAT_Y216: return wgdx::DirectXPixelFormat::Y216;
    case DXGI_FORMAT_Y410: return wgdx::DirectXPixelFormat::Y410;
    case DXGI_FORMAT_Y416: return wgdx::DirectXPixelFormat::Y416;
    case DXGI_FORMAT_YUY2: return wgdx::DirectXPixelFormat::Yuy2;
  }

  WINML_THROW_HR(E_INVALIDARG);
}

DXGI_FORMAT _winmli::GetDXGIFormatFromDirectXPixelFormat(_In_ wgdx::DirectXPixelFormat directXPixelFormat) {
  switch (directXPixelFormat) {
    case wgdx::DirectXPixelFormat::A8P8: return DXGI_FORMAT_A8P8;
    case wgdx::DirectXPixelFormat::A8UIntNormalized: return DXGI_FORMAT_A8_UNORM;
    case wgdx::DirectXPixelFormat::AI44: return DXGI_FORMAT_AI44;
    case wgdx::DirectXPixelFormat::Ayuv: return DXGI_FORMAT_AYUV;
    case wgdx::DirectXPixelFormat::B4G4R4A4UIntNormalized: return DXGI_FORMAT_B4G4R4A4_UNORM;
    case wgdx::DirectXPixelFormat::B5G5R5A1UIntNormalized: return DXGI_FORMAT_B5G5R5A1_UNORM;
    case wgdx::DirectXPixelFormat::B5G6R5UIntNormalized: return DXGI_FORMAT_B5G6R5_UNORM;
    case wgdx::DirectXPixelFormat::B8G8R8A8Typeless: return DXGI_FORMAT_B8G8R8A8_TYPELESS;
    case wgdx::DirectXPixelFormat::B8G8R8A8UIntNormalized: return DXGI_FORMAT_B8G8R8A8_UNORM;
    case wgdx::DirectXPixelFormat::B8G8R8A8UIntNormalizedSrgb: return DXGI_FORMAT_B8G8R8A8_UNORM_SRGB;
    case wgdx::DirectXPixelFormat::B8G8R8X8Typeless: return DXGI_FORMAT_B8G8R8X8_TYPELESS;
    case wgdx::DirectXPixelFormat::B8G8R8X8UIntNormalized: return DXGI_FORMAT_B8G8R8X8_UNORM;
    case wgdx::DirectXPixelFormat::B8G8R8X8UIntNormalizedSrgb: return DXGI_FORMAT_B8G8R8X8_UNORM_SRGB;
    case wgdx::DirectXPixelFormat::BC1Typeless: return DXGI_FORMAT_BC1_TYPELESS;
    case wgdx::DirectXPixelFormat::BC1UIntNormalized: return DXGI_FORMAT_BC1_UNORM;
    case wgdx::DirectXPixelFormat::BC1UIntNormalizedSrgb: return DXGI_FORMAT_BC1_UNORM_SRGB;
    case wgdx::DirectXPixelFormat::BC2Typeless: return DXGI_FORMAT_BC2_TYPELESS;
    case wgdx::DirectXPixelFormat::BC2UIntNormalized: return DXGI_FORMAT_BC2_UNORM;
    case wgdx::DirectXPixelFormat::BC2UIntNormalizedSrgb: return DXGI_FORMAT_BC2_UNORM_SRGB;
    case wgdx::DirectXPixelFormat::BC3Typeless: return DXGI_FORMAT_BC3_TYPELESS;
    case wgdx::DirectXPixelFormat::BC3UIntNormalized: return DXGI_FORMAT_BC3_UNORM;
    case wgdx::DirectXPixelFormat::BC3UIntNormalizedSrgb: return DXGI_FORMAT_BC3_UNORM_SRGB;
    case wgdx::DirectXPixelFormat::BC4IntNormalized: return DXGI_FORMAT_BC4_SNORM;
    case wgdx::DirectXPixelFormat::BC4Typeless: return DXGI_FORMAT_BC4_TYPELESS;
    case wgdx::DirectXPixelFormat::BC4UIntNormalized: return DXGI_FORMAT_BC4_UNORM;
    case wgdx::DirectXPixelFormat::BC5IntNormalized: return DXGI_FORMAT_BC5_SNORM;
    case wgdx::DirectXPixelFormat::BC5Typeless: return DXGI_FORMAT_BC5_TYPELESS;
    case wgdx::DirectXPixelFormat::BC5UIntNormalized: return DXGI_FORMAT_BC5_UNORM;
    case wgdx::DirectXPixelFormat::BC6H16Float: return DXGI_FORMAT_BC6H_SF16;
    case wgdx::DirectXPixelFormat::BC6H16UnsignedFloat: return DXGI_FORMAT_BC6H_UF16;
    case wgdx::DirectXPixelFormat::BC6HTypeless: return DXGI_FORMAT_BC6H_TYPELESS;
    case wgdx::DirectXPixelFormat::BC7Typeless: return DXGI_FORMAT_BC7_TYPELESS;
    case wgdx::DirectXPixelFormat::BC7UIntNormalized: return DXGI_FORMAT_BC7_UNORM;
    case wgdx::DirectXPixelFormat::BC7UIntNormalizedSrgb: return DXGI_FORMAT_BC7_UNORM_SRGB;
    case wgdx::DirectXPixelFormat::D16UIntNormalized: return DXGI_FORMAT_D16_UNORM;
    case wgdx::DirectXPixelFormat::D24UIntNormalizedS8UInt: return DXGI_FORMAT_D24_UNORM_S8_UINT;
    case wgdx::DirectXPixelFormat::D32Float: return DXGI_FORMAT_D32_FLOAT;
    case wgdx::DirectXPixelFormat::D32FloatS8X24UInt: return DXGI_FORMAT_D32_FLOAT_S8X24_UINT;
    case wgdx::DirectXPixelFormat::G8R8G8B8UIntNormalized: return DXGI_FORMAT_G8R8_G8B8_UNORM;
    case wgdx::DirectXPixelFormat::IA44: return DXGI_FORMAT_IA44;
    case wgdx::DirectXPixelFormat::NV11: return DXGI_FORMAT_NV11;
    case wgdx::DirectXPixelFormat::NV12: return DXGI_FORMAT_NV12;
    case wgdx::DirectXPixelFormat::Opaque420: return DXGI_FORMAT_420_OPAQUE;
    case wgdx::DirectXPixelFormat::P010: return DXGI_FORMAT_P010;
    case wgdx::DirectXPixelFormat::P016: return DXGI_FORMAT_P016;
    case wgdx::DirectXPixelFormat::P208: return DXGI_FORMAT_P208;
    case wgdx::DirectXPixelFormat::P8: return DXGI_FORMAT_P8;
    case wgdx::DirectXPixelFormat::R10G10B10A2Typeless: return DXGI_FORMAT_R10G10B10A2_TYPELESS;
    case wgdx::DirectXPixelFormat::R10G10B10A2UInt: return DXGI_FORMAT_R10G10B10A2_UINT;
    case wgdx::DirectXPixelFormat::R10G10B10A2UIntNormalized: return DXGI_FORMAT_R10G10B10A2_UNORM;
    case wgdx::DirectXPixelFormat::R10G10B10XRBiasA2UIntNormalized: return DXGI_FORMAT_R10G10B10_XR_BIAS_A2_UNORM;
    case wgdx::DirectXPixelFormat::R11G11B10Float: return DXGI_FORMAT_R11G11B10_FLOAT;
    case wgdx::DirectXPixelFormat::R16Float: return DXGI_FORMAT_R16_FLOAT;
    case wgdx::DirectXPixelFormat::R16G16B16A16Float: return DXGI_FORMAT_R16G16B16A16_FLOAT;
    case wgdx::DirectXPixelFormat::R16G16B16A16Int: return DXGI_FORMAT_R16G16B16A16_SINT;
    case wgdx::DirectXPixelFormat::R16G16B16A16IntNormalized: return DXGI_FORMAT_R16G16B16A16_SNORM;
    case wgdx::DirectXPixelFormat::R16G16B16A16Typeless: return DXGI_FORMAT_R16G16B16A16_TYPELESS;
    case wgdx::DirectXPixelFormat::R16G16B16A16UInt: return DXGI_FORMAT_R16G16B16A16_UINT;
    case wgdx::DirectXPixelFormat::R16G16B16A16UIntNormalized: return DXGI_FORMAT_R16G16B16A16_UNORM;
    case wgdx::DirectXPixelFormat::R16G16Float: return DXGI_FORMAT_R16G16_FLOAT;
    case wgdx::DirectXPixelFormat::R16G16Int: return DXGI_FORMAT_R16G16_SINT;
    case wgdx::DirectXPixelFormat::R16G16IntNormalized: return DXGI_FORMAT_R16G16_SNORM;
    case wgdx::DirectXPixelFormat::R16G16Typeless: return DXGI_FORMAT_R16G16_TYPELESS;
    case wgdx::DirectXPixelFormat::R16G16UInt: return DXGI_FORMAT_R16G16_UINT;
    case wgdx::DirectXPixelFormat::R16G16UIntNormalized: return DXGI_FORMAT_R16G16_UNORM;
    case wgdx::DirectXPixelFormat::R16Int: return DXGI_FORMAT_R16_SINT;
    case wgdx::DirectXPixelFormat::R16IntNormalized: return DXGI_FORMAT_R16_SNORM;
    case wgdx::DirectXPixelFormat::R16Typeless: return DXGI_FORMAT_R16_TYPELESS;
    case wgdx::DirectXPixelFormat::R16UInt: return DXGI_FORMAT_R16_UINT;
    case wgdx::DirectXPixelFormat::R16UIntNormalized: return DXGI_FORMAT_R16_UNORM;
    case wgdx::DirectXPixelFormat::R1UIntNormalized: return DXGI_FORMAT_R1_UNORM;
    case wgdx::DirectXPixelFormat::R24G8Typeless: return DXGI_FORMAT_R24G8_TYPELESS;
    case wgdx::DirectXPixelFormat::R24UIntNormalizedX8Typeless: return DXGI_FORMAT_R24_UNORM_X8_TYPELESS;
    case wgdx::DirectXPixelFormat::R32Float: return DXGI_FORMAT_R32_FLOAT;
    case wgdx::DirectXPixelFormat::R32FloatX8X24Typeless: return DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS;
    case wgdx::DirectXPixelFormat::R32G32B32A32Float: return DXGI_FORMAT_R32G32B32A32_FLOAT;
    case wgdx::DirectXPixelFormat::R32G32B32A32Int: return DXGI_FORMAT_R32G32B32A32_SINT;
    case wgdx::DirectXPixelFormat::R32G32B32A32Typeless: return DXGI_FORMAT_R32G32B32A32_TYPELESS;
    case wgdx::DirectXPixelFormat::R32G32B32A32UInt: return DXGI_FORMAT_R32G32B32A32_UINT;
    case wgdx::DirectXPixelFormat::R32G32B32Float: return DXGI_FORMAT_R32G32B32_FLOAT;
    case wgdx::DirectXPixelFormat::R32G32B32Int: return DXGI_FORMAT_R32G32B32_SINT;
    case wgdx::DirectXPixelFormat::R32G32B32Typeless: return DXGI_FORMAT_R32G32B32_TYPELESS;
    case wgdx::DirectXPixelFormat::R32G32B32UInt: return DXGI_FORMAT_R32G32B32_UINT;
    case wgdx::DirectXPixelFormat::R32G32Float: return DXGI_FORMAT_R32G32_FLOAT;
    case wgdx::DirectXPixelFormat::R32G32Int: return DXGI_FORMAT_R32G32_SINT;
    case wgdx::DirectXPixelFormat::R32G32Typeless: return DXGI_FORMAT_R32G32_TYPELESS;
    case wgdx::DirectXPixelFormat::R32G32UInt: return DXGI_FORMAT_R32G32_UINT;
    case wgdx::DirectXPixelFormat::R32G8X24Typeless: return DXGI_FORMAT_R32G8X24_TYPELESS;
    case wgdx::DirectXPixelFormat::R32Int: return DXGI_FORMAT_R32_SINT;
    case wgdx::DirectXPixelFormat::R32Typeless: return DXGI_FORMAT_R32_TYPELESS;
    case wgdx::DirectXPixelFormat::R32UInt: return DXGI_FORMAT_R32_UINT;
    case wgdx::DirectXPixelFormat::R8G8B8A8Int: return DXGI_FORMAT_R8G8B8A8_SINT;
    case wgdx::DirectXPixelFormat::R8G8B8A8IntNormalized: return DXGI_FORMAT_R8G8B8A8_SNORM;
    case wgdx::DirectXPixelFormat::R8G8B8A8Typeless: return DXGI_FORMAT_R8G8B8A8_TYPELESS;
    case wgdx::DirectXPixelFormat::R8G8B8A8UInt: return DXGI_FORMAT_R8G8B8A8_UINT;
    case wgdx::DirectXPixelFormat::R8G8B8A8UIntNormalized: return DXGI_FORMAT_R8G8B8A8_UNORM;
    case wgdx::DirectXPixelFormat::R8G8B8A8UIntNormalizedSrgb: return DXGI_FORMAT_R8G8B8A8_UNORM_SRGB;
    case wgdx::DirectXPixelFormat::R8G8B8G8UIntNormalized: return DXGI_FORMAT_R8G8_B8G8_UNORM;
    case wgdx::DirectXPixelFormat::R8G8Int: return DXGI_FORMAT_R8G8_SINT;
    case wgdx::DirectXPixelFormat::R8G8IntNormalized: return DXGI_FORMAT_R8G8_SNORM;
    case wgdx::DirectXPixelFormat::R8G8Typeless: return DXGI_FORMAT_R8G8_TYPELESS;
    case wgdx::DirectXPixelFormat::R8G8UInt: return DXGI_FORMAT_R8G8_UINT;
    case wgdx::DirectXPixelFormat::R8G8UIntNormalized: return DXGI_FORMAT_R8G8_UNORM;
    case wgdx::DirectXPixelFormat::R8Int: return DXGI_FORMAT_R8_SINT;
    case wgdx::DirectXPixelFormat::R8IntNormalized: return DXGI_FORMAT_R8_SNORM;
    case wgdx::DirectXPixelFormat::R8Typeless: return DXGI_FORMAT_R8_TYPELESS;
    case wgdx::DirectXPixelFormat::R8UInt: return DXGI_FORMAT_R8_UINT;
    case wgdx::DirectXPixelFormat::R8UIntNormalized: return DXGI_FORMAT_R8_UNORM;
    case wgdx::DirectXPixelFormat::R9G9B9E5SharedExponent: return DXGI_FORMAT_R9G9B9E5_SHAREDEXP;
    case wgdx::DirectXPixelFormat::SamplerFeedbackMinMipOpaque: return DXGI_FORMAT_SAMPLER_FEEDBACK_MIN_MIP_OPAQUE;
    case wgdx::DirectXPixelFormat::SamplerFeedbackMipRegionUsedOpaque: return DXGI_FORMAT_SAMPLER_FEEDBACK_MIP_REGION_USED_OPAQUE;
    case wgdx::DirectXPixelFormat::Unknown: return DXGI_FORMAT_UNKNOWN;
    case wgdx::DirectXPixelFormat::V208: return DXGI_FORMAT_V208;
    case wgdx::DirectXPixelFormat::V408: return DXGI_FORMAT_V408;
    case wgdx::DirectXPixelFormat::X24TypelessG8UInt: return DXGI_FORMAT_X24_TYPELESS_G8_UINT;
    case wgdx::DirectXPixelFormat::X32TypelessG8X24UInt: return DXGI_FORMAT_X32_TYPELESS_G8X24_UINT;
    case wgdx::DirectXPixelFormat::Y210: return DXGI_FORMAT_Y210;
    case wgdx::DirectXPixelFormat::Y216: return DXGI_FORMAT_Y216;
    case wgdx::DirectXPixelFormat::Y410: return DXGI_FORMAT_Y410;
    case wgdx::DirectXPixelFormat::Y416: return DXGI_FORMAT_Y416;
    case wgdx::DirectXPixelFormat::Yuy2: return DXGI_FORMAT_YUY2;
  }

  WINML_THROW_HR(E_INVALIDARG);
}

wgdx::DirectXPixelFormat _winmli::GetDirectXPixelFormatFromChannelType(_In_ _winml::ImageTensorChannelType channelType) {
  switch (channelType) {
    case _winml::kImageTensorChannelTypeBGR8:
      return wgdx::DirectXPixelFormat::B8G8R8A8UIntNormalized;
    case _winml::kImageTensorChannelTypeRGB8:
      return wgdx::DirectXPixelFormat::R8G8B8A8UIntNormalized;
    case _winml::kImageTensorChannelTypeGRAY8:
      return wgdx::DirectXPixelFormat::R8UIntNormalized;
  }

  WINML_THROW_HR(E_INVALIDARG);
}

wgdx::Direct3D11::IDirect3DDevice _winmli::GetDeviceFromDirect3DSurface(const wgdx::Direct3D11::IDirect3DSurface& d3dSurface) {
  assert(d3dSurface != nullptr);

  ComPtr<ID3D11Texture2D> spDx11Texture2D;
  ComPtr<IDirect3DDxgiInterfaceAccess> spDxgiInterfaceAccess = d3dSurface.as<IDirect3DDxgiInterfaceAccess>().get();
  WINML_THROW_IF_FAILED(spDxgiInterfaceAccess->GetInterface(IID_PPV_ARGS(&spDx11Texture2D)));

  ComPtr<ID3D11Device> spDx11Device;
  spDx11Texture2D->GetDevice(&spDx11Device);

  ComPtr<IDXGIDevice> spDXGIDevice;
  WINML_THROW_IF_FAILED(spDx11Device->QueryInterface(IID_PPV_ARGS(&spDXGIDevice)));

  ComPtr<::IInspectable> spInspectable;
  WINML_THROW_IF_FAILED(CreateDirect3D11DeviceFromDXGIDevice(spDXGIDevice.Get(), &spInspectable));

  wgdx::Direct3D11::IDirect3DDevice d3dDevice;
  WINML_THROW_IF_FAILED(spInspectable->QueryInterface(
      winrt::guid_of<wgdx::Direct3D11::IDirect3DDevice>(),
      reinterpret_cast<void**>(winrt::put_abi(d3dDevice))));

  return d3dDevice;
}

bool _winmli::TexturesHaveSameDevice(_In_ ID3D11Texture2D* pTexture1, _In_ ID3D11Texture2D* pTexture2) {
  if (pTexture1 && pTexture2) {
    ComPtr<ID3D11Device> spDevice1;
    pTexture1->GetDevice(&spDevice1);

    ComPtr<ID3D11Device> spDevice2;
    pTexture2->GetDevice(&spDevice2);

    return spDevice1.Get() == spDevice2.Get();
  }

  return false;
}

bool _winmli::TextureIsOnDevice(_In_ ID3D11Texture2D* pTexture, _In_ ID3D11Device* pDevice) {
  if (pTexture && pDevice) {
    ComPtr<ID3D11Device> spDevice1;
    pTexture->GetDevice(&spDevice1);

    return spDevice1.Get() == pDevice;
  }

  return false;
}

ComPtr<ID3D11Texture2D> _winmli::GetTextureFromDirect3DSurface(const wgdx::Direct3D11::IDirect3DSurface& d3dSurface) {
  auto spDxgiInterfaceAccess = d3dSurface.as<IDirect3DDxgiInterfaceAccess>();
  ComPtr<ID3D11Texture2D> d3d11Texture;
  WINML_THROW_IF_FAILED(spDxgiInterfaceAccess->GetInterface(IID_PPV_ARGS(&d3d11Texture)));

  return d3d11Texture;
}

bool _winmli::VideoFramesHaveSameDimensions(const wm::IVideoFrame& videoFrame1, const wm::IVideoFrame& videoFrame2) {
  if (videoFrame1 && videoFrame2) {
    auto desc1 = videoFrame1.Direct3DSurface().Description();
    auto desc2 = videoFrame2.Direct3DSurface().Description();

    return desc1.Width == desc2.Width && desc1.Height == desc2.Height;
  }

  return false;
}

bool _winmli::VideoFramesHaveSameDevice(const wm::IVideoFrame& videoFrame1, const wm::IVideoFrame& videoFrame2) {
  if (videoFrame1 && videoFrame2) {
    ComPtr<ID3D11Texture2D> spTexture1 = _winmli::GetTextureFromDirect3DSurface(videoFrame1.Direct3DSurface());
    ComPtr<ID3D11Texture2D> spTexture2 = _winmli::GetTextureFromDirect3DSurface(videoFrame2.Direct3DSurface());

    ComPtr<ID3D11Device> spDevice1, spDevice2;
    spTexture1->GetDevice(&spDevice1);
    spTexture2->GetDevice(&spDevice2);

    return spDevice1.Get() == spDevice2.Get();
  }

  return false;
}