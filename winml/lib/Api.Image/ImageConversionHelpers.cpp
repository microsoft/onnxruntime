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
  _Out_ LUID& luid
) {
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
  _Inout_ wm::VideoFrame& pOutputVideoFrame
) {
  wgi::BitmapBounds outputBounds = {0, 0, outputWidth, outputHeight};

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
  _In_ UINT32 outputHeight
) {
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
  } else if (luid.HighPart != outputLuid.HighPart || luid.LowPart != outputLuid.LowPart) {
    bNeedConversion = true;
  } else if (static_cast<uint32_t>(width) != outputWidth || static_cast<uint32_t>(height) != outputHeight) {
    bNeedConversion = true;
  } else if (outputLuid.HighPart != 0 || outputLuid.LowPart != 0) {
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
    TraceLoggingInt32(inputBounds.Height, "rH")
  );

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
  const wgdx::Direct3D11::IDirect3DSurface& direct3DSurface
) {
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
    case DXGI_FORMAT_B8G8R8A8_UNORM:
      return wgdx::DirectXPixelFormat::B8G8R8A8UIntNormalized;
    case DXGI_FORMAT_B8G8R8X8_UNORM:
      return wgdx::DirectXPixelFormat::B8G8R8X8UIntNormalized;
    case DXGI_FORMAT_R8G8B8A8_UNORM:
      return wgdx::DirectXPixelFormat::R8G8B8A8UIntNormalized;
    case DXGI_FORMAT_R8_UNORM:
      return wgdx::DirectXPixelFormat::R8UIntNormalized;
  }

  WINML_THROW_HR(E_INVALIDARG);
}

DXGI_FORMAT _winmli::GetDXGIFormatFromDirectXPixelFormat(_In_ wgdx::DirectXPixelFormat directXPixelFormat) {
  switch (directXPixelFormat) {
    case wgdx::DirectXPixelFormat::B8G8R8A8UIntNormalized:
      return DXGI_FORMAT_B8G8R8A8_UNORM;
    case wgdx::DirectXPixelFormat::B8G8R8X8UIntNormalized:
      return DXGI_FORMAT_B8G8R8X8_UNORM;
    case wgdx::DirectXPixelFormat::R8G8B8A8UIntNormalized:
      return DXGI_FORMAT_R8G8B8A8_UNORM;
    case wgdx::DirectXPixelFormat::R8UIntNormalized:
      return DXGI_FORMAT_R8_UNORM;
  }

  WINML_THROW_HR(E_INVALIDARG);
}

wgdx::DirectXPixelFormat _winmli::GetDirectXPixelFormatFromChannelType(_In_ _winml::ImageTensorChannelType channelType
) {
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

wgdx::Direct3D11::IDirect3DDevice _winmli::GetDeviceFromDirect3DSurface(
  const wgdx::Direct3D11::IDirect3DSurface& d3dSurface
) {
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
    winrt::guid_of<wgdx::Direct3D11::IDirect3DDevice>(), reinterpret_cast<void**>(winrt::put_abi(d3dDevice))
  ));

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
