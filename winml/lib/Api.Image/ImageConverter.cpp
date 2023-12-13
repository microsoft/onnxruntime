// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "lib/Api.Image/pch.h"
#include "inc/ImageConverter.h"
#include "inc/ImageConversionHelpers.h"
#include "inc/D3DDeviceCache.h"

using namespace Microsoft::WRL;

using namespace _winml;

void ImageConverter::SyncD3D11ToD3D12(_In_ D3DDeviceCache& device_cache, _In_ ID3D11Texture2D* pD3D11Texture) {
  assert(pD3D11Texture != nullptr);

  ComPtr<ID3D11Device> spTextureDevice;
  pD3D11Texture->GetDevice(&spTextureDevice);

  if (spTextureDevice.Get() == device_cache.GetD3D11Device()) {
    // If the texture is on D3DDeviceCache's device, we sync using D3DDeviceCache's fences
    device_cache.GPUSyncD3D11ToD3D12();
  } else {
    // Otherwise, sync using our own cached fences
    ComPtr<ID3D11Fence> spD3D11DeviceFence = FetchOrCreateFenceOnDevice(device_cache, spTextureDevice.Get());
    device_cache.SyncD3D11DeviceToConverter(spD3D11DeviceFence.Get());
  }
}

void ImageConverter::SyncD3D12ToD3D11(_In_ D3DDeviceCache& device_cache, _In_ ID3D11Texture2D* spTexture) {
  assert(spTexture != nullptr);

  ComPtr<ID3D11Device> spTextureDevice;
  spTexture->GetDevice(&spTextureDevice);

  if (spTextureDevice.Get() == device_cache.GetD3D11Device()) {
    // If the texture is on D3DDeviceCache's device, we sync using D3DDeviceCache's fences
    device_cache.GPUSyncD3D12ToD3D11();
  } else {
    // Otherwise, sync using our own cached fences
    ComPtr<ID3D11Fence> spD3D11DeviceFence = FetchOrCreateFenceOnDevice(device_cache, spTextureDevice.Get());
    device_cache.SyncConverterToD3D11Device(spD3D11DeviceFence.Get());
  }
}

ComPtr<ID3D11Fence> ImageConverter::FetchOrCreateFenceOnDevice(
  _In_ D3DDeviceCache& device_cache, _In_ ID3D11Device* pD3D11Device
) {
  assert(pD3D11Device != nullptr);

  ComPtr<ID3D11Fence> fence;
  UINT comPtrSize = static_cast<UINT>(sizeof(fence.GetAddressOf()));

  if (FAILED(pD3D11Device->GetPrivateData(device_cache.GetFenceGuid(), &comPtrSize, fence.GetAddressOf())) || fence.Get() == nullptr) {
    // There's no fence on the device, so create a new one
    ComPtr<ID3D11Device5> spD3D11Device5;
    WINML_THROW_IF_FAILED(pD3D11Device->QueryInterface(IID_PPV_ARGS(&spD3D11Device5)));
    WINML_THROW_IF_FAILED(spD3D11Device5->OpenSharedFence(device_cache.GetConverterFenceHandle(), IID_PPV_ARGS(&fence))
    );

    // Store the fence on the device
    WINML_THROW_IF_FAILED(spD3D11Device5->SetPrivateDataInterface(device_cache.GetFenceGuid(), fence.Get()));
  }

  return fence;
}

void ImageConverter::ResetCommandList(_In_ D3DDeviceCache& device_cache) {
  if (!command_list_) {
    assert(command_allocator_ == nullptr);

    WINML_THROW_IF_FAILED(device_cache.GetD3D12Device()->CreateCommandAllocator(
      device_cache.GetCommandQueue()->GetDesc().Type, IID_PPV_ARGS(command_allocator_.ReleaseAndGetAddressOf())
    ));

    WINML_THROW_IF_FAILED(device_cache.GetD3D12Device()->CreateCommandList(
      0,
      device_cache.GetCommandQueue()->GetDesc().Type,
      command_allocator_.Get(),
      pipeline_state_.Get(),
      IID_PPV_ARGS(command_list_.ReleaseAndGetAddressOf())
    ));
  } else {
    command_list_->Reset(command_allocator_.Get(), pipeline_state_.Get());
  }
}

void ImageConverter::ResetAllocator() {
  WINML_THROW_IF_FAILED(command_allocator_->Reset());
}

ComPtr<ID3D11Texture2D> ImageConverter::CreateTextureFromUnsupportedColorFormat(
  const wm::IVideoFrame& videoFrame,
  const wgi::BitmapBounds& inputBounds,
  const wgi::BitmapBounds& outputBounds,
  wgdx::DirectXPixelFormat newFormat
) {
  assert(videoFrame != nullptr);

  // Make sure we create the new video frame on the same device. We don't want the VideoFrame pipeline to implicitly share the texture between
  // 2 devices since we will need to do it ourselves anyway.
  auto device = _winmli::GetDeviceFromDirect3DSurface(videoFrame.Direct3DSurface());

  auto spNewVideoFrame =
    wm::VideoFrame::CreateAsDirect3D11SurfaceBacked(newFormat, outputBounds.Width, outputBounds.Height, device);
  videoFrame.as<wm::IVideoFrame2>().CopyToAsync(spNewVideoFrame, inputBounds, outputBounds).get();

  using namespace Windows::Graphics::DirectX::Direct3D11;

  auto spDxgiInterfaceAccess = spNewVideoFrame.Direct3DSurface().as<IDirect3DDxgiInterfaceAccess>();
  ComPtr<ID3D11Texture2D> d3d11Texture;
  WINML_THROW_IF_FAILED(spDxgiInterfaceAccess->GetInterface(IID_PPV_ARGS(&d3d11Texture)));

  return d3d11Texture;
}

void ImageConverter::CopyTextureIntoTexture(
  _In_ ID3D11Texture2D* pTextureFrom, _In_ const wgi::BitmapBounds& inputBounds, _Inout_ ID3D11Texture2D* pTextureTo
) {
  assert(pTextureFrom != nullptr);
  assert(pTextureTo != nullptr);

  D3D11_TEXTURE2D_DESC textureFromDesc, textureToDesc;
  pTextureFrom->GetDesc(&textureFromDesc);
  pTextureTo->GetDesc(&textureToDesc);

  assert(inputBounds.Width <= textureFromDesc.Width && inputBounds.Width <= textureToDesc.Width);
  assert(inputBounds.Height <= textureFromDesc.Height && inputBounds.Height <= textureToDesc.Height);

  ComPtr<ID3D11Device> spDeviceFrom, spDeviceTo;
  pTextureFrom->GetDevice(&spDeviceFrom);
  pTextureTo->GetDevice(&spDeviceTo);

  assert(spDeviceFrom.Get() == spDeviceTo.Get());

  ComPtr<ID3D11DeviceContext> spDeviceContext;
  spDeviceFrom->GetImmediateContext(&spDeviceContext);

  if (textureFromDesc.Width != textureToDesc.Width || textureFromDesc.Height != textureToDesc.Height) {
    // We can't copy the whole resource, so we have to use the slower CopySubresource() function
    D3D11_BOX cropBox = CD3D11_BOX(
      inputBounds.X, inputBounds.Y, 0, inputBounds.X + inputBounds.Width, inputBounds.Y + inputBounds.Height, 1
    );
    spDeviceContext->CopySubresourceRegion(pTextureTo, 0, 0, 0, 0, pTextureFrom, 0, &cropBox);
  } else {
    // Use the faster CopyResource() function since both textures have the same dimensions
    spDeviceContext->CopyResource(pTextureTo, pTextureFrom);
  }
}
