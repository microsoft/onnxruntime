//
//  Copyright (c) Microsoft Corporation.  All rights reserved.
//

#include "pch.h"
#include "inc/ImageConverter.h"
#include "inc/ImageConversionHelpers.h"
#include "inc/D3DDeviceCache.h"

using namespace Microsoft::WRL;
using namespace Windows::Graphics::DirectX::Direct3D11;
using namespace Windows::AI::MachineLearning::Internal;
using namespace winrt::Windows::AI::MachineLearning::implementation;
using namespace winrt::Windows::Media;
using namespace winrt::Windows::Graphics::Imaging;
using namespace winrt::Windows::Graphics::DirectX;
using namespace winrt::Windows::Graphics::DirectX::Direct3D11;

HRESULT ImageConverter::SyncD3D11ToD3D12(_In_ D3DDeviceCache& device_cache, _In_ ID3D11Texture2D* pD3D11Texture) {
  assert(pD3D11Texture != nullptr);

  ComPtr<ID3D11Device> spTextureDevice;
  pD3D11Texture->GetDevice(&spTextureDevice);

  if (spTextureDevice.Get() == device_cache.GetD3D11Device()) {
    // If the texture is on D3DDeviceCache's device, we sync using D3DDeviceCache's fences
    device_cache.GPUSyncD3D11ToD3D12();
  } else {
    // Otherwise, sync using our own cached fences
    ComPtr<ID3D11Fence> spD3D11DeviceFence;
    THROW_IF_FAILED(FetchOrCreateFenceOnDevice(device_cache, spTextureDevice.Get(), &spD3D11DeviceFence));

    device_cache.SyncD3D11DeviceToConverter(spD3D11DeviceFence.Get());
  }

  return S_OK;
}

HRESULT ImageConverter::SyncD3D12ToD3D11(_In_ D3DDeviceCache& device_cache, _In_ ID3D11Texture2D* spTexture) {
  assert(spTexture != nullptr);

  ComPtr<ID3D11Device> spTextureDevice;
  spTexture->GetDevice(&spTextureDevice);

  if (spTextureDevice.Get() == device_cache.GetD3D11Device()) {
    // If the texture is on D3DDeviceCache's device, we sync using D3DDeviceCache's fences
    device_cache.GPUSyncD3D12ToD3D11();
  } else {
    // Otherwise, sync using our own cached fences
    ComPtr<ID3D11Fence> spD3D11DeviceFence;
    THROW_IF_FAILED(FetchOrCreateFenceOnDevice(device_cache, spTextureDevice.Get(), &spD3D11DeviceFence));

    device_cache.SyncConverterToD3D11Device(spD3D11DeviceFence.Get());
  }

  return S_OK;
}

HRESULT ImageConverter::FetchOrCreateFenceOnDevice(_In_ D3DDeviceCache& device_cache, _In_ ID3D11Device* pD3D11Device, _Out_ ID3D11Fence** ppFence) {
  assert(pD3D11Device != nullptr);
  assert(ppFence != nullptr);

  UINT comPtrSize = static_cast<UINT>(sizeof(ppFence));

  if (FAILED(pD3D11Device->GetPrivateData(device_cache.GetFenceGuid(), &comPtrSize, ppFence)) || *ppFence == nullptr) {
    // There's no fence on the device, so create a new one
    ComPtr<ID3D11Device5> spD3D11Device5;
    THROW_IF_FAILED(pD3D11Device->QueryInterface(IID_PPV_ARGS(&spD3D11Device5)));
    THROW_IF_FAILED(spD3D11Device5->OpenSharedFence(device_cache.GetConverterFenceHandle(), IID_PPV_ARGS(ppFence)));

    // Store the fence on the device
    THROW_IF_FAILED(spD3D11Device5->SetPrivateDataInterface(device_cache.GetFenceGuid(), *ppFence));
  }

  return S_OK;
}

HRESULT ImageConverter::ResetCommandList(_In_ D3DDeviceCache& device_cache) {
  if (!command_list_) {
    assert(command_allocator_ == nullptr);

    THROW_IF_FAILED(device_cache.GetD3D12Device()->CreateCommandAllocator(
        device_cache.GetCommandQueue()->GetDesc().Type,
        IID_PPV_ARGS(command_allocator_.ReleaseAndGetAddressOf())));

    THROW_IF_FAILED(device_cache.GetD3D12Device()->CreateCommandList(
        0,
        device_cache.GetCommandQueue()->GetDesc().Type,
        command_allocator_.Get(),
        pipeline_state_.Get(),
        IID_PPV_ARGS(command_list_.ReleaseAndGetAddressOf())));
  } else {
    command_list_->Reset(command_allocator_.Get(), pipeline_state_.Get());
  }

  return S_OK;
}

HRESULT ImageConverter::ResetAllocator() {
  return command_allocator_->Reset();
}

HRESULT ImageConverter::CreateTextureFromUnsupportedColorFormat(
    _In_ const IVideoFrame& videoFrame,
    _In_ const BitmapBounds& inputBounds,
    _In_ const BitmapBounds& outputBounds,
    _In_ DirectXPixelFormat newFormat,
    _Out_ ID3D11Texture2D** ppTexture) {
  assert(videoFrame != nullptr);

  // Make sure we create the new video frame on the same device. We don't want the VideoFrame pipeline to implicitly share the texture between
  // 2 devices since we will need to do it ourselves anyway.
  IDirect3DDevice device;
  WINML_THROW_IF_FAILED(ImageConversionHelpers::GetDeviceFromDirect3DSurface(videoFrame.Direct3DSurface(), device));

  VideoFrame spNewVideoFrame = VideoFrame::CreateAsDirect3D11SurfaceBacked(newFormat, outputBounds.Width, outputBounds.Height, device);
  videoFrame.as<IVideoFrame2>().CopyToAsync(spNewVideoFrame, inputBounds, outputBounds).get();

  auto spDxgiInterfaceAccess = spNewVideoFrame.Direct3DSurface().as<IDirect3DDxgiInterfaceAccess>();
  THROW_IF_FAILED(spDxgiInterfaceAccess->GetInterface(IID_PPV_ARGS(ppTexture)));

  return S_OK;
}

HRESULT ImageConverter::CopyTextureIntoTexture(_In_ ID3D11Texture2D* pTextureFrom, _In_ const BitmapBounds& inputBounds, _Inout_ ID3D11Texture2D* pTextureTo) {
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
    D3D11_BOX cropBox = CD3D11_BOX(inputBounds.X, inputBounds.Y, 0, inputBounds.X + inputBounds.Width, inputBounds.Y + inputBounds.Height, 1);
    spDeviceContext->CopySubresourceRegion(pTextureTo, 0, 0, 0, 0, pTextureFrom, 0, &cropBox);
  } else {
    // Use the faster CopyResource() function since both textures have the same dimensions
    spDeviceContext->CopyResource(pTextureTo, pTextureFrom);
  }

  return S_OK;
}

HRESULT ImageConverter::ShareD3D11Texture(_In_ ID3D11Texture2D* pTexture, _In_ ID3D12Device* pDevice, _Outptr_ ID3D12Resource** ppResource) {
  assert(pTexture != nullptr);
  assert(pDevice != nullptr);

  ComPtr<IDXGIResource1> spDxgiResource;
  THROW_IF_FAILED(pTexture->QueryInterface(IID_PPV_ARGS(&spDxgiResource)));

  HANDLE hSharedTexture;
  THROW_IF_FAILED(spDxgiResource->CreateSharedHandle(nullptr, GENERIC_ALL, nullptr, &hSharedTexture));

  wil::unique_handle safe(hSharedTexture);
  THROW_IF_FAILED(pDevice->OpenSharedHandle(safe.get(), IID_PPV_ARGS(ppResource)));

  _sharedHandle = safe.get();

  return S_OK;
}