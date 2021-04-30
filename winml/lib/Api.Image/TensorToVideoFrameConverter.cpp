// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "pch.h"

#include <winmeta.h>  // winmeta needed for TraceLoggingKeyword
#include <TraceLoggingProvider.h>
#include <TraceloggingConfig.h>
#include <evntrace.h>
#include <MemoryBuffer.h>

#include "inc/D3DDeviceCache.h"
#include "inc/TensorToVideoFrameConverter.h"
#include "CpuDetensorizer.h"

#include "inc/ImageConversionHelpers.h"
#include "LearningModelDevice.h"
#include "EventTimer.h"

#include "robuffer.h"
#include "inc/DisjointBufferHelpers.h"

using namespace Microsoft::WRL;
using namespace Windows::Graphics::DirectX::Direct3D11;

using namespace _winml;

class GPUTensorToDX12TextureTelemetryEvent {
 public:
  GPUTensorToDX12TextureTelemetryEvent(const ImageTensorDescription& tensorDesc) {
    runtime_session_id_ = telemetry_helper.GetRuntimeSessionId();
    TraceLoggingWrite(
        winml_trace_logging_provider,
        "GPUTensorToDX12TextureStart",
        TraceLoggingKeyword(WINML_PROVIDER_KEYWORD_DEFAULT),
        TraceLoggingHexInt32(tensorDesc.channelType, "Type"),
        TraceLoggingInt64(tensorDesc.sizes[2], "Height"),
        TraceLoggingInt64(tensorDesc.sizes[3], "Width"),
        TraceLoggingInt32(runtime_session_id_, "runtimeSessionId"),
        TelemetryPrivacyDataTag(PDT_ProductAndServiceUsage),
        TraceLoggingBool(true, "UTCReplace_AppSessionGuid"),
        TraceLoggingKeyword(MICROSOFT_KEYWORD_MEASURES));
  }
  ~GPUTensorToDX12TextureTelemetryEvent() {
    TraceLoggingWrite(
        winml_trace_logging_provider,
        "GPUTensorToDX12TextureStop",
        TraceLoggingKeyword(WINML_PROVIDER_KEYWORD_DEFAULT),
        TraceLoggingHexInt32(S_OK, "HRESULT"),
        TraceLoggingInt32(runtime_session_id_, "runtimeSessionId"),
        TelemetryPrivacyDataTag(PDT_ProductAndServiceUsage),
        TraceLoggingBool(true, "UTCReplace_AppSessionGuid"),
        TraceLoggingKeyword(MICROSOFT_KEYWORD_MEASURES));
  }

private:
  int runtime_session_id_;
};

class ConvertGPUTensorToSoftwareBitmapTelemetryEvent {
 public:
  ConvertGPUTensorToSoftwareBitmapTelemetryEvent(const ImageTensorDescription& tensorDesc) {
    runtime_session_id_ = telemetry_helper.GetRuntimeSessionId();
    TraceLoggingWrite(
        winml_trace_logging_provider,
        "ConvertGPUTensorToSoftwareBitmapStart",
        TraceLoggingKeyword(WINML_PROVIDER_KEYWORD_DEFAULT),
        TraceLoggingHexInt32(tensorDesc.channelType, "Type"),
        TraceLoggingInt64(tensorDesc.sizes[2], "Height"),
        TraceLoggingInt64(tensorDesc.sizes[3], "Width"),
        TraceLoggingInt32(runtime_session_id_, "runtimeSessionId"),
        TelemetryPrivacyDataTag(PDT_ProductAndServiceUsage),
        TraceLoggingBool(true, "UTCReplace_AppSessionGuid"),
        TraceLoggingKeyword(MICROSOFT_KEYWORD_MEASURES));
  }
  ~ConvertGPUTensorToSoftwareBitmapTelemetryEvent() {
    TraceLoggingWrite(
        winml_trace_logging_provider,
        "ConvertGPUTensorToSoftwareBitmapStop",
        TraceLoggingKeyword(WINML_PROVIDER_KEYWORD_DEFAULT),
        TraceLoggingHexInt32(S_OK, "HRESULT"),
        TraceLoggingInt32(runtime_session_id_, "runtimeSessionId"),
        TelemetryPrivacyDataTag(PDT_ProductAndServiceUsage),
        TraceLoggingBool(true, "UTCReplace_AppSessionGuid"),
        TraceLoggingKeyword(MICROSOFT_KEYWORD_MEASURES));
  }

private:
  int runtime_session_id_;
};

class ConvertCPUTensorToVideoFrameWithSoftwareBitmapTelemetryEvent {
 public:
  ConvertCPUTensorToVideoFrameWithSoftwareBitmapTelemetryEvent(const ImageTensorDescription& tensorDesc) {
    runtime_session_id_ = telemetry_helper.GetRuntimeSessionId();
    TraceLoggingWrite(
        winml_trace_logging_provider,
        "ConvertCPUTensorToVideoFrameWithSoftwareBitmapStart",
        TraceLoggingKeyword(WINML_PROVIDER_KEYWORD_DEFAULT),
        TraceLoggingHexInt32(tensorDesc.channelType, "Type"),
        TraceLoggingInt64(tensorDesc.sizes[2], "Height"),
        TraceLoggingInt64(tensorDesc.sizes[3], "Width"),
        TraceLoggingInt32(runtime_session_id_, "runtimeSessionId"),
        TelemetryPrivacyDataTag(PDT_ProductAndServiceUsage),
        TraceLoggingBool(true, "UTCReplace_AppSessionGuid"),
        TraceLoggingKeyword(MICROSOFT_KEYWORD_MEASURES));
  }
  ~ConvertCPUTensorToVideoFrameWithSoftwareBitmapTelemetryEvent() {
    TraceLoggingWrite(
        winml_trace_logging_provider,
        "ConvertCPUTensorToVideoFrameWithSoftwareBitmapStop",
        TraceLoggingKeyword(WINML_PROVIDER_KEYWORD_DEFAULT),
        TraceLoggingHexInt32(S_OK, "HRESULT"),
        TraceLoggingInt32(runtime_session_id_, "runtimeSessionId"),
        TelemetryPrivacyDataTag(PDT_ProductAndServiceUsage),
        TraceLoggingBool(true, "UTCReplace_AppSessionGuid"),
        TraceLoggingKeyword(MICROSOFT_KEYWORD_MEASURES));
  }

private:
  int runtime_session_id_;
};

void TensorToVideoFrameConverter::DX12TensorToVideoFrame(
    _In_ UINT32 batchIdx,
    _In_ winml::LearningModelSession& session,
    _In_ ID3D12Resource* pInputTensor,
    _In_ const _winml::ImageTensorDescription& tensorDesc,
    _Inout_ wm::VideoFrame& destVideoFrame) {
  CWinMLAutoLock lock(&lock_);

  auto spDevice = session.Device().as<winmlp::LearningModelDevice>();
  _winml::D3DDeviceCache* pDeviceCache = spDevice->GetD3DDeviceCache();

  wgdx::Direct3D11::IDirect3DSurface spDestDirect3DSurface = destVideoFrame.Direct3DSurface();
  wgi::SoftwareBitmap softwareBitmap = destVideoFrame.SoftwareBitmap();

  if (softwareBitmap) {
   ConvertGPUTensorToSoftwareBitmap(batchIdx, pInputTensor, *pDeviceCache, tensorDesc, softwareBitmap);
  } else if (spDestDirect3DSurface) {
    bool isUAVSupportedFormat = _winmli::FormatSupportedForUAV(
        pDeviceCache->GetD3D12Device(),
        _winmli::GetDXGIFormatFromDirectXPixelFormat(spDestDirect3DSurface.Description().Format));

    // UAV support for formats is device dependent
    if (!isUAVSupportedFormat) {
      ConvertDX12TensorToUnsupportedVideoFrameFormat(batchIdx, pInputTensor, *pDeviceCache, tensorDesc, destVideoFrame);
    } else {
      ComPtr<ID3D11Texture2D> spVideoFrameTexture = _winmli::GetTextureFromDirect3DSurface(destVideoFrame.Direct3DSurface());

      D3D11_TEXTURE2D_DESC videoFrameTextureDesc;
      spVideoFrameTexture->GetDesc(&videoFrameTextureDesc);
      wgi::BitmapBounds bounds = {0, 0, videoFrameTextureDesc.Width, videoFrameTextureDesc.Height};

      if (_winmli::TextureIsOnDevice(spVideoFrameTexture.Get(), pDeviceCache->GetD3D11Device())) {
        // The texture is on our device, so we can just create own texture, share it and cache it
        if (!output_resource_) {
          output_resource_ = CreateShareableD3D12Texture(videoFrameTextureDesc, pDeviceCache->GetD3D12Device());
          D3D11_cached_texture_ = ShareD3D12Texture(output_resource_.Get(), pDeviceCache->GetD3D11Device());
        } else {
          D3D12_RESOURCE_DESC cachedTextureDesc = output_resource_->GetDesc();

          if (cachedTextureDesc.Width != videoFrameTextureDesc.Width || cachedTextureDesc.Height != videoFrameTextureDesc.Height || cachedTextureDesc.Format != videoFrameTextureDesc.Format) {
            // The dimensions or format don't match, so we need to re-create our texture
            output_resource_ = CreateShareableD3D12Texture(videoFrameTextureDesc, pDeviceCache->GetD3D12Device());
            D3D11_cached_texture_ = ShareD3D12Texture(output_resource_.Get(), pDeviceCache->GetD3D11Device());
          }
        }

        // Detensorize
        ConvertGPUTensorToDX12Texture(batchIdx, pInputTensor, *pDeviceCache, tensorDesc, output_resource_.Get());

        // Make sure that detensorization is done
        SyncD3D12ToD3D11(*pDeviceCache, D3D11_cached_texture_.Get());

        // Finally, copy the detensorized texture to the user's device
        CopyTextureIntoTexture(D3D11_cached_texture_.Get(), bounds, spVideoFrameTexture.Get());
      } else {
        // We are not on the same device, so we can't rely on our own cached texture
        ComPtr<ID3D11Device> spTextureDevice;
        spVideoFrameTexture->GetDevice(&spTextureDevice);

        ComPtr<ID3D11Texture2D> spSharedD3D11Texture;
        HANDLE sharedHandle = nullptr;
        UINT comPtrSize = static_cast<UINT>(sizeof(spSharedD3D11Texture.GetAddressOf()));
        UINT handleSize = static_cast<UINT>(sizeof(sharedHandle));

        if ((FAILED(spVideoFrameTexture->GetPrivateData(_d3d11TextureGUID, &comPtrSize, spSharedD3D11Texture.GetAddressOf())) || !spSharedD3D11Texture.Get()) || (FAILED(spVideoFrameTexture->GetPrivateData(_handleGUID, &handleSize, &sharedHandle)) || sharedHandle != shared_handle_)) {
          // Create a new shared texture that we cache on the video frame texture
          output_resource_ = CreateShareableD3D12Texture(videoFrameTextureDesc, pDeviceCache->GetD3D12Device());
          spSharedD3D11Texture = ShareD3D12Texture(output_resource_.Get(), spTextureDevice.Get());

          // Cache the shared texture on the video frame texture in order to tie their lifetime together
          WINML_THROW_IF_FAILED(spVideoFrameTexture->SetPrivateDataInterface(_d3d11TextureGUID, spSharedD3D11Texture.Get()));
          WINML_THROW_IF_FAILED(spVideoFrameTexture->SetPrivateData(_handleGUID, sizeof(shared_handle_), &shared_handle_));
        }

        // Detensorize
        ConvertGPUTensorToDX12Texture(batchIdx, pInputTensor, *pDeviceCache, tensorDesc, output_resource_.Get());

        // Make sure that detensorization is done
        SyncD3D12ToD3D11(*pDeviceCache, spSharedD3D11Texture.Get());

        // Finally, copy the detensorized texture to the user's device
        CopyTextureIntoTexture(spSharedD3D11Texture.Get(), bounds, spVideoFrameTexture.Get());
      }
    }
  } else {
    // Invalid video frame
    WINML_THROW_HR(E_INVALIDARG);
  }
}

ComPtr<ID3D12Resource> TensorToVideoFrameConverter::CreateShareableD3D12Texture(
    const D3D11_TEXTURE2D_DESC& d3d11Desc,
    ID3D12Device* d3d12Device) {
  D3D12_HEAP_PROPERTIES heapProps {};
  heapProps.Type = D3D12_HEAP_TYPE_DEFAULT;

  D3D12_RESOURCE_DESC resDesc {};
  resDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
  resDesc.Width = d3d11Desc.Width;
  resDesc.Height = d3d11Desc.Height;
  resDesc.DepthOrArraySize = static_cast<UINT16>(d3d11Desc.ArraySize);
  resDesc.MipLevels = static_cast<UINT16>(d3d11Desc.MipLevels);
  resDesc.Format = d3d11Desc.Format;
  resDesc.SampleDesc = d3d11Desc.SampleDesc;
  resDesc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
  resDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET | D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS;

  ComPtr<ID3D12Resource> d3d12Resource;
  WINML_THROW_IF_FAILED(d3d12Device->CreateCommittedResource(
    &heapProps,
    D3D12_HEAP_FLAG_SHARED,
    &resDesc,
    D3D12_RESOURCE_STATE_COPY_DEST,
    nullptr,
    IID_PPV_ARGS(&d3d12Resource)));

  return d3d12Resource;
}

void TensorToVideoFrameConverter::ConvertDX12TensorToUnsupportedVideoFrameFormat(
    _In_ UINT32 batchIdx,
    _In_ ID3D12Resource* pInputTensor,
    _In_ _winml::D3DDeviceCache& device_cache,
    _In_ const ImageTensorDescription& tensorDesc,
    _Inout_ wm::VideoFrame& unsupportedVideoFrame) {
  assert(pInputTensor != nullptr);

  // Find the first supported format and convert to it
  auto supportedFormatIter = std::find_if(
      _winmli::supportedWinMLFormats.begin(),
      _winmli::supportedWinMLFormats.end(),
      [&device_cache](DXGI_FORMAT format) { return _winmli::FormatSupportedForUAV(device_cache.GetD3D12Device(), format); });

  WINML_THROW_HR_IF_FALSE_MSG(
      E_INVALIDARG,
      supportedFormatIter != _winmli::supportedWinMLFormats.end(),
      "Detensorization for this format is unsupported on the current device.");

  D3D11_TEXTURE2D_DESC supportedDesc {};
  supportedDesc.Width = unsupportedVideoFrame.Direct3DSurface().Description().Width;
  supportedDesc.Height = unsupportedVideoFrame.Direct3DSurface().Description().Height;
  supportedDesc.MipLevels = 1;
  supportedDesc.ArraySize = 1;
  supportedDesc.Format = *supportedFormatIter;
  supportedDesc.SampleDesc.Count = 1;
  supportedDesc.SampleDesc.Quality = 0;
  supportedDesc.Usage = D3D11_USAGE_DEFAULT;

  ComPtr<ID3D11Texture2D> unsupportedTexture = _winmli::GetTextureFromDirect3DSurface(unsupportedVideoFrame.Direct3DSurface());

  ComPtr<ID3D11Device> d3d11Device;
  unsupportedTexture->GetDevice(&d3d11Device);

  output_resource_ = CreateShareableD3D12Texture(supportedDesc, device_cache.GetD3D12Device());
  ComPtr<ID3D11Texture2D> spSharedD3D11Texture = ShareD3D12Texture(output_resource_.Get(), d3d11Device.Get());

  ComPtr<IDXGISurface> dxgiSurface;
  WINML_THROW_IF_FAILED(spSharedD3D11Texture->QueryInterface(IID_PPV_ARGS(&dxgiSurface)));

  ComPtr<IInspectable> inspectableSurface;
  WINML_THROW_IF_FAILED(CreateDirect3D11SurfaceFromDXGISurface(dxgiSurface.Get(), &inspectableSurface));

  wgdx::Direct3D11::IDirect3DSurface surface;
  WINML_THROW_IF_FAILED(inspectableSurface->QueryInterface(winrt::guid_of<decltype(surface)>(), reinterpret_cast<void**>(winrt::put_abi(surface))));
  converted_video_frame_ = wm::VideoFrame::CreateWithDirect3D11Surface(surface);

  // Detensorize
  ConvertGPUTensorToDX12Texture(batchIdx, pInputTensor, device_cache, tensorDesc, output_resource_.Get());

  // Wait for the D3D12 work to complete before using the resource
  SyncD3D12ToD3D11(device_cache, spSharedD3D11Texture.Get());

  // Finally, convert and copy the texture to the destination video frame
  converted_video_frame_.CopyToAsync(unsupportedVideoFrame).get();
}

ComPtr<ID3D11Texture2D> TensorToVideoFrameConverter::ShareD3D12Texture(ID3D12Resource* pResource, ID3D11Device* pDevice)
{
    assert(pResource != nullptr);
    assert(pDevice != nullptr);

    ComPtr<ID3D12Device> d3d12Device;
    WINML_THROW_IF_FAILED(pResource->GetDevice(IID_PPV_ARGS(&d3d12Device)));

    HANDLE hSharedTexture;
    WINML_THROW_IF_FAILED(d3d12Device->CreateSharedHandle(pResource, nullptr, GENERIC_ALL, nullptr, &hSharedTexture));

    ComPtr<ID3D11Device1> device1;
    WINML_THROW_IF_FAILED(pDevice->QueryInterface(IID_PPV_ARGS(&device1)));

    wil::unique_handle safeHandle(hSharedTexture);

    ComPtr<ID3D11Texture2D> d3d11Texture;
    WINML_THROW_IF_FAILED(device1->OpenSharedResource1(safeHandle.get(), IID_PPV_ARGS(&d3d11Texture)));

    shared_handle_ = safeHandle.get();

    return d3d11Texture;
}

void TensorToVideoFrameConverter::SoftwareTensorToVideoFrame(
    _In_ winml::LearningModelSession& session,
    _In_ BYTE* pCPUTensorToConvert,
    _In_ ImageTensorDescription tensorDesc,
    _Inout_ wm::VideoFrame& pDestVideoFrame) {
  CWinMLAutoLock lock(&lock_);
  wm::IVideoFrame spTensorFrame;
  UINT32 outputWidth = 0;
  UINT32 outputHeight = 0;

  UINT32 tensorHeight = static_cast<UINT32>(tensorDesc.sizes[2]);
  UINT32 tensorWidth = static_cast<UINT32>(tensorDesc.sizes[3]);
  // create a bitmap bounds for the whole image/tensor
  wgi::BitmapBounds inputBounds =
      {
          0,
          0,
          tensorWidth,
          tensorHeight};

  wgi::SoftwareBitmap spOutputSoftwareBitmap = pDestVideoFrame.SoftwareBitmap();
  wgdx::Direct3D11::IDirect3DSurface spOutputSurface = pDestVideoFrame.Direct3DSurface();

  // only one of softwarebitmap or direct3Dsurface should be non-null
  if ((spOutputSoftwareBitmap == nullptr && spOutputSurface == nullptr) || (spOutputSoftwareBitmap != nullptr && spOutputSurface != nullptr)) {
    WINML_THROW_HR(E_INVALIDARG);
  }
  if (spOutputSoftwareBitmap) {
    outputWidth = spOutputSoftwareBitmap.PixelWidth();
    outputHeight = spOutputSoftwareBitmap.PixelHeight();
  } else {
    wgdx::Direct3D11::Direct3DSurfaceDescription description;
    description = spOutputSurface.Description();
    outputWidth = description.Width;
    outputHeight = description.Height;
  }

  if (_winmli::NeedsVideoFrameConversion(pDestVideoFrame, {}, {0, 0, (UINT32)tensorWidth, (UINT32)tensorHeight}, tensorWidth, tensorHeight)) {
    if (converted_video_frame_ == nullptr ||
        _winmli::NeedsVideoFrameConversion(converted_video_frame_, {}, {0, 0, (UINT32)tensorWidth, (UINT32)tensorHeight}, tensorWidth, tensorHeight)) {
      converted_video_frame_ = wm::VideoFrame::CreateWithSoftwareBitmap(
          wgi::SoftwareBitmap(wgi::BitmapPixelFormat::Bgra8, tensorWidth, tensorHeight));
    }

    spTensorFrame = converted_video_frame_;
  } else {
    spTensorFrame = pDestVideoFrame;
    converted_video_frame_ = nullptr;
  }
  auto bitmap = spTensorFrame.SoftwareBitmap();
  ConvertCPUTensorToSoftwareBitmap(
      pCPUTensorToConvert,
      tensorDesc,
      bitmap);

  if (converted_video_frame_) {
    _winmli::ConvertVideoFrameToVideoFrame(
        converted_video_frame_,
        inputBounds,
        outputWidth,
        outputHeight,
        pDestVideoFrame);
  }
}

void TensorToVideoFrameConverter::ConvertGPUTensorToDX12Texture(
    _In_ UINT32 batchIdx,
    _In_ ID3D12Resource* pInputResource,
    _In_ _winml::D3DDeviceCache& device_cache,
    _In_ const ImageTensorDescription& tensorDesc,
    _Inout_ ID3D12Resource* pOutputResource) {
  assert(pInputResource != nullptr);
  assert(pOutputResource != nullptr);

  CWinMLAutoLock lock(&lock_);
  D3D12_RESOURCE_DESC inputDesc = pInputResource->GetDesc();
  D3D12_RESOURCE_DESC outputDesc = pOutputResource->GetDesc();
  CD3DX12_VIEWPORT viewport((float)0, (float)0, (float)outputDesc.Width, (float)outputDesc.Height);
  CD3DX12_RECT scissorRect(0, 0, (LONG)outputDesc.Width, outputDesc.Height);
  ComPtr<ID3D12Device> spDx12Device = device_cache.GetD3D12Device();

  // we're inside a lock from the caller of this function, so it's ok to use this static
  static EventTimer eventTimer;
  std::optional<GPUTensorToDX12TextureTelemetryEvent> telemetryLogger;
  if (eventTimer.Start()) {
    telemetryLogger.emplace(tensorDesc);
  }

  WINML_THROW_HR_IF_FALSE_MSG(
      E_INVALIDARG,
      outputDesc.Format == DXGI_FORMAT_B8G8R8A8_UNORM || outputDesc.Format == DXGI_FORMAT_R8G8B8A8_UNORM || outputDesc.Format == DXGI_FORMAT_R8_UNORM,
      "Format was output image %d. Output image format must be Bgra8, Rgba8 or Gray8.",
      outputDesc.Format);

  // Validate input description
  WINML_THROW_HR_IF_FALSE_MSG(E_INVALIDARG, inputDesc.Height != 0, "Invalid input image height provided. Height is set to zero.");
  WINML_THROW_HR_IF_FALSE_MSG(E_INVALIDARG, inputDesc.Width != 0, "Invalid input image height provided. Height is set to zero.");

  // Validate output description
  WINML_THROW_HR_IF_FALSE_MSG(E_INVALIDARG, outputDesc.Height != 0, "Invalid input image height provided. Height is set to zero.");
  WINML_THROW_HR_IF_FALSE_MSG(E_INVALIDARG, outputDesc.Width != 0, "Invalid input image height provided. Height is set to zero.");

  // Validate Tensor description
  WINML_THROW_HR_IF_FALSE_MSG(E_INVALIDARG, tensorDesc.dataType == kImageTensorDataTypeFloat32 || tensorDesc.dataType == kImageTensorDataTypeFloat16, "Target tensor description must either be kImageTensorDataTypeFloat32, or kImageTensorDataTypeFloat16. %d was supplied.", tensorDesc.dataType);
  WINML_THROW_HR_IF_FALSE_MSG(E_INVALIDARG, tensorDesc.channelType != kImageTensorChannelTypeRGB8 || tensorDesc.sizes[1] == 3, "Target tensor description expects kImageTensorChannelTypeRGB8, but has %lld channels specified instead of 3.", tensorDesc.sizes[1]);
  WINML_THROW_HR_IF_FALSE_MSG(E_INVALIDARG, tensorDesc.channelType != kImageTensorChannelTypeBGR8 || tensorDesc.sizes[1] == 3, "Target tensor description expects kImageTensorChannelTypeBGR8, but has %lld channels specified instead of 3.", tensorDesc.sizes[1]);
  WINML_THROW_HR_IF_FALSE_MSG(E_INVALIDARG, tensorDesc.channelType != kImageTensorChannelTypeGRAY8 || tensorDesc.sizes[1] == 1, "Target tensor description expects kImageTensorChannelTypeGRAY8, but has %lld channels specified instead of 1.", tensorDesc.sizes[1]);
  WINML_THROW_HR_IF_FALSE_MSG(E_INVALIDARG, tensorDesc.sizes[2] == outputDesc.Height, "Target tensor height (%lld) does not match input height (%lu).", tensorDesc.sizes[2], outputDesc.Height);
  WINML_THROW_HR_IF_FALSE_MSG(E_INVALIDARG, tensorDesc.sizes[3] == (UINT)outputDesc.Width, "Target tensor width (%lld) does not match input width (%lu).", tensorDesc.sizes[3], (UINT)outputDesc.Width);

  // Create descriptor heaps
  UINT srvUavDescriptorSize = spDx12Device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

  // Create a UAV resource for the shader
  D3D12_RESOURCE_DESC outputResourceDesc = output_resource_->GetDesc();
  outputResourceDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

  if (!UAV_resource_ || outputDesc.Format != UAV_resource_->GetDesc().Format || outputDesc.Width != UAV_resource_->GetDesc().Width || outputDesc.Height != UAV_resource_->GetDesc().Height) {
    CD3DX12_HEAP_PROPERTIES prop(D3D12_HEAP_TYPE_DEFAULT);
    WINML_THROW_IF_FAILED(device_cache.GetD3D12Device()->CreateCommittedResource(
        &prop,
        D3D12_HEAP_FLAG_NONE,
        &outputResourceDesc,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        nullptr,
        IID_PPV_ARGS(&UAV_resource_)));
  }

  if (descriptor_heap_ == nullptr) {
    // Describe and create a shader resource view (SRV) and unordered access view (UAV) descriptor heap.
    D3D12_DESCRIPTOR_HEAP_DESC srvUavHeapDesc = {};
    srvUavHeapDesc.NumDescriptors = DescriptorCount;
    srvUavHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    srvUavHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    WINML_THROW_IF_FAILED(spDx12Device->CreateDescriptorHeap(&srvUavHeapDesc, IID_PPV_ARGS(&descriptor_heap_)));
    descriptor_heap_->SetName(L"Detensorize Descriptor Heap");
  }

  // Create SRV and UAV for input and output respectively
  {
    D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = CreateSRVDescriptor(batchIdx, inputDesc, tensorDesc);
    CD3DX12_CPU_DESCRIPTOR_HANDLE srvHandle(descriptor_heap_->GetCPUDescriptorHandleForHeapStart(), SrvBufferIdx, srvUavDescriptorSize);
    spDx12Device->CreateShaderResourceView(pInputResource, &srvDesc, srvHandle);

    D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
    uavDesc.Format = outputResourceDesc.Format;
    uavDesc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
    CD3DX12_CPU_DESCRIPTOR_HANDLE uavHandle(descriptor_heap_->GetCPUDescriptorHandleForHeapStart(), UavBufferIdx, srvUavDescriptorSize);
    spDx12Device->CreateUnorderedAccessView(UAV_resource_.Get(), nullptr, &uavDesc, uavHandle);
  }

  //
  // Pipeline setup for shader operation
  //
  PipelineStateCacheType type = PipelineStateCacheType::kFloat32;
  if (tensorDesc.dataType == kImageTensorDataTypeFloat16) {
    type = PipelineStateCacheType::kFloat16;
  }

  // Set the origin format
  PipelineStateCacheFormat formatFrom = PipelineStateCacheFormat::kBGR8;
  if (tensorDesc.channelType == kImageTensorChannelTypeRGB8) {
    formatFrom = PipelineStateCacheFormat::kRGB8;
  } else if (inputDesc.Format == kImageTensorChannelTypeGRAY8) {
    formatFrom = PipelineStateCacheFormat::kGRAY8;
  }

  // Set the destination format
  PipelineStateCacheFormat formatTo = PipelineStateCacheFormat::kBGR8;
  if (outputDesc.Format == DXGI_FORMAT_R8G8B8A8_UNORM) {
    formatTo = PipelineStateCacheFormat::kRGB8;
  } else if (outputDesc.Format == DXGI_FORMAT_R8_UNORM) {
    formatTo = PipelineStateCacheFormat::kGRAY8;
  }

  root_signature_ = device_cache.GetDetensorizeRootSignature();
  pipeline_state_ = device_cache.GetCachedPipelineState(type, formatFrom, formatTo, PipelineStateCacheOperation::kDetensorize);

  ResetCommandList(device_cache);

  // Write compute commands into the command list and put it into the queue.
  {
    command_list_->SetComputeRootSignature(root_signature_.Get());

    ID3D12DescriptorHeap* ppHeaps[] = {descriptor_heap_.Get()};
    command_list_->SetDescriptorHeaps(_countof(ppHeaps), ppHeaps);

    // This code currently re-uses the same decriptors each execution, which is unsafe if previous executions are in flight.
    if (fence_completion_value_ > 0)
    {
        device_cache.WaitForFenceValue(fence_completion_value_);
    }

    CD3DX12_GPU_DESCRIPTOR_HANDLE srvHandle(descriptor_heap_->GetGPUDescriptorHandleForHeapStart(), SrvBufferIdx, srvUavDescriptorSize);
    CD3DX12_GPU_DESCRIPTOR_HANDLE uavHandle(descriptor_heap_->GetGPUDescriptorHandleForHeapStart(), UavBufferIdx, srvUavDescriptorSize);
    {
      ConstantBufferCS constantBufferCS = {};
      constantBufferCS.height = static_cast<UINT>(tensorDesc.sizes[2]);
      constantBufferCS.width = static_cast<UINT>(tensorDesc.sizes[3]);
      command_list_->SetComputeRoot32BitConstants(0, 2, &constantBufferCS, 0);
    }
    command_list_->SetComputeRootDescriptorTable(1, srvHandle);
    command_list_->SetComputeRootDescriptorTable(2, uavHandle);

    auto dispatchWidth = static_cast<UINT>((tensorDesc.sizes[3] - 1) / 16 + 1);
    auto dispatchHeight = static_cast<UINT>((tensorDesc.sizes[2] - 1) / 4 + 1);
    CD3DX12_RESOURCE_BARRIER barrier = CD3DX12_RESOURCE_BARRIER::Transition(pInputResource, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    command_list_->ResourceBarrier(1, &barrier);
    command_list_->Dispatch(dispatchWidth, dispatchHeight, 1);
    barrier = CD3DX12_RESOURCE_BARRIER::Transition(pInputResource, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    command_list_->ResourceBarrier(1, &barrier);
    barrier = CD3DX12_RESOURCE_BARRIER::Transition(UAV_resource_.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
    // Copy the UAV data to the output resource after detensorization
    command_list_->ResourceBarrier(1, &barrier);
    command_list_->CopyResource(pOutputResource, UAV_resource_.Get());
    barrier = CD3DX12_RESOURCE_BARRIER::Transition(UAV_resource_.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    command_list_->ResourceBarrier(1, &barrier);

    WINML_THROW_IF_FAILED(command_list_->Close());
    ID3D12CommandList* pComputeToGPUCLs[] = {command_list_.Get()};
    device_cache.GetCommandQueue()->ExecuteCommandLists(ARRAYSIZE(pComputeToGPUCLs), pComputeToGPUCLs);
    fence_completion_value_ = device_cache.QueueFenceToD3D12();
  }
}

void TensorToVideoFrameConverter::ConvertGPUTensorToSoftwareBitmap(
    _In_ UINT32 batchIdx,
    _In_ ID3D12Resource* pInputTensor,
    _In_ _winml::D3DDeviceCache& device_cache,
    _In_ const ImageTensorDescription& tensorDesc,
    _Inout_ wgi::SoftwareBitmap& softwareBitmap) {
  assert(pInputTensor != nullptr);
  assert(softwareBitmap != nullptr);

  // we're inside a lock from the caller of this function, so it's ok to use this static
  static EventTimer eventTimer;
  std::optional<ConvertGPUTensorToSoftwareBitmapTelemetryEvent> telemetryLogger;
  if (eventTimer.Start()) {
    telemetryLogger.emplace(tensorDesc);
  }

  uint32_t tensorElementSize = tensorDesc.dataType == kImageTensorDataTypeFloat32 ? 4 : 2;
  uint32_t singleVideoFramebufferSize = static_cast<uint32_t>(tensorDesc.sizes[1] * tensorDesc.sizes[2] * tensorDesc.sizes[3] * tensorElementSize);

  // TODO: Make an allocator for readback heaps
  if (!readback_heap_ || readback_heap_->GetDesc().Width < singleVideoFramebufferSize) {
    CD3DX12_HEAP_PROPERTIES prop(D3D12_HEAP_TYPE_READBACK);
    CD3DX12_RESOURCE_DESC buffer = CD3DX12_RESOURCE_DESC::Buffer(singleVideoFramebufferSize);
    WINML_THROW_IF_FAILED(device_cache.GetD3D12Device()->CreateCommittedResource(
        &prop,
        D3D12_HEAP_FLAG_NONE,
        &buffer,
        D3D12_RESOURCE_STATE_COPY_DEST,
        nullptr,
        IID_PPV_ARGS(&readback_heap_)));
  }

  ResetCommandList(device_cache);

  auto barrier = CD3DX12_RESOURCE_BARRIER::Transition(pInputTensor, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
  command_list_->ResourceBarrier(1, &barrier);

  command_list_->CopyBufferRegion(readback_heap_.Get(), 0, pInputTensor, singleVideoFramebufferSize * batchIdx, singleVideoFramebufferSize);

  WINML_THROW_IF_FAILED(command_list_->Close());
  ID3D12CommandList* ppCommandLists[] = {command_list_.Get()};
  device_cache.GetCommandQueue()->ExecuteCommandLists(_countof(ppCommandLists), ppCommandLists);

  // Sync to make sure the the heap received all the data
  device_cache.SyncD3D12ToCPU();

  void* pCPUTensorBuffer = nullptr;
  CD3DX12_RANGE range(0, singleVideoFramebufferSize);
  WINML_THROW_IF_FAILED(readback_heap_->Map(0, &range, &pCPUTensorBuffer));

  // We avoid the Video Frame pipeline by manually downloading the GPU data to the CPU and detensorize while we are filling the readback heap
  ConvertCPUTensorToSoftwareBitmap(pCPUTensorBuffer, tensorDesc, softwareBitmap);
  CD3DX12_RANGE range2(0, 0);
  readback_heap_->Unmap(0, &range2);
}

void TensorToVideoFrameConverter::ConvertBatchedDX12TensorToBuffers(
    _In_ ID3D12Resource* input_tensor,
    _In_ size_t buffer_size_in_bytes,
    _In_ _winml::D3DDeviceCache& device_cache,
    _Inout_ const std::vector<wss::IBuffer>& buffers) {
  assert(input_tensor != nullptr);

  // TODO: Make an allocator for readback heaps
  if (!readback_heap_ || readback_heap_->GetDesc().Width < buffer_size_in_bytes) {
    CD3DX12_HEAP_PROPERTIES prop(D3D12_HEAP_TYPE_READBACK);
    CD3DX12_RESOURCE_DESC buffer = CD3DX12_RESOURCE_DESC::Buffer(buffer_size_in_bytes);
    WINML_THROW_IF_FAILED(device_cache.GetD3D12Device()->CreateCommittedResource(
        &prop,
        D3D12_HEAP_FLAG_NONE,
        &buffer,
        D3D12_RESOURCE_STATE_COPY_DEST,
        nullptr,
        IID_PPV_ARGS(&readback_heap_)));
  }

  ResetCommandList(device_cache);

  auto barrier = CD3DX12_RESOURCE_BARRIER::Transition(input_tensor, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
  command_list_->ResourceBarrier(1, &barrier);
  command_list_->CopyBufferRegion(readback_heap_.Get(), 0, input_tensor, 0, buffer_size_in_bytes);

  WINML_THROW_IF_FAILED(command_list_->Close());
  ID3D12CommandList* ppCommandLists[] = {command_list_.Get()};
  device_cache.GetCommandQueue()->ExecuteCommandLists(_countof(ppCommandLists), ppCommandLists);

  // Sync to make sure the the heap received all the data
  device_cache.SyncD3D12ToCPU();

  byte* readback_buffer = nullptr;
  CD3DX12_RANGE range(0, buffer_size_in_bytes);
  WINML_THROW_IF_FAILED(readback_heap_->Map(0, &range, reinterpret_cast<void**>(&readback_buffer)));
  auto readback_buffer_span = gsl::span<byte>(readback_buffer, buffer_size_in_bytes);
  _winml::StoreSpanIntoDisjointBuffers(
      buffers.size(),
      [&](size_t i) {
        byte* buffer_start = nullptr;
        auto byte_access = buffers[i].as<Windows::Storage::Streams::IBufferByteAccess>();
        byte_access->Buffer(&buffer_start);
        return gsl::span<byte>(buffer_start, static_cast<size_t>(buffers[i].Capacity()));
      },
      readback_buffer_span);

  CD3DX12_RANGE range2(0, 0);
  readback_heap_->Unmap(0, &range2);
}

D3D12_SHADER_RESOURCE_VIEW_DESC TensorToVideoFrameConverter::CreateSRVDescriptor(
    const UINT32 batchIdx,
    const D3D12_RESOURCE_DESC& resourceDesc,
    const _winml::ImageTensorDescription& desc) {
  UINT uiTensorElementSize =
      desc.dataType == kImageTensorDataTypeFloat32 ? sizeof(UINT) : sizeof(uint16_t);

  D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
  srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
  srvDesc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
  UINT singleImageSize = static_cast<UINT>(desc.sizes[1] * desc.sizes[2] * desc.sizes[3]);
  srvDesc.Buffer.FirstElement = batchIdx * desc.sizes[1] * desc.sizes[2] * desc.sizes[3];
  srvDesc.Buffer.NumElements = singleImageSize;
  srvDesc.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_NONE;

  if (desc.dataType == kImageTensorDataTypeFloat32) {
    // fp32 uses structured buffers so the format can be set to unknown,
    // and the stride needs to be set.
    srvDesc.Format = resourceDesc.Format;
    srvDesc.Buffer.StructureByteStride = uiTensorElementSize;
  } else if (desc.dataType == kImageTensorDataTypeFloat16) {
    // fp16 uses unstructured buffers because structured buffers dont support fp16 on
    // most hardware. The format can be set to unknown to a specific known format,
    // and the stride must be zeroed.
    srvDesc.Format = DXGI_FORMAT_R16_FLOAT;
    srvDesc.Buffer.StructureByteStride = 0;
  } else {
    WINML_THROW_HR_IF_FALSE_MSG(
        E_INVALIDARG,
        false,
        "Tensorization conversion is only supported to kImageTensorDataTypeFloat32, or kImageTensorDataTypeFloat16.");
  }

  return srvDesc;
}

void TensorToVideoFrameConverter::ConvertCPUTensorToSoftwareBitmap(
    _In_ void* pCPUTensor,
    _In_ const ImageTensorDescription& tensorDesc,
    _Inout_ wgi::SoftwareBitmap& softwareBitmap) {

  // we're inside a lock from the caller of this function, so it's ok to use this static
  static EventTimer eventTimer;
  std::optional<ConvertCPUTensorToVideoFrameWithSoftwareBitmapTelemetryEvent> telemetryLogger;
  if (eventTimer.Start()) {
    telemetryLogger.emplace(tensorDesc);
  }

  auto height = softwareBitmap.PixelHeight();
  auto width = softwareBitmap.PixelWidth();
  auto format = softwareBitmap.BitmapPixelFormat();

  // Validate input description
  WINML_THROW_HR_IF_FALSE_MSG(
      E_INVALIDARG,
      format == wgi::BitmapPixelFormat::Bgra8 || format == wgi::BitmapPixelFormat::Rgba8 || format == wgi::BitmapPixelFormat::Gray8,
      "Format was input image %d. Input image format must Bgra8, Rgba8 or Gray8.",
      format);
  WINML_THROW_HR_IF_FALSE_MSG(E_INVALIDARG, height > 0, "Output input image height provided. Height is set to zero.");
  WINML_THROW_HR_IF_FALSE_MSG(E_INVALIDARG, width > 0, "Output input image width provided. Width is set to zero.");

  // Validate Tensor description
  WINML_THROW_HR_IF_FALSE_MSG(E_INVALIDARG, tensorDesc.dataType == kImageTensorDataTypeFloat32 || tensorDesc.dataType == kImageTensorDataTypeFloat16, "Target tensor description must either be kImageTensorDataTypeFloat32, or kImageTensorDataTypeFloat16. %d was supplied.", tensorDesc.dataType);
  WINML_THROW_HR_IF_FALSE_MSG(E_INVALIDARG, tensorDesc.channelType != kImageTensorChannelTypeRGB8 || tensorDesc.sizes[1] == 3, "Target tensor description expects kImageTensorChannelTypeRGB8, but has %lld channels specified instead of 3.", tensorDesc.sizes[1]);
  WINML_THROW_HR_IF_FALSE_MSG(E_INVALIDARG, tensorDesc.channelType != kImageTensorChannelTypeBGR8 || tensorDesc.sizes[1] == 3, "Target tensor description expects kImageTensorChannelTypeBGR8, but has %lld channels specified instead of 3.", tensorDesc.sizes[1]);
  WINML_THROW_HR_IF_FALSE_MSG(E_INVALIDARG, tensorDesc.channelType != kImageTensorChannelTypeGRAY8 || tensorDesc.sizes[1] == 1, "Target tensor description expects kImageTensorChannelTypeGRAY8, but has %lld channels specified instead of 1.", tensorDesc.sizes[1]);
  WINML_THROW_HR_IF_FALSE_MSG(
      E_INVALIDARG,
      tensorDesc.channelType == kImageTensorChannelTypeGRAY8 ||
          tensorDesc.channelType == kImageTensorChannelTypeBGR8 ||
          tensorDesc.channelType == kImageTensorChannelTypeRGB8,
      "Target tensor description expects kImageTensorChannelTypeGRAY8, kImageTensorChannelTypeBGR8, or kImageTensorChannelTypeRGB8 but has %d was specified.",
      tensorDesc.channelType);
  WINML_THROW_HR_IF_FALSE_MSG(E_INVALIDARG, tensorDesc.sizes[2] == (UINT)height, "Target tensor height (%lld) does not match input height (%lu).", tensorDesc.sizes[2], (UINT)height);
  WINML_THROW_HR_IF_FALSE_MSG(E_INVALIDARG, tensorDesc.sizes[3] == (UINT)width, "Target tensor width (%lld) does not match input width (%lu).", tensorDesc.sizes[3], (UINT)width);

  // get the byte buffer out of a softwarebitmap
  BYTE* pData = nullptr;
  UINT32 uiCapacity = 0;

  wgi::BitmapBuffer spBitmapBuffer(softwareBitmap.LockBuffer(wgi::BitmapBufferAccessMode::Write));
  wf::IMemoryBufferReference reference = spBitmapBuffer.CreateReference();
  auto spByteAccess = reference.as<Windows::Foundation::IMemoryBufferByteAccess>();
  WINML_THROW_IF_FAILED(spByteAccess->GetBuffer(&pData, &uiCapacity));

  uint32_t bufferWidth = uiCapacity / height;

  ImageTensorChannelType targetChannelType = _winmli::GetChannelTypeFromSoftwareBitmap(softwareBitmap);

  if (tensorDesc.dataType == kImageTensorDataTypeFloat32) {
    WINML_THROW_IF_FAILED(CpuDetensorizer::Detensorize<float>(
      tensorDesc.channelType,
      targetChannelType, 
      tensorDesc.pixelRange, 
      static_cast<float*>(pCPUTensor),
      bufferWidth, 
      height,
      width,
      pData));
  } else if (tensorDesc.dataType == kImageTensorDataTypeFloat16) {
    WINML_THROW_IF_FAILED(CpuDetensorizer::Detensorize<DirectX::PackedVector::HALF>(
      tensorDesc.channelType,
      targetChannelType,
      tensorDesc.pixelRange,
      static_cast<DirectX::PackedVector::HALF*>(pCPUTensor),
      bufferWidth,
      height,
      width,
      pData));
  }
}