// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "lib/Api.Image/pch.h"

#include <winmeta.h>  // winmeta needed for TraceLoggingKeyword
#include <TraceLoggingProvider.h>
#include <TraceloggingConfig.h>
#include <evntrace.h>
#include <MemoryBuffer.h>

#include "inc/VideoFrameToTensorConverter.h"
#include "CpuTensorizer.h"
#include "inc/D3DDeviceCache.h"

#include "LearningModelDevice.h"
#include "EventTimer.h"

#include "robuffer.h"
#include "inc/DisjointBufferHelpers.h"

using namespace Microsoft::WRL;
using namespace Windows::Graphics::DirectX::Direct3D11;

using namespace _winml;

class DX12TextureToGPUTensorTelemetryEvent {
 public:
  DX12TextureToGPUTensorTelemetryEvent(const ImageTensorDescription& tensorDesc) {
    runtime_session_id_ = telemetry_helper.GetRuntimeSessionId();
    TraceLoggingWrite(
      winml_trace_logging_provider,
      "DX12TextureToGPUTensorStart",
      TraceLoggingKeyword(WINML_PROVIDER_KEYWORD_DEFAULT),
      TraceLoggingHexInt32(tensorDesc.channelType, "Type"),
      TraceLoggingInt64(tensorDesc.sizes[2], "Height"),
      TraceLoggingInt64(tensorDesc.sizes[3], "Width"),
      TraceLoggingInt32(runtime_session_id_, "runtimeSessionId"),
      TelemetryPrivacyDataTag(PDT_ProductAndServiceUsage),
      TraceLoggingBool(true, "UTCReplace_AppSessionGuid"),
      TraceLoggingKeyword(MICROSOFT_KEYWORD_MEASURES)
    );
  }
  ~DX12TextureToGPUTensorTelemetryEvent() {
    TraceLoggingWrite(
      winml_trace_logging_provider,
      "DX12TextureToGPUTensorStop",
      TraceLoggingKeyword(WINML_PROVIDER_KEYWORD_DEFAULT),
      TraceLoggingHexInt32(S_OK, "HRESULT"),
      TraceLoggingInt32(runtime_session_id_, "runtimeSessionId"),
      TelemetryPrivacyDataTag(PDT_ProductAndServiceUsage),
      TraceLoggingBool(true, "UTCReplace_AppSessionGuid"),
      TraceLoggingKeyword(MICROSOFT_KEYWORD_MEASURES)
    );
  }

 private:
  int runtime_session_id_;
};

class SoftwareBitmapToGPUTensorTelemetryEvent {
 public:
  SoftwareBitmapToGPUTensorTelemetryEvent(const ImageTensorDescription& tensorDesc) {
    runtime_session_id_ = telemetry_helper.GetRuntimeSessionId();
    TraceLoggingWrite(
      winml_trace_logging_provider,
      "SoftwareBitmapToGPUTensorStart",
      TraceLoggingKeyword(WINML_PROVIDER_KEYWORD_DEFAULT),
      TraceLoggingHexInt32(tensorDesc.channelType, "Type"),
      TraceLoggingInt64(tensorDesc.sizes[2], "Height"),
      TraceLoggingInt64(tensorDesc.sizes[3], "Width"),
      TraceLoggingInt32(runtime_session_id_, "runtimeSessionId"),
      TelemetryPrivacyDataTag(PDT_ProductAndServiceUsage),
      TraceLoggingBool(true, "UTCReplace_AppSessionGuid"),
      TraceLoggingKeyword(MICROSOFT_KEYWORD_MEASURES)
    );
  }
  ~SoftwareBitmapToGPUTensorTelemetryEvent() {
    TraceLoggingWrite(
      winml_trace_logging_provider,
      "SoftwareBitmapToGPUTensorStop",
      TraceLoggingKeyword(WINML_PROVIDER_KEYWORD_DEFAULT),
      TraceLoggingHexInt32(S_OK, "HRESULT"),
      TraceLoggingInt32(runtime_session_id_, "runtimeSessionId"),
      TelemetryPrivacyDataTag(PDT_ProductAndServiceUsage),
      TraceLoggingBool(true, "UTCReplace_AppSessionGuid"),
      TraceLoggingKeyword(MICROSOFT_KEYWORD_MEASURES)
    );
  }

 private:
  int runtime_session_id_;
};

class ConvertVideoFrameWithSoftwareBitmapToCPUTensorTelemetryEvent {
 public:
  ConvertVideoFrameWithSoftwareBitmapToCPUTensorTelemetryEvent(const ImageTensorDescription& tensorDesc) {
    runtime_session_id_ = telemetry_helper.GetRuntimeSessionId();
    TraceLoggingWrite(
      winml_trace_logging_provider,
      "ConvertVideoFrameWithSoftwareBitmapToCPUTensorStart",
      TraceLoggingKeyword(WINML_PROVIDER_KEYWORD_DEFAULT),
      TraceLoggingHexInt32(tensorDesc.channelType, "Type"),
      TraceLoggingInt64(tensorDesc.sizes[2], "Height"),
      TraceLoggingInt64(tensorDesc.sizes[3], "Width"),
      TraceLoggingInt32(runtime_session_id_, "runtimeSessionId"),
      TelemetryPrivacyDataTag(PDT_ProductAndServiceUsage),
      TraceLoggingBool(true, "UTCReplace_AppSessionGuid"),
      TraceLoggingKeyword(MICROSOFT_KEYWORD_MEASURES)
    );
  }
  ~ConvertVideoFrameWithSoftwareBitmapToCPUTensorTelemetryEvent() {
    TraceLoggingWrite(
      winml_trace_logging_provider,
      "ConvertVideoFrameWithSoftwareBitmapToCPUTensorStop",
      TraceLoggingKeyword(WINML_PROVIDER_KEYWORD_DEFAULT),
      TraceLoggingHexInt32(S_OK, "HRESULT"),
      TraceLoggingInt32(runtime_session_id_, "runtimeSessionId"),
      TelemetryPrivacyDataTag(PDT_ProductAndServiceUsage),
      TraceLoggingBool(true, "UTCReplace_AppSessionGuid"),
      TraceLoggingKeyword(MICROSOFT_KEYWORD_MEASURES)
    );
  }

 private:
  int runtime_session_id_;
};

void VideoFrameToTensorConverter::VideoFrameToSoftwareTensor(
  _In_ const wm::IVideoFrame& inputVideoFrame,
  _In_ const wgi::BitmapBounds& inputBounds,
  _In_ const ImageTensorDescription& tensorDesc,
  _Out_ BYTE* pOutputCPUTensor
) {
  CWinMLAutoLock lock(&lock_);

  wgi::SoftwareBitmap spInputSoftwareBitmap = inputVideoFrame.SoftwareBitmap();
  wgdx::Direct3D11::IDirect3DSurface spInputSurface = inputVideoFrame.Direct3DSurface();

  // only one of softwarebitmap or direct3Dsurface should be non-null
  if ((spInputSoftwareBitmap == nullptr && spInputSurface == nullptr) || (spInputSoftwareBitmap != nullptr && spInputSurface != nullptr)) {
    WINML_THROW_IF_FAILED(E_INVALIDARG);
  }

  UINT32 tensorHeight = static_cast<UINT32>(tensorDesc.sizes[2]);
  UINT32 tensorWidth = static_cast<UINT32>(tensorDesc.sizes[3]);
  if (spInputSurface || _winmli::NeedsVideoFrameConversion(inputVideoFrame, {}, inputBounds, tensorWidth, tensorHeight)) {
    if (converted_video_frame_ == nullptr || _winmli::NeedsVideoFrameConversion(converted_video_frame_, {}, {0, 0, (UINT32)tensorWidth, (UINT32)tensorHeight}, tensorWidth, tensorHeight)) {
      converted_video_frame_ = wm::VideoFrame::CreateWithSoftwareBitmap(
        wgi::SoftwareBitmap(wgi::BitmapPixelFormat::Bgra8, tensorWidth, tensorHeight)
      );
    }

    // Resize the input VideoFrame to converted_video_frame_
    _winmli::ConvertVideoFrameToVideoFrame(
      inputVideoFrame, inputBounds, tensorWidth, tensorHeight, converted_video_frame_
    );

    ConvertSoftwareBitmapToCPUTensor(
      converted_video_frame_.SoftwareBitmap(),
      tensorDesc,
      {0, 0, (UINT32)tensorWidth, (UINT32)tensorHeight},
      pOutputCPUTensor
    );
  } else {
    ConvertSoftwareBitmapToCPUTensor(inputVideoFrame.SoftwareBitmap(), tensorDesc, inputBounds, pOutputCPUTensor);
  }
}

ComPtr<ID3D12Resource> VideoFrameToTensorConverter::ShareD3D11Texture(
  ID3D11Texture2D* pTexture, ID3D12Device* pDevice
) {
  assert(pTexture != nullptr);
  assert(pDevice != nullptr);

  ComPtr<IDXGIResource1> spDxgiResource;
  WINML_THROW_IF_FAILED(pTexture->QueryInterface(IID_PPV_ARGS(&spDxgiResource)));

  HANDLE hSharedTexture;
  WINML_THROW_IF_FAILED(spDxgiResource->CreateSharedHandle(nullptr, GENERIC_ALL, nullptr, &hSharedTexture));

  wil::unique_handle safeHandle(hSharedTexture);

  ComPtr<ID3D12Resource> d3d12Resource;
  WINML_THROW_IF_FAILED(pDevice->OpenSharedHandle(safeHandle.get(), IID_PPV_ARGS(&d3d12Resource)));

  shared_handle_ = safeHandle.get();

  return d3d12Resource;
}

void VideoFrameToTensorConverter::VideoFrameToDX12Tensor(
  _In_ const UINT32 batchIdx,
  _In_ winml::LearningModelSession& session,
  _In_ const wm::IVideoFrame& inputVideoFrame,
  _In_ const wgi::BitmapBounds& inputBounds,
  _In_ const ImageTensorDescription& tensorDesc,
  _Inout_ ID3D12Resource* pOutputTensor
) {
  // Validate Tensor description
  WINML_THROW_HR_IF_FALSE_MSG(
    E_INVALIDARG,
    tensorDesc.dataType == kImageTensorDataTypeFloat32 || tensorDesc.dataType == kImageTensorDataTypeFloat16,
    "Target tensor description must either be kImageTensorDataTypeFloat32, or kImageTensorDataTypeFloat16. %d was supplied.",
    tensorDesc.dataType
  );
  WINML_THROW_HR_IF_FALSE_MSG(
    E_INVALIDARG,
    tensorDesc.channelType != kImageTensorChannelTypeRGB8 || tensorDesc.sizes[1] == 3,
    "Target tensor description expects kImageTensorChannelTypeRGB8, but has %lld channels specified instead of 3.",
    tensorDesc.sizes[1]
  );
  WINML_THROW_HR_IF_FALSE_MSG(
    E_INVALIDARG,
    tensorDesc.channelType != kImageTensorChannelTypeBGR8 || tensorDesc.sizes[1] == 3,
    "Target tensor description expects kImageTensorChannelTypeBGR8, but has %lld channels specified instead of 3.",
    tensorDesc.sizes[1]
  );
  WINML_THROW_HR_IF_FALSE_MSG(
    E_INVALIDARG,
    tensorDesc.channelType != kImageTensorChannelTypeGRAY8 || tensorDesc.sizes[1] == 1,
    "Target tensor description expects kImageTensorChannelTypeGRAY8, but has %lld channels specified instead of 1.",
    tensorDesc.sizes[1]
  );

  CWinMLAutoLock lock(&lock_);
  auto device = session.Device().as<winmlp::LearningModelDevice>();
  _winml::D3DDeviceCache* pDeviceCache = device->GetD3DDeviceCache();
  wgdx::Direct3D11::IDirect3DSurface spDirect3DSurface = inputVideoFrame.Direct3DSurface();

  if (inputVideoFrame.SoftwareBitmap()) {
    ConvertSoftwareBitmapToGPUTensor(batchIdx, inputVideoFrame, *pDeviceCache, inputBounds, tensorDesc, pOutputTensor);
  } else if (spDirect3DSurface) {
    ComPtr<ID3D11Texture2D> spVideoFrameTexture;
    wgi::BitmapBounds scaledBounds = inputBounds;

    // TODO: Scale during the tensorization phase instead of using the video frame pipeline when the input bounds are not the same size as the tensor
    if (!_winmli::DirectXPixelFormatSupported(spDirect3DSurface.Description().Format) ||
            static_cast<UINT>(inputBounds.Width) != tensorDesc.sizes[3] ||
            static_cast<UINT>(inputBounds.Height) != tensorDesc.sizes[2]) {
      // Force the VideoFrame to not do a conversion if the format is supported since we do it during the tensorization anyway
      wgdx::DirectXPixelFormat newFormat = _winmli::DirectXPixelFormatSupported(spDirect3DSurface.Description().Format)
        ? spDirect3DSurface.Description().Format
        : _winmli::GetDirectXPixelFormatFromChannelType(tensorDesc.channelType);

      // Change the input bounds since the video frame pipeline already cropped the texture
      scaledBounds = {0, 0, static_cast<uint32_t>(tensorDesc.sizes[3]), static_cast<uint32_t>(tensorDesc.sizes[2])};

      // Use the Video Frame pipeline if we don't have our own converter for this color format
      spVideoFrameTexture =
        CreateTextureFromUnsupportedColorFormat(inputVideoFrame, inputBounds, scaledBounds, newFormat);
    } else {
      // If the color format is known or the input widths are not smaller than the tensor desc, just use the video frame as is
      spVideoFrameTexture = _winmli::GetTextureFromDirect3DSurface(spDirect3DSurface);
    }

    D3D11_TEXTURE2D_DESC videoFrameTextureDesc;
    spVideoFrameTexture->GetDesc(&videoFrameTextureDesc);

    if (_winmli::TextureIsOnDevice(spVideoFrameTexture.Get(), pDeviceCache->GetD3D11Device())) {
      // The texture is on our device, so we can just create own texture, share it and cache it
      if (!D3D11_cached_texture_) {
        WINML_THROW_IF_FAILED(
          pDeviceCache->GetD3D11Device()->CreateTexture2D(&videoFrameTextureDesc, nullptr, &D3D11_cached_texture_)
        );
        input_D3D12_resource_ = ShareD3D11Texture(D3D11_cached_texture_.Get(), pDeviceCache->GetD3D12Device());
      } else {
        D3D11_TEXTURE2D_DESC cachedTextureDesc;
        D3D11_cached_texture_->GetDesc(&cachedTextureDesc);

        if (cachedTextureDesc.Width != scaledBounds.Width || cachedTextureDesc.Height != scaledBounds.Height ||
                    cachedTextureDesc.Format != videoFrameTextureDesc.Format) {
          // The dimensions or format don't match, so we need to re-create our texture
          WINML_THROW_IF_FAILED(
            pDeviceCache->GetD3D11Device()->CreateTexture2D(&videoFrameTextureDesc, nullptr, &D3D11_cached_texture_)
          );
          input_D3D12_resource_ = ShareD3D11Texture(D3D11_cached_texture_.Get(), pDeviceCache->GetD3D12Device());
        }
      }

      CopyTextureIntoTexture(spVideoFrameTexture.Get(), scaledBounds, D3D11_cached_texture_.Get());
    } else {
      // We are not on the same device, so we can't rely on our cached texture
      ComPtr<ID3D11Device> spTextureDevice;
      spVideoFrameTexture->GetDevice(&spTextureDevice);

      ComPtr<ID3D11Texture2D> spSharedD3D11Texture;
      HANDLE sharedHandle = nullptr;
      UINT comPtrSize = static_cast<UINT>(sizeof(spSharedD3D11Texture.GetAddressOf()));
      UINT handleSize = static_cast<UINT>(sizeof(sharedHandle));

      if ((FAILED(spVideoFrameTexture->GetPrivateData(
                     d3d11_texture_GUID_, &comPtrSize, spSharedD3D11Texture.GetAddressOf()
                 )) ||
                 !spSharedD3D11Texture.Get()) ||
                (FAILED(spVideoFrameTexture->GetPrivateData(handle_GUID_, &handleSize, &sharedHandle)) ||
                 sharedHandle != shared_handle_)) {
        // Create a new shared texture that we cache on the video frame texture
        WINML_THROW_IF_FAILED(spTextureDevice->CreateTexture2D(&videoFrameTextureDesc, nullptr, &spSharedD3D11Texture));

        input_D3D12_resource_ = ShareD3D11Texture(spSharedD3D11Texture.Get(), pDeviceCache->GetD3D12Device());

        // Cache the shared texture on the video frame texture in order to tie their lifetime together
        WINML_THROW_IF_FAILED(
          spVideoFrameTexture->SetPrivateDataInterface(d3d11_texture_GUID_, spSharedD3D11Texture.Get())
        );
        WINML_THROW_IF_FAILED(spVideoFrameTexture->SetPrivateData(handle_GUID_, sizeof(shared_handle_), &shared_handle_)
        );
      }

      // Copy from the video frame texture to the shared texture
      CopyTextureIntoTexture(spVideoFrameTexture.Get(), scaledBounds, spSharedD3D11Texture.Get());
    }

    // Sync to make sure that the D3D11 texture is done copying
    SyncD3D11ToD3D12(*pDeviceCache, spVideoFrameTexture.Get());

    // We cropped the texture, shared it and converted it to a known color format, so it's time to tensorize
    // TODO: merge all videoframes to a single DX12Texture Resource before call ConvertDX12TextureToGPUTensor.
    ConvertDX12TextureToGPUTensor(batchIdx, input_D3D12_resource_.Get(), *pDeviceCache, tensorDesc, pOutputTensor);
  } else {
    // Invalid video frame
    WINML_THROW_IF_FAILED(E_INVALIDARG);
  }
}

void VideoFrameToTensorConverter::ConvertDX12TextureToGPUTensor(
  _In_ UINT32 batchIdx,
  _In_ ID3D12Resource* pInputResource,
  _In_ _winml::D3DDeviceCache& device_cache,
  _In_ const ImageTensorDescription& tensorDesc,
  _Inout_ ID3D12Resource* pOutputResource
) {
  assert(pInputResource != nullptr);
  assert(pOutputResource != nullptr);

  CWinMLAutoLock lock(&lock_);
  D3D12_RESOURCE_DESC inputDesc = pInputResource->GetDesc();
  D3D12_RESOURCE_DESC outputDesc = pOutputResource->GetDesc();
  ComPtr<ID3D12Device> spDx12Device = device_cache.GetD3D12Device();

  // we're inside a lock from the caller of this function, so it's ok to use this static
  static EventTimer eventTimer;
  std::optional<DX12TextureToGPUTensorTelemetryEvent> telemetryLogger;
  if (eventTimer.Start()) {
    telemetryLogger.emplace(tensorDesc);
  }

  // Validate input description
  WINML_THROW_HR_IF_FALSE_MSG(
    E_INVALIDARG,
    inputDesc.Format == DXGI_FORMAT_B8G8R8X8_UNORM || inputDesc.Format == DXGI_FORMAT_B8G8R8A8_UNORM ||
      inputDesc.Format == DXGI_FORMAT_R8G8B8A8_UNORM || inputDesc.Format == DXGI_FORMAT_R8_UNORM,
    "Format was input image %d. Input image format must Bgra8, Rgba8 or Gray8.",
    inputDesc.Format
  );

  WINML_THROW_HR_IF_FALSE_MSG(
    E_INVALIDARG, inputDesc.Width != 0, "Invalid input image height provided. Width is set to zero."
  );
  WINML_THROW_HR_IF_FALSE_MSG(
    E_INVALIDARG, inputDesc.Height != 0, "Invalid input image height provided. Height is set to zero."
  );

  // Validate Tensor description
  WINML_THROW_HR_IF_FALSE_MSG(
    E_INVALIDARG,
    tensorDesc.dataType == kImageTensorDataTypeFloat32 || tensorDesc.dataType == kImageTensorDataTypeFloat16,
    "Target tensor description must either be kImageTensorDataTypeFloat32, or kImageTensorDataTypeFloat16. %d was supplied.",
    tensorDesc.dataType
  );
  WINML_THROW_HR_IF_FALSE_MSG(
    E_INVALIDARG,
    tensorDesc.channelType != kImageTensorChannelTypeRGB8 || tensorDesc.sizes[1] == 3,
    "Target tensor description expects kImageTensorChannelTypeRGB8, but has %lld channels specified instead of 3.",
    tensorDesc.sizes[1]
  );
  WINML_THROW_HR_IF_FALSE_MSG(
    E_INVALIDARG,
    tensorDesc.channelType != kImageTensorChannelTypeBGR8 || tensorDesc.sizes[1] == 3,
    "Target tensor description expects kImageTensorChannelTypeBGR8, but has %lld channels specified instead of 3.",
    tensorDesc.sizes[1]
  );
  WINML_THROW_HR_IF_FALSE_MSG(
    E_INVALIDARG,
    tensorDesc.channelType != kImageTensorChannelTypeGRAY8 || tensorDesc.sizes[1] == 1,
    "Target tensor description expects kImageTensorChannelTypeGRAY8, but has %lld channels specified instead of 1.",
    tensorDesc.sizes[1]
  );
  WINML_THROW_HR_IF_FALSE_MSG(
    E_INVALIDARG,
    tensorDesc.sizes[2] == inputDesc.Height,
    "Target tensor height (%lld) does not match input height (%lu).",
    tensorDesc.sizes[2],
    inputDesc.Height
  );
  WINML_THROW_HR_IF_FALSE_MSG(
    E_INVALIDARG,
    tensorDesc.sizes[3] == (UINT)inputDesc.Width,
    "Target tensor width (%lld) does not match input width (%lu).",
    tensorDesc.sizes[3],
    (UINT)inputDesc.Width
  );

  UINT uiTensorElementSize = tensorDesc.dataType == kImageTensorDataTypeFloat32 ? sizeof(FLOAT) : sizeof(uint16_t);

  // Validate Tensor Resource
  {
    D3D12_HEAP_PROPERTIES outputHeapProperties;
    D3D12_HEAP_FLAGS outputHeapFlags;

    WINML_THROW_IF_FAILED(pOutputResource->GetHeapProperties(&outputHeapProperties, &outputHeapFlags));

    UINT64 ullNumElementsTensor = 1;
    for (UINT uiIdx = 0; uiIdx < kImageTensorDimensionCountMax; uiIdx++) {
      WINML_THROW_IF_FAILED(ULongLongMult(ullNumElementsTensor, tensorDesc.sizes[uiIdx], &ullNumElementsTensor));
    }
    if (ullNumElementsTensor > UINT_MAX) {
      WINML_THROW_IF_FAILED(E_INVALIDARG);
    }

    UINT64 ullTensorSize = 0;
    WINML_THROW_IF_FAILED(ULongLongMult(ullNumElementsTensor, uiTensorElementSize, &ullTensorSize));

    if (outputDesc.Width < ullTensorSize || outputDesc.Height != 1 ||
            outputDesc.Dimension != D3D12_RESOURCE_DIMENSION_BUFFER ||
            !(outputDesc.Flags & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS) ||
            outputHeapProperties.Type != D3D12_HEAP_TYPE_DEFAULT) {
      WINML_THROW_IF_FAILED(E_INVALIDARG);
    }
  }

  {
    ComPtr<ID3D12Device> spDx12DeviceIn, spDx12DeviceOut;
    WINML_THROW_IF_FAILED(pInputResource->GetDevice(IID_PPV_ARGS(&spDx12DeviceIn)));
    WINML_THROW_IF_FAILED(pOutputResource->GetDevice(IID_PPV_ARGS(&spDx12DeviceOut)));

    if (spDx12Device != spDx12DeviceIn || spDx12Device != spDx12DeviceOut) {
      // Both input and output should have the same device
      WINML_THROW_IF_FAILED(E_INVALIDARG);
    }
  }

  // Create descriptor heaps.
  UINT srvUavDescriptorSize = spDx12Device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

  if (descriptor_heap_ == nullptr) {
    // Describe and create a shader resource view (SRV) and unordered access view (UAV) descriptor heap.
    D3D12_DESCRIPTOR_HEAP_DESC srvUavHeapDesc = {};
    srvUavHeapDesc.NumDescriptors = DescriptorCount;
    srvUavHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    srvUavHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    WINML_THROW_IF_FAILED(spDx12Device->CreateDescriptorHeap(&srvUavHeapDesc, IID_PPV_ARGS(&descriptor_heap_)));
    descriptor_heap_->SetName(L"Tensorize Descriptor Heap");
  }

  // Create SRV and UAV for input and output respectively
  {
    D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
    srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srvDesc.Format = inputDesc.Format;
    srvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    srvDesc.Texture2D.MipLevels = 1;
    CD3DX12_CPU_DESCRIPTOR_HANDLE srvHandle(
      descriptor_heap_->GetCPUDescriptorHandleForHeapStart(), SrvBufferIdx, srvUavDescriptorSize
    );
    spDx12Device->CreateShaderResourceView(pInputResource, &srvDesc, srvHandle);

    D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc = CreateUAVDescription(batchIdx, outputDesc, tensorDesc);
    CD3DX12_CPU_DESCRIPTOR_HANDLE uavHandle(
      descriptor_heap_->GetCPUDescriptorHandleForHeapStart(), UavBufferIdx, srvUavDescriptorSize
    );
    spDx12Device->CreateUnorderedAccessView(pOutputResource, nullptr, &uavDesc, uavHandle);
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
  if (inputDesc.Format == DXGI_FORMAT_R8G8B8A8_UNORM) {
    formatFrom = PipelineStateCacheFormat::kRGB8;
  } else if (inputDesc.Format == DXGI_FORMAT_R8_UNORM) {
    formatFrom = PipelineStateCacheFormat::kGRAY8;
  }

  // Set the destination format
  PipelineStateCacheFormat formatTo = PipelineStateCacheFormat::kBGR8;
  if (tensorDesc.channelType == kImageTensorChannelTypeRGB8) {
    formatTo = PipelineStateCacheFormat::kRGB8;
  } else if (tensorDesc.channelType == kImageTensorChannelTypeGRAY8) {
    formatTo = PipelineStateCacheFormat::kGRAY8;
  }

  root_signature_ = device_cache.GetTensorizeRootSignature();
  pipeline_state_ =
    device_cache.GetCachedPipelineState(type, formatFrom, formatTo, PipelineStateCacheOperation::kTensorize);

  ResetCommandList(device_cache);

  // Write compute commands into the command list and put it into the queue.
  {
    command_list_->SetComputeRootSignature(root_signature_.Get());

    ID3D12DescriptorHeap* ppHeaps[] = {descriptor_heap_.Get()};
    command_list_->SetDescriptorHeaps(_countof(ppHeaps), ppHeaps);

    // This code currently re-uses the same decriptors each execution, which is unsafe if previous executions are in flight.
    if (fence_completion_value_ > 0) {
      device_cache.WaitForFenceValue(fence_completion_value_);
    }

    CD3DX12_GPU_DESCRIPTOR_HANDLE srvHandle(
      descriptor_heap_->GetGPUDescriptorHandleForHeapStart(), SrvBufferIdx, srvUavDescriptorSize
    );
    CD3DX12_GPU_DESCRIPTOR_HANDLE uavHandle(
      descriptor_heap_->GetGPUDescriptorHandleForHeapStart(), UavBufferIdx, srvUavDescriptorSize
    );
    {
      ConstantBufferCS constantBufferCS = {};
      constantBufferCS.height = inputDesc.Height;
      constantBufferCS.width = (UINT)inputDesc.Width;
      command_list_->SetComputeRoot32BitConstants(0, 2, &constantBufferCS, 0);
    }
    command_list_->SetComputeRootDescriptorTable(1, srvHandle);
    command_list_->SetComputeRootDescriptorTable(2, uavHandle);

    UINT64 dispatchWidth = (inputDesc.Width - 1) / 16 + 1;
    UINT64 dispatchHeight = (static_cast<UINT64>(inputDesc.Height) - 1) / 4 + 1;
    command_list_->Dispatch(static_cast<uint32_t>(dispatchWidth), static_cast<uint32_t>(dispatchHeight), 1);

    WINML_THROW_IF_FAILED(command_list_->Close());

    ID3D12CommandList* pComputeToGPUCLs[] = {command_list_.Get()};

    device_cache.GetCommandQueue()->ExecuteCommandLists(ARRAYSIZE(pComputeToGPUCLs), pComputeToGPUCLs);
    fence_completion_value_ = device_cache.QueueFenceToD3D12();
  }
}

void VideoFrameToTensorConverter::ConvertSoftwareBitmapToGPUTensor(
  _In_ UINT32 batchIdx,
  _In_ const wm::IVideoFrame& videoFrame,
  _In_ _winml::D3DDeviceCache& device_cache,
  _In_ const wgi::BitmapBounds& inputBounds,
  _In_ const ImageTensorDescription& tensorDesc,
  _Inout_ ID3D12Resource* pOutputResource
) {
  assert(pOutputResource != nullptr);
  assert(videoFrame.SoftwareBitmap() != nullptr);

  // we're inside a lock from the caller of this function, so it's ok to use this static
  static EventTimer eventTimer;
  std::optional<SoftwareBitmapToGPUTensorTelemetryEvent> telemetryLogger;
  if (eventTimer.Start()) {
    telemetryLogger.emplace(tensorDesc);
  }

  wgi::SoftwareBitmap convertedSoftwareBitmap = nullptr;
  wgi::BitmapBounds scaledBounds = inputBounds;

  // TODO: Scale during the tensorization phase instead of using the video frame pipeline when the input bounds are not the same size as the tensor
  if (static_cast<UINT>(inputBounds.Width) != tensorDesc.sizes[3] || static_cast<UINT>(inputBounds.Height) != tensorDesc.sizes[2]) {
    scaledBounds = {0, 0, static_cast<uint32_t>(tensorDesc.sizes[3]), static_cast<uint32_t>(tensorDesc.sizes[2])};

    // Force the VideoFrame to not do a conversion if the format is supported since we do it during the tensorization anyway
    wgi::BitmapPixelFormat newPixelFormat = _winmli::SoftwareBitmapFormatSupported(videoFrame.SoftwareBitmap())
      ? videoFrame.SoftwareBitmap().BitmapPixelFormat()
      : _winmli::GetBitmapPixelFormatFromChannelType(tensorDesc.channelType);

    convertedSoftwareBitmap = wgi::SoftwareBitmap(
      newPixelFormat, static_cast<int32_t>(tensorDesc.sizes[3]), static_cast<int32_t>(tensorDesc.sizes[2])
    );
    wm::VideoFrame convertedVideoFrame = wm::VideoFrame::CreateWithSoftwareBitmap(convertedSoftwareBitmap);
    videoFrame.as<wm::IVideoFrame2>().CopyToAsync(convertedVideoFrame, inputBounds, scaledBounds).get();

    convertedSoftwareBitmap = convertedVideoFrame.SoftwareBitmap();
  } else if (!_winmli::SoftwareBitmapFormatSupported(videoFrame.SoftwareBitmap())) {
    convertedSoftwareBitmap = wgi::SoftwareBitmap::Convert(
      videoFrame.SoftwareBitmap(), _winmli::GetBitmapPixelFormatFromChannelType(tensorDesc.channelType)
    );
  } else {
    // We don't need a conversion
    convertedSoftwareBitmap = videoFrame.SoftwareBitmap();
  }

  assert(convertedSoftwareBitmap != nullptr);

  D3D12_RESOURCE_DESC outputDesc = pOutputResource->GetDesc();

  uint32_t tensorElementSize = tensorDesc.dataType == kImageTensorDataTypeFloat32 ? 4 : 2;
  uint32_t bufferSize =
    static_cast<uint32_t>(tensorDesc.sizes[1] * tensorDesc.sizes[2] * tensorDesc.sizes[3] * tensorElementSize);

  // TODO: Make an allocator for upload heaps
  if (!upload_heap_ || upload_heap_->GetDesc().Width < bufferSize) {
    WINML_THROW_IF_FAILED(device_cache.GetD3D12Device()->CreateCommittedResource(
      &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD),
      D3D12_HEAP_FLAG_NONE,
      &CD3DX12_RESOURCE_DESC::Buffer(bufferSize),
      D3D12_RESOURCE_STATE_GENERIC_READ,
      nullptr,
      IID_PPV_ARGS(&upload_heap_)
    ));
  }

  void* pCPUTensorBuffer = nullptr;
  WINML_THROW_IF_FAILED(upload_heap_->Map(0, &CD3DX12_RANGE(0, 0), &pCPUTensorBuffer));

  // We avoid the Video Frame pipeline by manually sending the CPU data to the GPU, and we tensorize while we are filling the
  // upload heap. The image may already have been cropped/scaled by the video frame pipeline, so we send the scaled bounds
  // instead of the initial input bounds
  ConvertSoftwareBitmapToCPUTensor(convertedSoftwareBitmap, tensorDesc, scaledBounds, pCPUTensorBuffer);

  upload_heap_->Unmap(0, &CD3DX12_RANGE(0, bufferSize));

  ResetCommandList(device_cache);

  auto barrier = CD3DX12_RESOURCE_BARRIER::Transition(
    pOutputResource, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_DEST
  );
  command_list_->ResourceBarrier(1, &barrier);

  command_list_->CopyBufferRegion(pOutputResource, bufferSize * batchIdx, upload_heap_.Get(), 0, bufferSize);

  WINML_THROW_IF_FAILED(command_list_->Close());
  ID3D12CommandList* ppCommandLists[] = {command_list_.Get()};
  device_cache.GetCommandQueue()->ExecuteCommandLists(_countof(ppCommandLists), ppCommandLists);
}

void VideoFrameToTensorConverter::ConvertBuffersToBatchedGPUTensor(
  _In_ const std::vector<wss::IBuffer>& buffers,
  _In_ size_t buffer_size_in_bytes,
  _In_ _winml::D3DDeviceCache& device_cache,
  _Inout_ ID3D12Resource* output_resource
) {
  // Copy the cpu memory into the gpu resource
  if (!upload_heap_ || upload_heap_->GetDesc().Width < buffer_size_in_bytes) {
    WINML_THROW_IF_FAILED(device_cache.GetD3D12Device()->CreateCommittedResource(
      &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD),
      D3D12_HEAP_FLAG_NONE,
      &CD3DX12_RESOURCE_DESC::Buffer(buffer_size_in_bytes),
      D3D12_RESOURCE_STATE_GENERIC_READ,
      nullptr,
      IID_PPV_ARGS(&upload_heap_)
    ));
  }

  byte* gpu_buffer = nullptr;
  WINML_THROW_IF_FAILED(upload_heap_->Map(0, &CD3DX12_RANGE(0, 0), reinterpret_cast<void**>(&gpu_buffer)));
  auto gpu_buffer_span = gsl::span<byte>(gpu_buffer, buffer_size_in_bytes);

  _winml::LoadSpanFromDisjointBuffers(
    buffers.size(),
    [&](size_t i) {
      byte* buffer_start = nullptr;
      auto byte_access = buffers[i].as<Windows::Storage::Streams::IBufferByteAccess>();
      byte_access->Buffer(&buffer_start);
      return gsl::span<byte>(buffer_start, static_cast<size_t>(buffers[i].Capacity()));
    },
    gpu_buffer_span
  );

  upload_heap_->Unmap(0, &CD3DX12_RANGE(0, buffer_size_in_bytes));

  ResetCommandList(device_cache);

  auto barrier1 =
    CD3DX12_RESOURCE_BARRIER::Transition(output_resource, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
  command_list_->ResourceBarrier(1, &barrier1);
  command_list_->CopyBufferRegion(output_resource, 0, upload_heap_.Get(), 0, buffer_size_in_bytes);
  auto barrier2 = CD3DX12_RESOURCE_BARRIER::Transition(
    output_resource, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS
  );
  command_list_->ResourceBarrier(1, &barrier2);
  WINML_THROW_IF_FAILED(command_list_->Close());
  ID3D12CommandList* lists[] = {command_list_.Get()};
  device_cache.GetCommandQueue()->ExecuteCommandLists(_countof(lists), lists);
}

D3D12_UNORDERED_ACCESS_VIEW_DESC VideoFrameToTensorConverter::CreateUAVDescription(
  const UINT32 batchIdx, const D3D12_RESOURCE_DESC& resourceDesc, const _winml::ImageTensorDescription& desc
) {
  UINT uiTensorElementSize = desc.dataType == kImageTensorDataTypeFloat32 ? sizeof(UINT) : sizeof(uint16_t);

  D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
  uavDesc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
  UINT singleImageSize = static_cast<UINT>(desc.sizes[1] * desc.sizes[2] * desc.sizes[3]);
  uavDesc.Buffer.FirstElement = batchIdx * desc.sizes[1] * desc.sizes[2] * desc.sizes[3];
  uavDesc.Buffer.NumElements = singleImageSize;
  uavDesc.Buffer.CounterOffsetInBytes = 0;
  uavDesc.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_NONE;

  if (desc.dataType == kImageTensorDataTypeFloat32) {
    // fp32 uses structured buffers so the format can be set to unknown,
    // and the stride needs to be set.
    uavDesc.Format = DXGI_FORMAT_UNKNOWN;
    uavDesc.Buffer.StructureByteStride = uiTensorElementSize;
  } else if (desc.dataType == kImageTensorDataTypeFloat16) {
    // fp16 uses unstructured buffers because structured buffers dont support fp16 on
    // most hardware. The format can be set to unknown to a specific known format,
    // and the stride must be zeroed.
    uavDesc.Format = DXGI_FORMAT_R16_FLOAT;
    uavDesc.Buffer.StructureByteStride = 0;
  } else {
    WINML_THROW_HR_IF_FALSE_MSG(
      E_INVALIDARG,
      false,
      "Tensorization conversion is only supported to kImageTensorDataTypeFloat32, or kImageTensorDataTypeFloat16."
    );
  }

  return uavDesc;
}

void VideoFrameToTensorConverter::ConvertSoftwareBitmapToCPUTensor(
  _In_ const wgi::SoftwareBitmap& softwareBitmap,
  _In_ const _winml::ImageTensorDescription& tensorDesc,
  _In_ const wgi::BitmapBounds& inputBounds,
  _Inout_ void* pCPUTensor
) {
  assert(softwareBitmap != nullptr);

  // we're inside a lock from the caller of this function, so it's ok to use this static
  static EventTimer eventTimer;
  std::optional<ConvertVideoFrameWithSoftwareBitmapToCPUTensorTelemetryEvent> telemetryLogger;
  if (eventTimer.Start()) {
    telemetryLogger.emplace(tensorDesc);
  }

  auto height = softwareBitmap.PixelHeight();
  auto width = softwareBitmap.PixelWidth();
  auto format = softwareBitmap.BitmapPixelFormat();

  // Validate input description
  WINML_THROW_HR_IF_FALSE_MSG(
    E_INVALIDARG,
    format == wgi::BitmapPixelFormat::Bgra8 || format == wgi::BitmapPixelFormat::Rgba8 ||
      format == wgi::BitmapPixelFormat::Gray8,
    "Format was input image %d. Input image format must Bgra8, Rgba8 or Gray8.",
    format
  );
  WINML_THROW_HR_IF_FALSE_MSG(E_INVALIDARG, height > 0, "Invalid input image height provided. Height is set to zero.");
  WINML_THROW_HR_IF_FALSE_MSG(E_INVALIDARG, width > 0, "Invalid input image width provided. Height is set to zero.");

  // Validate Tensor description
  WINML_THROW_HR_IF_FALSE_MSG(
    E_INVALIDARG,
    tensorDesc.dataType == kImageTensorDataTypeFloat32 || tensorDesc.dataType == kImageTensorDataTypeFloat16,
    "Target tensor description must either be kImageTensorDataTypeFloat32, or kImageTensorDataTypeFloat16. %d was supplied.",
    tensorDesc.dataType
  );
  WINML_THROW_HR_IF_FALSE_MSG(
    E_INVALIDARG,
    tensorDesc.channelType != kImageTensorChannelTypeRGB8 || tensorDesc.sizes[1] == 3,
    "Target tensor description expects kImageTensorChannelTypeRGB8, but has %lld channels specified instead of 3.",
    tensorDesc.sizes[1]
  );
  WINML_THROW_HR_IF_FALSE_MSG(
    E_INVALIDARG,
    tensorDesc.channelType != kImageTensorChannelTypeBGR8 || tensorDesc.sizes[1] == 3,
    "Target tensor description expects kImageTensorChannelTypeBGR8, but has %lld channels specified instead of 3.",
    tensorDesc.sizes[1]
  );
  WINML_THROW_HR_IF_FALSE_MSG(
    E_INVALIDARG,
    tensorDesc.channelType != kImageTensorChannelTypeGRAY8 || tensorDesc.sizes[1] == 1,
    "Target tensor description expects kImageTensorChannelTypeGRAY8, but has %lld channels specified instead of 1.",
    tensorDesc.sizes[1]
  );
  WINML_THROW_HR_IF_FALSE_MSG(
    E_INVALIDARG,
    tensorDesc.channelType == kImageTensorChannelTypeGRAY8 || tensorDesc.channelType == kImageTensorChannelTypeBGR8 ||
      tensorDesc.channelType == kImageTensorChannelTypeRGB8,
    "Target tensor description expects kImageTensorChannelTypeGRAY8, kImageTensorChannelTypeBGR8, or kImageTensorChannelTypeRGB8 but has %d was specified.",
    tensorDesc.channelType
  );
  WINML_THROW_HR_IF_FALSE_MSG(
    E_INVALIDARG,
    tensorDesc.sizes[2] == (UINT)inputBounds.Height,
    "Target tensor height (%lld) does not match input height (%lu).",
    tensorDesc.sizes[2],
    inputBounds.Height
  );
  WINML_THROW_HR_IF_FALSE_MSG(
    E_INVALIDARG,
    tensorDesc.sizes[3] == (UINT)inputBounds.Width,
    "Target tensor width (%lld) does not match input width (%lu).",
    tensorDesc.sizes[3],
    inputBounds.Width
  );

  // get the byte buffer out of a softwarebitmap
  BYTE* pData = nullptr;
  UINT32 bufferSize = 0;
  wgi::BitmapBuffer spBitmapBuffer(softwareBitmap.LockBuffer(wgi::BitmapBufferAccessMode::Read));
  wf::IMemoryBufferReference reference = spBitmapBuffer.CreateReference();
  auto spByteAccess = reference.as<Windows::Foundation::IMemoryBufferByteAccess>();
  WINML_THROW_IF_FAILED(spByteAccess->GetBuffer(&pData, &bufferSize));

  UINT32 bufferWidth = bufferSize / height;

  ImageTensorChannelType channelType = _winmli::GetChannelTypeFromSoftwareBitmap(softwareBitmap);

  if (tensorDesc.dataType == _winml::kImageTensorDataTypeFloat32) {
    WINML_THROW_IF_FAILED(CpuTensorizer::TensorizeData<float>(
      channelType,
      tensorDesc.channelType,
      tensorDesc.pixelRange,
      pData,
      bufferWidth,
      inputBounds,
      reinterpret_cast<float*>(pCPUTensor)
    ));
  } else if (tensorDesc.dataType == _winml::kImageTensorDataTypeFloat16) {
    WINML_THROW_IF_FAILED(CpuTensorizer::TensorizeData<DirectX::PackedVector::HALF>(
      channelType,
      tensorDesc.channelType,
      tensorDesc.pixelRange,
      pData,
      bufferWidth,
      inputBounds,
      reinterpret_cast<DirectX::PackedVector::HALF*>(pCPUTensor)
    ));
  }
}
