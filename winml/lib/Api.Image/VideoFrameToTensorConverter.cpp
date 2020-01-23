// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "pch.h"

#include <winmeta.h>  // winmeta needed for TraceLoggingKeyword
#include <TraceLoggingProvider.h>
#include <evntrace.h>
#include <MemoryBuffer.h>

#include "inc/VideoFrameToTensorConverter.h"
#include "CpuTensorizer.h"
#include "inc/D3DDeviceCache.h"

#include "LearningModelDevice.h"

using namespace Microsoft::WRL;
using namespace Windows::AI::MachineLearning::Internal;
using namespace Windows::Graphics::DirectX::Direct3D11;
using namespace winrt::Windows::Graphics::Imaging;
using namespace winrt::Windows::Graphics::DirectX::Direct3D11;
using namespace winrt::Windows::Media;
using namespace winrt::Windows::AI::MachineLearning::implementation;
using namespace winrt::Windows::Graphics::DirectX;

class DX12TextureToGPUTensorTelemetryEvent {
 public:
  DX12TextureToGPUTensorTelemetryEvent(const ImageTensorDescription& tensorDesc) {
    TraceLoggingWrite(
        winml_trace_logging_provider,
        "DX12TextureToGPUTensor",
        TraceLoggingKeyword(WINML_PROVIDER_KEYWORD_DEFAULT),
        TraceLoggingOpcode(EVENT_TRACE_TYPE_START),
        TraceLoggingHexInt32(tensorDesc.channelType, "Type"),
        TraceLoggingInt64(tensorDesc.sizes[2], "Height"),
        TraceLoggingInt64(tensorDesc.sizes[3], "Width"));
  }
  ~DX12TextureToGPUTensorTelemetryEvent() {
    TraceLoggingWrite(
        winml_trace_logging_provider,
        "DX12TextureToGPUTensor",
        TraceLoggingKeyword(WINML_PROVIDER_KEYWORD_DEFAULT),
        TraceLoggingOpcode(EVENT_TRACE_TYPE_STOP),
        TraceLoggingHexInt32(S_OK, "HRESULT"));
  }
};

class ConvertVideoFrameWithSoftwareBitmapToCPUTensorTelemetryEvent {
 public:
  ConvertVideoFrameWithSoftwareBitmapToCPUTensorTelemetryEvent(const ImageTensorDescription& tensorDesc) {
    TraceLoggingWrite(
        winml_trace_logging_provider,
        "ConvertVideoFrameWithSoftwareBitmapToCPUTensor",
        TraceLoggingKeyword(WINML_PROVIDER_KEYWORD_DEFAULT),
        TraceLoggingOpcode(EVENT_TRACE_TYPE_START),
        TraceLoggingHexInt32(tensorDesc.channelType, "Type"),
        TraceLoggingInt64(tensorDesc.sizes[2], "Height"),
        TraceLoggingInt64(tensorDesc.sizes[3], "Width"));
  }
  ~ConvertVideoFrameWithSoftwareBitmapToCPUTensorTelemetryEvent() {
    TraceLoggingWrite(
        winml_trace_logging_provider,
        "ConvertVideoFrameWithSoftwareBitmapToCPUTensor",
        TraceLoggingKeyword(WINML_PROVIDER_KEYWORD_DEFAULT),
        TraceLoggingOpcode(EVENT_TRACE_TYPE_STOP),
        TraceLoggingHexInt32(S_OK, "HRESULT"));
  }
};

void VideoFrameToTensorConverter::VideoFrameToSoftwareTensor(
    _In_ const IVideoFrame& inputVideoFrame,
    _In_ const BitmapBounds& inputBounds,
    _In_ const ImageTensorDescription& tensorDesc,
    _Out_ BYTE* pOutputCPUTensor) {
  CWinMLAutoLock lock(&lock_);

  winrt::Windows::Graphics::Imaging::SoftwareBitmap spInputSoftwareBitmap = inputVideoFrame.SoftwareBitmap();
  winrt::Windows::Graphics::DirectX::Direct3D11::IDirect3DSurface spInputSurface = inputVideoFrame.Direct3DSurface();

  // only one of softwarebitmap or direct3Dsurface should be non-null
  if ((spInputSoftwareBitmap == nullptr && spInputSurface == nullptr) || (spInputSoftwareBitmap != nullptr && spInputSurface != nullptr)) {
    WINML_THROW_IF_FAILED(E_INVALIDARG);
  }

  UINT32 tensorHeight = static_cast<UINT32>(tensorDesc.sizes[2]);
  UINT32 tensorWidth = static_cast<UINT32>(tensorDesc.sizes[3]);
  if (spInputSurface || ImageConversionHelpers::NeedsVideoFrameConversion(inputVideoFrame, {}, inputBounds, tensorWidth, tensorHeight)) {
    if (converted_video_frame_ == nullptr ||
        ImageConversionHelpers::NeedsVideoFrameConversion(converted_video_frame_, {}, {0, 0, (UINT32)tensorWidth, (UINT32)tensorHeight}, tensorWidth, tensorHeight)) {
      converted_video_frame_ = VideoFrame::CreateWithSoftwareBitmap(SoftwareBitmap(BitmapPixelFormat::Bgra8, tensorWidth, tensorHeight));
    }

    // Resize the input VideoFrame to converted_video_frame_
    ImageConversionHelpers::ConvertVideoFrameToVideoFrame(
        inputVideoFrame,
        inputBounds,
        tensorWidth,
        tensorHeight,
        converted_video_frame_);

    ConvertSoftwareBitmapToCPUTensor(
        converted_video_frame_.SoftwareBitmap(),
        tensorDesc,
        {0, 0, (UINT32)tensorWidth, (UINT32)tensorHeight},
        pOutputCPUTensor);
  } else {
    ConvertSoftwareBitmapToCPUTensor(
        inputVideoFrame.SoftwareBitmap(),
        tensorDesc,
        inputBounds,
        pOutputCPUTensor);
  }
}

ComPtr<ID3D12Resource> VideoFrameToTensorConverter::ShareD3D11Texture(ID3D11Texture2D* pTexture, ID3D12Device* pDevice)
{
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
    _In_ winrt::Windows::AI::MachineLearning::LearningModelSession& session,
    _In_ const IVideoFrame& inputVideoFrame,
    _In_ const BitmapBounds& inputBounds,
    _In_ const ImageTensorDescription& tensorDesc,
    _Inout_ ID3D12Resource* pOutputTensor) {
  // Validate Tensor description
  WINML_THROW_HR_IF_FALSE_MSG(E_INVALIDARG, tensorDesc.dataType == kImageTensorDataTypeFloat32 || tensorDesc.dataType == kImageTensorDataTypeFloat16, "Target tensor description must either be kImageTensorDataTypeFloat32, or kImageTensorDataTypeFloat16. %d was supplied.", tensorDesc.dataType);
  WINML_THROW_HR_IF_FALSE_MSG(E_INVALIDARG, tensorDesc.channelType != kImageTensorChannelTypeRGB8 || tensorDesc.sizes[1] == 3, "Target tensor description expects kImageTensorChannelTypeRGB8, but has %lld channels specified instead of 3.", tensorDesc.sizes[1]);
  WINML_THROW_HR_IF_FALSE_MSG(E_INVALIDARG, tensorDesc.channelType != kImageTensorChannelTypeBGR8 || tensorDesc.sizes[1] == 3, "Target tensor description expects kImageTensorChannelTypeBGR8, but has %lld channels specified instead of 3.", tensorDesc.sizes[1]);
  WINML_THROW_HR_IF_FALSE_MSG(E_INVALIDARG, tensorDesc.channelType != kImageTensorChannelTypeGRAY8 || tensorDesc.sizes[1] == 1, "Target tensor description expects kImageTensorChannelTypeGRAY8, but has %lld channels specified instead of 1.", tensorDesc.sizes[1]);

  CWinMLAutoLock lock(&lock_);
  auto device = session.Device().as<LearningModelDevice>();
  D3DDeviceCache* pDeviceCache = device->GetD3DDeviceCache();
  IDirect3DSurface spDirect3DSurface = inputVideoFrame.Direct3DSurface();

  if (inputVideoFrame.SoftwareBitmap()) {
    ConvertSoftwareBitmapToGPUTensor(batchIdx, inputVideoFrame, *pDeviceCache, inputBounds, tensorDesc, pOutputTensor);
  } else if (spDirect3DSurface) {
    ComPtr<ID3D11Texture2D> spVideoFrameTexture;
    BitmapBounds scaledBounds = inputBounds;

    // TODO: Scale during the tensorization phase instead of using the video frame pipeline when the input bounds are not the same size as the tensor
    if (!ImageConversionHelpers::DirectXPixelFormatSupported(spDirect3DSurface.Description().Format) || static_cast<UINT>(inputBounds.Width) != tensorDesc.sizes[3] || static_cast<UINT>(inputBounds.Height) != tensorDesc.sizes[2]) {
      // Force the VideoFrame to not do a conversion if the format is supported since we do it during the tensorization anyway
      DirectXPixelFormat newFormat = ImageConversionHelpers::DirectXPixelFormatSupported(spDirect3DSurface.Description().Format)
                                         ? spDirect3DSurface.Description().Format
                                         : ImageConversionHelpers::GetDirectXPixelFormatFromChannelType(tensorDesc.channelType);

      // Change the input bounds since the video frame pipeline already cropped the texture
      scaledBounds = {0, 0, static_cast<uint32_t>(tensorDesc.sizes[3]), static_cast<uint32_t>(tensorDesc.sizes[2])};

      // Use the Video Frame pipeline if we don't have our own converter for this color format
      spVideoFrameTexture = CreateTextureFromUnsupportedColorFormat(inputVideoFrame, inputBounds, scaledBounds, newFormat);
    } else {
      // If the color format is known or the input widths are not smaller than the tensor desc, just use the video frame as is
      spVideoFrameTexture = ImageConversionHelpers::GetTextureFromDirect3DSurface(spDirect3DSurface);
    }

    D3D11_TEXTURE2D_DESC videoFrameTextureDesc;
    spVideoFrameTexture->GetDesc(&videoFrameTextureDesc);

    if (ImageConversionHelpers::TextureIsOnDevice(spVideoFrameTexture.Get(), pDeviceCache->GetD3D11Device())) {
      // The texture is on our device, so we can just create own texture, share it and cache it
      if (!D3D11_cached_texture_) {
        WINML_THROW_IF_FAILED(pDeviceCache->GetD3D11Device()->CreateTexture2D(&videoFrameTextureDesc, nullptr, &D3D11_cached_texture_));
        input_D3D12_resource_ = ShareD3D11Texture(D3D11_cached_texture_.Get(), pDeviceCache->GetD3D12Device());
      } else {
        D3D11_TEXTURE2D_DESC cachedTextureDesc;
        D3D11_cached_texture_->GetDesc(&cachedTextureDesc);

        if (cachedTextureDesc.Width != scaledBounds.Width || cachedTextureDesc.Height != scaledBounds.Height || cachedTextureDesc.Format != videoFrameTextureDesc.Format) {
          // The dimensions or format don't match, so we need to re-create our texture
          WINML_THROW_IF_FAILED(pDeviceCache->GetD3D11Device()->CreateTexture2D(&videoFrameTextureDesc, nullptr, &D3D11_cached_texture_));
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

      if ((FAILED(spVideoFrameTexture->GetPrivateData(d3d11_texture_GUID_, &comPtrSize, spSharedD3D11Texture.GetAddressOf())) || !spSharedD3D11Texture.Get()) || (FAILED(spVideoFrameTexture->GetPrivateData(handle_GUID_, &handleSize, &sharedHandle)) || sharedHandle != shared_handle_)) {
        // Create a new shared texture that we cache on the video frame texture
        WINML_THROW_IF_FAILED(spTextureDevice->CreateTexture2D(&videoFrameTextureDesc, nullptr, &spSharedD3D11Texture));

        input_D3D12_resource_ = ShareD3D11Texture(spSharedD3D11Texture.Get(), pDeviceCache->GetD3D12Device());

        // Cache the shared texture on the video frame texture in order to tie their lifetime together
        WINML_THROW_IF_FAILED(spVideoFrameTexture->SetPrivateDataInterface(d3d11_texture_GUID_, spSharedD3D11Texture.Get()));
        WINML_THROW_IF_FAILED(spVideoFrameTexture->SetPrivateData(handle_GUID_, sizeof(shared_handle_), &shared_handle_));
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
    _In_ winrt::Windows::AI::MachineLearning::implementation::D3DDeviceCache& device_cache,
    _In_ const ImageTensorDescription& tensorDesc,
    _Inout_ ID3D12Resource* pOutputResource) {
  assert(pInputResource != nullptr);
  assert(pOutputResource != nullptr);

  CWinMLAutoLock lock(&lock_);
  D3D12_RESOURCE_DESC inputDesc = pInputResource->GetDesc();
  D3D12_RESOURCE_DESC outputDesc = pOutputResource->GetDesc();
  ComPtr<ID3D12Device> spDx12Device = device_cache.GetD3D12Device();

  DX12TextureToGPUTensorTelemetryEvent telemetrylogger(tensorDesc);

  // Validate input description
  WINML_THROW_HR_IF_FALSE_MSG(
      E_INVALIDARG,
      inputDesc.Format == DXGI_FORMAT_B8G8R8X8_UNORM || inputDesc.Format == DXGI_FORMAT_B8G8R8A8_UNORM || inputDesc.Format == DXGI_FORMAT_R8G8B8A8_UNORM || inputDesc.Format == DXGI_FORMAT_R8_UNORM,
      "Format was input image %d. Input image format must Bgra8, Rgba8 or Gray8.",
      inputDesc.Format);

  WINML_THROW_HR_IF_FALSE_MSG(E_INVALIDARG, inputDesc.Width != 0, "Invalid input image height provided. Width is set to zero.");
  WINML_THROW_HR_IF_FALSE_MSG(E_INVALIDARG, inputDesc.Height != 0, "Invalid input image height provided. Height is set to zero.");

  // Validate Tensor description
  WINML_THROW_HR_IF_FALSE_MSG(E_INVALIDARG, tensorDesc.dataType == kImageTensorDataTypeFloat32 || tensorDesc.dataType == kImageTensorDataTypeFloat16, "Target tensor description must either be kImageTensorDataTypeFloat32, or kImageTensorDataTypeFloat16. %d was supplied.", tensorDesc.dataType);
  WINML_THROW_HR_IF_FALSE_MSG(E_INVALIDARG, tensorDesc.channelType != kImageTensorChannelTypeRGB8 || tensorDesc.sizes[1] == 3, "Target tensor description expects kImageTensorChannelTypeRGB8, but has %lld channels specified instead of 3.", tensorDesc.sizes[1]);
  WINML_THROW_HR_IF_FALSE_MSG(E_INVALIDARG, tensorDesc.channelType != kImageTensorChannelTypeBGR8 || tensorDesc.sizes[1] == 3, "Target tensor description expects kImageTensorChannelTypeBGR8, but has %lld channels specified instead of 3.", tensorDesc.sizes[1]);
  WINML_THROW_HR_IF_FALSE_MSG(E_INVALIDARG, tensorDesc.channelType != kImageTensorChannelTypeGRAY8 || tensorDesc.sizes[1] == 1, "Target tensor description expects kImageTensorChannelTypeGRAY8, but has %lld channels specified instead of 1.", tensorDesc.sizes[1]);
  WINML_THROW_HR_IF_FALSE_MSG(E_INVALIDARG, tensorDesc.sizes[2] == inputDesc.Height, "Target tensor height (%lld) does not match input height (%d).", tensorDesc.sizes[2], inputDesc.Height);
  WINML_THROW_HR_IF_FALSE_MSG(E_INVALIDARG, tensorDesc.sizes[3] == (UINT)inputDesc.Width, "Target tensor width (%lld) does not match input width (%d).", tensorDesc.sizes[3], (UINT)inputDesc.Width);

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

    if (outputDesc.Width < ullTensorSize ||
        outputDesc.Height != 1 ||
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
    CD3DX12_CPU_DESCRIPTOR_HANDLE srvHandle(descriptor_heap_->GetCPUDescriptorHandleForHeapStart(), SrvBufferIdx, srvUavDescriptorSize);
    spDx12Device->CreateShaderResourceView(pInputResource, &srvDesc, srvHandle);

    D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc = CreateUAVDescription(batchIdx, outputDesc, tensorDesc);
    CD3DX12_CPU_DESCRIPTOR_HANDLE uavHandle(descriptor_heap_->GetCPUDescriptorHandleForHeapStart(), UavBufferIdx, srvUavDescriptorSize);
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
  pipeline_state_ = device_cache.GetCachedPipelineState(type, formatFrom, formatTo, PipelineStateCacheOperation::kTensorize);

  ResetCommandList(device_cache);

  // Write compute commands into the command list and put it into the queue.
  {
    command_list_->SetComputeRootSignature(root_signature_.Get());

    ID3D12DescriptorHeap* ppHeaps[] = {descriptor_heap_.Get()};
    command_list_->SetDescriptorHeaps(_countof(ppHeaps), ppHeaps);

    CD3DX12_GPU_DESCRIPTOR_HANDLE srvHandle(descriptor_heap_->GetGPUDescriptorHandleForHeapStart(), SrvBufferIdx, srvUavDescriptorSize);
    CD3DX12_GPU_DESCRIPTOR_HANDLE uavHandle(descriptor_heap_->GetGPUDescriptorHandleForHeapStart(), UavBufferIdx, srvUavDescriptorSize);
    {
      ConstantBufferCS constantBufferCS = {};
      constantBufferCS.height = inputDesc.Height;
      constantBufferCS.width = (UINT)inputDesc.Width;
      command_list_->SetComputeRoot32BitConstants(0, 2, &constantBufferCS, 0);
    }
    command_list_->SetComputeRootDescriptorTable(1, srvHandle);
    command_list_->SetComputeRootDescriptorTable(2, uavHandle);

    UINT64 dispatchWidth = (inputDesc.Width - 1) / 16 + 1;
    UINT64 dispatchHeight = (inputDesc.Height - 1) / 4 + 1;
    command_list_->Dispatch(static_cast<uint32_t>(dispatchWidth), static_cast<uint32_t>(dispatchHeight), 1);

    WINML_THROW_IF_FAILED(command_list_->Close());

    ID3D12CommandList* pComputeToGPUCLs[] = {command_list_.Get()};

    device_cache.GetCommandQueue()->ExecuteCommandLists(ARRAYSIZE(pComputeToGPUCLs), pComputeToGPUCLs);
  }
}

void VideoFrameToTensorConverter::ConvertSoftwareBitmapToGPUTensor(
    _In_ UINT32 batchIdx,
    _In_ const IVideoFrame& videoFrame,
    _In_ D3DDeviceCache& device_cache,
    _In_ const BitmapBounds& inputBounds,
    _In_ const ImageTensorDescription& tensorDesc,
    _Inout_ ID3D12Resource* pOutputResource) {
  assert(pOutputResource != nullptr);
  assert(videoFrame.SoftwareBitmap() != nullptr);

  DX12TextureToGPUTensorTelemetryEvent telemetrylogger(tensorDesc);

  SoftwareBitmap convertedSoftwareBitmap = nullptr;
  BitmapBounds scaledBounds = inputBounds;

  // TODO: Scale during the tensorization phase instead of using the video frame pipeline when the input bounds are not the same size as the tensor
  if (static_cast<UINT>(inputBounds.Width) != tensorDesc.sizes[3] || static_cast<UINT>(inputBounds.Height) != tensorDesc.sizes[2]) {
    scaledBounds = {0, 0, static_cast<uint32_t>(tensorDesc.sizes[3]), static_cast<uint32_t>(tensorDesc.sizes[2])};

    // Force the VideoFrame to not do a conversion if the format is supported since we do it during the tensorization anyway
    BitmapPixelFormat newPixelFormat = ImageConversionHelpers::SoftwareBitmapFormatSupported(videoFrame.SoftwareBitmap())
                                           ? videoFrame.SoftwareBitmap().BitmapPixelFormat()
                                           : ImageConversionHelpers::GetBitmapPixelFormatFromChannelType(tensorDesc.channelType);

    convertedSoftwareBitmap = SoftwareBitmap(newPixelFormat, static_cast<int32_t>(tensorDesc.sizes[3]), static_cast<int32_t>(tensorDesc.sizes[2]));
    VideoFrame convertedVideoFrame = VideoFrame::CreateWithSoftwareBitmap(convertedSoftwareBitmap);
    videoFrame.as<IVideoFrame2>().CopyToAsync(convertedVideoFrame, inputBounds, scaledBounds).get();

    convertedSoftwareBitmap = convertedVideoFrame.SoftwareBitmap();
  } else if (!ImageConversionHelpers::SoftwareBitmapFormatSupported(videoFrame.SoftwareBitmap())) {
    convertedSoftwareBitmap = SoftwareBitmap::Convert(videoFrame.SoftwareBitmap(), ImageConversionHelpers::GetBitmapPixelFormatFromChannelType(tensorDesc.channelType));
  } else {
    // We don't need a conversion
    convertedSoftwareBitmap = videoFrame.SoftwareBitmap();
  }

  assert(convertedSoftwareBitmap != nullptr);

  D3D12_RESOURCE_DESC outputDesc = pOutputResource->GetDesc();

  uint32_t tensorElementSize = tensorDesc.dataType == kImageTensorDataTypeFloat32 ? 4 : 2;
  uint32_t bufferSize = static_cast<uint32_t>(tensorDesc.sizes[1] * tensorDesc.sizes[2] * tensorDesc.sizes[3] * tensorElementSize);

  // TODO: Make an allocator for upload heaps
  if (!upload_heap_ || upload_heap_->GetDesc().Width < bufferSize) {
    const auto heapType = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD);
    const auto resourceDescription = CD3DX12_RESOURCE_DESC::Buffer(bufferSize);
    WINML_THROW_IF_FAILED(device_cache.GetD3D12Device()->CreateCommittedResource(
        &heapType,
        D3D12_HEAP_FLAG_NONE,
        &resourceDescription,
        D3D12_RESOURCE_STATE_GENERIC_READ,
        nullptr,
        IID_PPV_ARGS(&upload_heap_)));
  }

  void* pCPUTensorBuffer = nullptr;
  const auto mapRange = CD3DX12_RANGE(0, 0);
  WINML_THROW_IF_FAILED(upload_heap_->Map(0, &mapRange, &pCPUTensorBuffer));

  // We avoid the Video Frame pipeline by manually sending the CPU data to the GPU, and we tensorize while we are filling the
  // upload heap. The image may already have been cropped/scaled by the video frame pipeline, so we send the scaled bounds
  // instead of the initial input bounds
  ConvertSoftwareBitmapToCPUTensor(convertedSoftwareBitmap, tensorDesc, scaledBounds, pCPUTensorBuffer);

  const auto unmapRange = CD3DX12_RANGE(0, bufferSize);
  upload_heap_->Unmap(0, &unmapRange);

  ResetCommandList(device_cache);
  command_list_->CopyBufferRegion(pOutputResource, bufferSize * batchIdx, upload_heap_.Get(), 0, bufferSize);

  WINML_THROW_IF_FAILED(command_list_->Close());
  ID3D12CommandList* ppCommandLists[] = {command_list_.Get()};
  device_cache.GetCommandQueue()->ExecuteCommandLists(_countof(ppCommandLists), ppCommandLists);
}

D3D12_UNORDERED_ACCESS_VIEW_DESC VideoFrameToTensorConverter::CreateUAVDescription(
    const UINT32 batchIdx,
    const D3D12_RESOURCE_DESC& resourceDesc,
    const ImageTensorDescription& desc) {
  UINT uiTensorElementSize =
      desc.dataType == kImageTensorDataTypeFloat32 ? sizeof(UINT) : sizeof(uint16_t);

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
        "Tensorization conversion is only supported to kImageTensorDataTypeFloat32, or kImageTensorDataTypeFloat16.");
  }

  return uavDesc;
}

void VideoFrameToTensorConverter::ConvertSoftwareBitmapToCPUTensor(
    _In_ const SoftwareBitmap& softwareBitmap,
    _In_ const ImageTensorDescription& tensorDesc,
    _In_ const BitmapBounds& inputBounds,
    _Inout_ void* pCPUTensor) {
  assert(softwareBitmap != nullptr);

  ConvertVideoFrameWithSoftwareBitmapToCPUTensorTelemetryEvent telemetrylogger(tensorDesc);

  auto height = softwareBitmap.PixelHeight();
  auto width = softwareBitmap.PixelWidth();
  auto format = softwareBitmap.BitmapPixelFormat();

  // Validate input description
  WINML_THROW_HR_IF_FALSE_MSG(
      E_INVALIDARG,
      format == BitmapPixelFormat::Bgra8 || format == BitmapPixelFormat::Rgba8 || format == BitmapPixelFormat::Gray8,
      "Format was input image %d. Input image format must Bgra8, Rgba8 or Gray8.",
      format);
  WINML_THROW_HR_IF_FALSE_MSG(E_INVALIDARG, height > 0, "Invalid input image height provided. Height is set to zero.");
  WINML_THROW_HR_IF_FALSE_MSG(E_INVALIDARG, width > 0, "Invalid input image width provided. Height is set to zero.");

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
  WINML_THROW_HR_IF_FALSE_MSG(E_INVALIDARG, tensorDesc.sizes[2] == (UINT)inputBounds.Height, "Target tensor height (%lld) does not match input height (%d).", tensorDesc.sizes[2], (UINT)inputBounds.Height);
  WINML_THROW_HR_IF_FALSE_MSG(E_INVALIDARG, tensorDesc.sizes[3] == (UINT)inputBounds.Width, "Target tensor width (%lld) does not match input width (%d).", tensorDesc.sizes[3], (UINT)inputBounds.Width);

  // get the byte buffer out of a softwarebitmap
  BYTE* pData = nullptr;
  UINT32 bufferSize = 0;
  winrt::Windows::Graphics::Imaging::BitmapBuffer spBitmapBuffer(softwareBitmap.LockBuffer(winrt::Windows::Graphics::Imaging::BitmapBufferAccessMode::Read));
  winrt::Windows::Foundation::IMemoryBufferReference reference = spBitmapBuffer.CreateReference();
  auto spByteAccess = reference.as<Windows::Foundation::IMemoryBufferByteAccess>();
  WINML_THROW_IF_FAILED(spByteAccess->GetBuffer(&pData, &bufferSize));

  UINT32 bufferWidth = bufferSize / height;

  ImageTensorChannelType channelType = ImageConversionHelpers::GetChannelTypeFromSoftwareBitmap(softwareBitmap);

  if (tensorDesc.dataType == kImageTensorDataTypeFloat32) {
    WINML_THROW_IF_FAILED(CpuTensorizer::TensorizeData<float>(channelType, tensorDesc.channelType, pData, bufferWidth, inputBounds, reinterpret_cast<float*>(pCPUTensor)));
  } else if (tensorDesc.dataType == kImageTensorDataTypeFloat16) {
    WINML_THROW_IF_FAILED(CpuTensorizer::TensorizeData<DirectX::PackedVector::HALF>(channelType, tensorDesc.channelType, pData, bufferWidth, inputBounds, reinterpret_cast<DirectX::PackedVector::HALF*>(pCPUTensor)));
  }
}