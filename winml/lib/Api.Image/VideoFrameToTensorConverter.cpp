//
//  Copyright (c) Microsoft Corporation.  All rights reserved.
//

#include "pch.h"

#include <winmeta.h> // winmeta needed for TraceLoggingKeyword
#include <TraceLoggingProvider.h>
#include <telemetry\MicrosoftTelemetry.h>
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

class DX12TextureToGPUTensorTelemetryEvent
{
public:
    DX12TextureToGPUTensorTelemetryEvent(const IMG_TENSOR_DESC &tensorDesc)
    {
#ifndef WINML_TELEMETRY_DISABLED
        TraceLoggingWrite(
            g_hWinMLTraceLoggingProvider,
            "DX12TextureToGPUTensor",
            TraceLoggingKeyword(WINML_PROVIDER_KEYWORD_DEFAULT),
            TraceLoggingOpcode(EVENT_TRACE_TYPE_START),
            TraceLoggingHexInt32(tensorDesc.channelType, "Type"),
            TraceLoggingInt32(tensorDesc.sizes[2], "Height"),
            TraceLoggingInt32(tensorDesc.sizes[3], "Width")
        );
#endif
    }
    ~DX12TextureToGPUTensorTelemetryEvent()
    {
#ifndef WINML_TELEMETRY_DISABLED
        TraceLoggingWrite(
            g_hWinMLTraceLoggingProvider,
            "DX12TextureToGPUTensor",
            TraceLoggingKeyword(WINML_PROVIDER_KEYWORD_DEFAULT),
            TraceLoggingOpcode(EVENT_TRACE_TYPE_STOP),
            TraceLoggingHexInt32(S_OK, "HRESULT")
        );
#endif
    }
};

class ConvertVideoFrameWithSoftwareBitmapToCPUTensorTelemetryEvent
{
public:
    ConvertVideoFrameWithSoftwareBitmapToCPUTensorTelemetryEvent(const IMG_TENSOR_DESC &tensorDesc)
    {
#ifndef WINML_TELEMETRY_DISABLED
        TraceLoggingWrite(
            g_hWinMLTraceLoggingProvider,
            "ConvertVideoFrameWithSoftwareBitmapToCPUTensor",
            TraceLoggingKeyword(WINML_PROVIDER_KEYWORD_DEFAULT),
            TraceLoggingOpcode(EVENT_TRACE_TYPE_START),
            TraceLoggingHexInt32(tensorDesc.channelType, "Type"),
            TraceLoggingInt32(tensorDesc.sizes[2], "Height"),
            TraceLoggingInt32(tensorDesc.sizes[3], "Width")
        );
#endif
    }
    ~ConvertVideoFrameWithSoftwareBitmapToCPUTensorTelemetryEvent()
    {
#ifndef WINML_TELEMETRY_DISABLED
        TraceLoggingWrite(
            g_hWinMLTraceLoggingProvider,
            "ConvertVideoFrameWithSoftwareBitmapToCPUTensor",
            TraceLoggingKeyword(WINML_PROVIDER_KEYWORD_DEFAULT),
            TraceLoggingOpcode(EVENT_TRACE_TYPE_STOP),
            TraceLoggingHexInt32(S_OK, "HRESULT")
        );
#endif
    }
};

HRESULT VideoFrameToTensorConverter::VideoFrameToSoftwareTensor(
    _In_ const IVideoFrame& inputVideoFrame,
    _In_ const BitmapBounds& inputBounds,
    _In_ const IMG_TENSOR_DESC& tensorDesc,
    _Out_ BYTE* pOutputCPUTensor
)
{
    CWinML_AutoLock lock(&_Lock);

    winrt::Windows::Graphics::Imaging::SoftwareBitmap spInputSoftwareBitmap = inputVideoFrame.SoftwareBitmap();
    winrt::Windows::Graphics::DirectX::Direct3D11::IDirect3DSurface spInputSurface = inputVideoFrame.Direct3DSurface();

    // only one of softwarebitmap or direct3Dsurface should be non-null
    if ((spInputSoftwareBitmap == nullptr && spInputSurface == nullptr) || (spInputSoftwareBitmap != nullptr && spInputSurface != nullptr))
    {
        THROW_IF_FAILED(E_INVALIDARG);
    }


    UINT32 tensorHeight = tensorDesc.sizes[2];
    UINT32 tensorWidth = tensorDesc.sizes[3];
    if (spInputSurface || ImageConversionHelpers::NeedsVideoFrameConversion(inputVideoFrame, {}, inputBounds, tensorWidth, tensorHeight))
    {
        if (_spConvertedVideoFrame == nullptr ||
            ImageConversionHelpers::NeedsVideoFrameConversion(_spConvertedVideoFrame, {}, { 0, 0, (UINT32)tensorWidth, (UINT32)tensorHeight }, tensorWidth, tensorHeight))
        {
            _spConvertedVideoFrame = VideoFrame::CreateWithSoftwareBitmap(SoftwareBitmap(BitmapPixelFormat::Bgra8, tensorWidth, tensorHeight));
        }

        // Resize the input VideoFrame to _spConvertedVideoFrame
        THROW_IF_FAILED(ImageConversionHelpers::ConvertVideoFrameToVideoFrame(
            inputVideoFrame,
            inputBounds,
            tensorWidth,
            tensorHeight,
            _spConvertedVideoFrame));

        THROW_IF_FAILED(ConvertSoftwareBitmapToCPUTensor(
            _spConvertedVideoFrame.SoftwareBitmap(),
            tensorDesc,
            { 0, 0, (UINT32)tensorWidth, (UINT32)tensorHeight },
            pOutputCPUTensor));
    }
    else
    {
        THROW_IF_FAILED(ConvertSoftwareBitmapToCPUTensor(
            inputVideoFrame.SoftwareBitmap(),
            tensorDesc,
            inputBounds,
            pOutputCPUTensor));
    }

    return S_OK;
}

HRESULT VideoFrameToTensorConverter::VideoFrameToDX12Tensor(
    _In_ const UINT32 batchIdx,
    _In_ winrt::Windows::AI::MachineLearning::LearningModelSession& session,
    _In_ const IVideoFrame& inputVideoFrame,
    _In_ const BitmapBounds& inputBounds,
    _In_ const IMG_TENSOR_DESC& tensorDesc,
    _Inout_ ID3D12Resource* pOutputTensor
)
{
    // Validate Tensor description
    WINML_THROW_HR_IF_FALSE_MSG(E_INVALIDARG, tensorDesc.dataType == IMG_TENSOR_DATA_TYPE_FLOAT32 || tensorDesc.dataType == IMG_TENSOR_DATA_TYPE_FLOAT16, "Target tensor description must either be IMG_TENSOR_DATA_TYPE_FLOAT32, or IMG_TENSOR_DATA_TYPE_FLOAT16. %d was supplied.", tensorDesc.dataType);
    WINML_THROW_HR_IF_FALSE_MSG(E_INVALIDARG, tensorDesc.channelType != IMG_TENSOR_CHANNEL_TYPE_RGB_8 || tensorDesc.sizes[1] == 3, "Target tensor description expects IMG_TENSOR_CHANNEL_TYPE_RGB_8, but has %d channels specified instead of 3.", tensorDesc.sizes[1]);
    WINML_THROW_HR_IF_FALSE_MSG(E_INVALIDARG, tensorDesc.channelType != IMG_TENSOR_CHANNEL_TYPE_BGR_8 || tensorDesc.sizes[1] == 3, "Target tensor description expects IMG_TENSOR_CHANNEL_TYPE_BGR_8, but has %d channels specified instead of 3.", tensorDesc.sizes[1]);
    WINML_THROW_HR_IF_FALSE_MSG(E_INVALIDARG, tensorDesc.channelType != IMG_TENSOR_CHANNEL_TYPE_GRAY_8 || tensorDesc.sizes[1] == 1, "Target tensor description expects IMG_TENSOR_CHANNEL_TYPE_GRAY_8, but has %d channels specified instead of 1.", tensorDesc.sizes[1]);

    CWinML_AutoLock lock(&_Lock);
    auto device = session.Device().as<LearningModelDevice>();
    D3DDeviceCache* pDeviceCache = device->GetD3DDeviceCache();
    IDirect3DSurface spDirect3DSurface = inputVideoFrame.Direct3DSurface();

    if (inputVideoFrame.SoftwareBitmap())
    {
        WINML_THROW_IF_FAILED(ConvertSoftwareBitmapToGPUTensor(batchIdx, inputVideoFrame, *pDeviceCache, inputBounds, tensorDesc, pOutputTensor));
    }
    else if (spDirect3DSurface)
    {
        ComPtr<ID3D11Texture2D> spVideoFrameTexture;
        BitmapBounds scaledBounds = inputBounds;

        // TODO: Scale during the tensorization phase instead of using the video frame pipeline when the input bounds are not the same size as the tensor
        if (!ImageConversionHelpers::DirectXPixelFormatSupported(spDirect3DSurface.Description().Format)
            || static_cast<UINT>(inputBounds.Width) != tensorDesc.sizes[3] || static_cast<UINT>(inputBounds.Height) != tensorDesc.sizes[2])
        {
            // Force the VideoFrame to not do a conversion if the format is supported since we do it during the tensorization anyway
            DirectXPixelFormat newFormat = ImageConversionHelpers::DirectXPixelFormatSupported(spDirect3DSurface.Description().Format)
                ? spDirect3DSurface.Description().Format
                : ImageConversionHelpers::GetDirectXPixelFormatFromChannelType(tensorDesc.channelType);

            // Change the input bounds since the video frame pipeline already cropped the texture
            scaledBounds = { 0, 0, tensorDesc.sizes[3], tensorDesc.sizes[2] };

            // Use the Video Frame pipeline if we don't have our own converter for this color format
            WINML_THROW_IF_FAILED(CreateTextureFromUnsupportedColorFormat(inputVideoFrame, inputBounds, scaledBounds, newFormat, &spVideoFrameTexture));
        }
        else
        {
            // If the color format is known or the input widths are not smaller than the tensor desc, just use the video frame as is
            WINML_THROW_IF_FAILED(ImageConversionHelpers::GetTextureFromDirect3DSurface(spDirect3DSurface, &spVideoFrameTexture));
        }

        D3D11_TEXTURE2D_DESC videoFrameTextureDesc;
        spVideoFrameTexture->GetDesc(&videoFrameTextureDesc);

        if (ImageConversionHelpers::TextureIsOnDevice(spVideoFrameTexture.Get(), pDeviceCache->GetD3D11Device()))
        {
            // The texture is on our device, so we can just create own texture, share it and cache it
            if (!_spD3D11CachedTexture)
            {
                WINML_THROW_IF_FAILED(pDeviceCache->GetD3D11Device()->CreateTexture2D(&videoFrameTextureDesc, nullptr, &_spD3D11CachedTexture));
                WINML_THROW_IF_FAILED(ShareD3D11Texture(_spD3D11CachedTexture.Get(), pDeviceCache->GetD3D12Device(), &_spInputD3D12Resource));
            }
            else
            {
                D3D11_TEXTURE2D_DESC cachedTextureDesc;
                _spD3D11CachedTexture->GetDesc(&cachedTextureDesc);

                if (cachedTextureDesc.Width != scaledBounds.Width || cachedTextureDesc.Height != scaledBounds.Height || cachedTextureDesc.Format != videoFrameTextureDesc.Format)
                {
                    // The dimensions or format don't match, so we need to re-create our texture
                    WINML_THROW_IF_FAILED(pDeviceCache->GetD3D11Device()->CreateTexture2D(&videoFrameTextureDesc, nullptr, &_spD3D11CachedTexture));
                    WINML_THROW_IF_FAILED(ShareD3D11Texture(_spD3D11CachedTexture.Get(), pDeviceCache->GetD3D12Device(), &_spInputD3D12Resource));
                }
            }

            WINML_THROW_IF_FAILED(CopyTextureIntoTexture(spVideoFrameTexture.Get(), scaledBounds, _spD3D11CachedTexture.Get()));
        }
        else
        {
            // We are not on the same device, so we can't rely on our cached texture
            ComPtr<ID3D11Device> spTextureDevice;
            spVideoFrameTexture->GetDevice(&spTextureDevice);

            ComPtr<ID3D11Texture2D> spSharedD3D11Texture;
            HANDLE sharedHandle = nullptr;
            UINT comPtrSize = static_cast<UINT>(sizeof(spSharedD3D11Texture.GetAddressOf()));
            UINT handleSize = static_cast<UINT>(sizeof(sharedHandle));

            if ((FAILED(spVideoFrameTexture->GetPrivateData(_d3d11TextureGUID, &comPtrSize, spSharedD3D11Texture.GetAddressOf())) || !spSharedD3D11Texture.Get())
                || (FAILED(spVideoFrameTexture->GetPrivateData(_handleGUID, &handleSize, &sharedHandle)) || sharedHandle != _sharedHandle))
            {
                // Create a new shared texture that we cache on the video frame texture
                WINML_THROW_IF_FAILED(spTextureDevice->CreateTexture2D(&videoFrameTextureDesc, nullptr, &spSharedD3D11Texture));

                WINML_THROW_IF_FAILED(ShareD3D11Texture(spSharedD3D11Texture.Get(), pDeviceCache->GetD3D12Device(), &_spInputD3D12Resource));

                // Cache the shared texture on the video frame texture in order to tie their lifetime together
                WINML_THROW_IF_FAILED(spVideoFrameTexture->SetPrivateDataInterface(_d3d11TextureGUID, spSharedD3D11Texture.Get()));
                WINML_THROW_IF_FAILED(spVideoFrameTexture->SetPrivateData(_handleGUID, sizeof(_sharedHandle), &_sharedHandle));
            }

            // Copy from the video frame texture to the shared texture
            WINML_THROW_IF_FAILED(CopyTextureIntoTexture(spVideoFrameTexture.Get(), scaledBounds, spSharedD3D11Texture.Get()));
        }

        // Sync to make sure that the D3D11 texture is done copying
        WINML_THROW_IF_FAILED(SyncD3D11ToD3D12(*pDeviceCache, spVideoFrameTexture.Get()));

        // We cropped the texture, shared it and converted it to a known color format, so it's time to tensorize
        // TODO: merge all videoframes to a single DX12Texture Resource before call ConvertDX12TextureToGPUTensor.
        WINML_THROW_IF_FAILED(ConvertDX12TextureToGPUTensor(batchIdx, _spInputD3D12Resource.Get(), *pDeviceCache, tensorDesc, pOutputTensor));
    }
    else
    {
        // Invalid video frame
        WINML_THROW_IF_FAILED(E_INVALIDARG);
    }

    return S_OK;
}

HRESULT VideoFrameToTensorConverter::ConvertDX12TextureToGPUTensor(
    _In_ UINT32 batchIdx,
    _In_ ID3D12Resource* pInputResource,
    _In_ winrt::Windows::AI::MachineLearning::implementation::D3DDeviceCache& deviceCache,
    _In_ const IMG_TENSOR_DESC& tensorDesc,
    _Inout_ ID3D12Resource* pOutputResource
)
{
    assert(pInputResource != nullptr);
    assert(pOutputResource != nullptr);

    CWinML_AutoLock lock(&_Lock);
    D3D12_RESOURCE_DESC inputDesc = pInputResource->GetDesc();
    D3D12_RESOURCE_DESC outputDesc = pOutputResource->GetDesc();
    ComPtr<ID3D12Device> spDx12Device = deviceCache.GetD3D12Device();

    DX12TextureToGPUTensorTelemetryEvent telemetrylogger(tensorDesc);

    // Validate input description
    WINML_THROW_HR_IF_FALSE_MSG(
        E_INVALIDARG,
        inputDesc.Format == DXGI_FORMAT_B8G8R8X8_UNORM || inputDesc.Format == DXGI_FORMAT_B8G8R8A8_UNORM || inputDesc.Format == DXGI_FORMAT_R8G8B8A8_UNORM || inputDesc.Format == DXGI_FORMAT_R8_UNORM,
        "Format was input image %d. Input image format must Bgra8, Rgba8 or Gray8.",
        inputDesc.Format
    );

    WINML_THROW_HR_IF_FALSE_MSG(E_INVALIDARG, inputDesc.Width != 0, "Invalid input image height provided. Width is set to zero.");
    WINML_THROW_HR_IF_FALSE_MSG(E_INVALIDARG, inputDesc.Height != 0, "Invalid input image height provided. Height is set to zero.");

    // Validate Tensor description
    WINML_THROW_HR_IF_FALSE_MSG(E_INVALIDARG, tensorDesc.dataType == IMG_TENSOR_DATA_TYPE_FLOAT32 || tensorDesc.dataType == IMG_TENSOR_DATA_TYPE_FLOAT16, "Target tensor description must either be IMG_TENSOR_DATA_TYPE_FLOAT32, or IMG_TENSOR_DATA_TYPE_FLOAT16. %d was supplied.", tensorDesc.dataType);
    WINML_THROW_HR_IF_FALSE_MSG(E_INVALIDARG, tensorDesc.channelType != IMG_TENSOR_CHANNEL_TYPE_RGB_8 || tensorDesc.sizes[1] == 3, "Target tensor description expects IMG_TENSOR_CHANNEL_TYPE_RGB_8, but has %d channels specified instead of 3.", tensorDesc.sizes[1]);
    WINML_THROW_HR_IF_FALSE_MSG(E_INVALIDARG, tensorDesc.channelType != IMG_TENSOR_CHANNEL_TYPE_BGR_8 || tensorDesc.sizes[1] == 3, "Target tensor description expects IMG_TENSOR_CHANNEL_TYPE_BGR_8, but has %d channels specified instead of 3.", tensorDesc.sizes[1]);
    WINML_THROW_HR_IF_FALSE_MSG(E_INVALIDARG, tensorDesc.channelType != IMG_TENSOR_CHANNEL_TYPE_GRAY_8 || tensorDesc.sizes[1] == 1, "Target tensor description expects IMG_TENSOR_CHANNEL_TYPE_GRAY_8, but has %d channels specified instead of 1.", tensorDesc.sizes[1]);
    WINML_THROW_HR_IF_FALSE_MSG(E_INVALIDARG, tensorDesc.sizes[2] == inputDesc.Height, "Target tensor height (%d) does not match input height (%d).", tensorDesc.sizes[2], inputDesc.Height);
    WINML_THROW_HR_IF_FALSE_MSG(E_INVALIDARG, tensorDesc.sizes[3] == (UINT)inputDesc.Width, "Target tensor width (%d) does not match input width (%d).", tensorDesc.sizes[3], (UINT)inputDesc.Width);

    UINT uiTensorElementSize = tensorDesc.dataType == IMG_TENSOR_DATA_TYPE_FLOAT32 ? sizeof(FLOAT) : sizeof(uint16_t);

    // Validate Tensor Resource
    {
        D3D12_HEAP_PROPERTIES outputHeapProperties;
        D3D12_HEAP_FLAGS outputHeapFlags;

        WINML_THROW_IF_FAILED(pOutputResource->GetHeapProperties(&outputHeapProperties, &outputHeapFlags));

        UINT64 ullNumElementsTensor = 1;
        for (UINT uiIdx = 0; uiIdx < IMG_TENSOR_DIMENSION_COUNT_MAX; uiIdx++)
        {
            WINML_THROW_IF_FAILED(ULongLongMult(ullNumElementsTensor, tensorDesc.sizes[uiIdx], &ullNumElementsTensor));
        }
        if (ullNumElementsTensor > UINT_MAX)
        {
            WINML_THROW_IF_FAILED(E_INVALIDARG);
        }

        UINT64 ullTensorSize = 0;
        WINML_THROW_IF_FAILED(ULongLongMult(ullNumElementsTensor, uiTensorElementSize, &ullTensorSize));

        if (outputDesc.Width < ullTensorSize ||
            outputDesc.Height != 1 ||
            outputDesc.Dimension != D3D12_RESOURCE_DIMENSION_BUFFER ||
            !(outputDesc.Flags & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS) ||
            outputHeapProperties.Type != D3D12_HEAP_TYPE_DEFAULT)
        {
            WINML_THROW_IF_FAILED(E_INVALIDARG);
        }
    }

    {
        ComPtr<ID3D12Device> spDx12DeviceIn, spDx12DeviceOut;
        WINML_THROW_IF_FAILED(pInputResource->GetDevice(IID_PPV_ARGS(&spDx12DeviceIn)));
        WINML_THROW_IF_FAILED(pOutputResource->GetDevice(IID_PPV_ARGS(&spDx12DeviceOut)));

        if (spDx12Device != spDx12DeviceIn || spDx12Device != spDx12DeviceOut)
        {
            // Both input and output should have the same device
            WINML_THROW_IF_FAILED(E_INVALIDARG);
        }
    }

    // Create descriptor heaps.
    UINT srvUavDescriptorSize = spDx12Device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

    if (_spDescriptorHeap == nullptr)
    {
        // Describe and create a shader resource view (SRV) and unordered access view (UAV) descriptor heap.
        D3D12_DESCRIPTOR_HEAP_DESC srvUavHeapDesc = {};
        srvUavHeapDesc.NumDescriptors = DescriptorCount;
        srvUavHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        srvUavHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
        WINML_THROW_IF_FAILED(spDx12Device->CreateDescriptorHeap(&srvUavHeapDesc, IID_PPV_ARGS(&_spDescriptorHeap)));
        _spDescriptorHeap->SetName(L"Tensorize Descriptor Heap");
    }

    // Create SRV and UAV for input and output respectively
    {
        D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
        srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        srvDesc.Format = inputDesc.Format;
        srvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
        srvDesc.Texture2D.MipLevels = 1;
        CD3DX12_CPU_DESCRIPTOR_HANDLE srvHandle(_spDescriptorHeap->GetCPUDescriptorHandleForHeapStart(), SrvBufferIdx, srvUavDescriptorSize);
        spDx12Device->CreateShaderResourceView(pInputResource, &srvDesc, srvHandle);

        D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc = CreateUAVDescription(batchIdx, outputDesc, tensorDesc);
        CD3DX12_CPU_DESCRIPTOR_HANDLE uavHandle(_spDescriptorHeap->GetCPUDescriptorHandleForHeapStart(), UavBufferIdx, srvUavDescriptorSize);
        spDx12Device->CreateUnorderedAccessView(pOutputResource, nullptr, &uavDesc, uavHandle);
    }

    //
    // Pipeline setup for shader operation
    //
    PipelineStateCacheType type = PipelineStateCacheType::Float32;
    if (tensorDesc.dataType == IMG_TENSOR_DATA_TYPE_FLOAT16)
    {
        type = PipelineStateCacheType::Float16;
    }

    // Set the origin format
    PipelineStateCacheFormat formatFrom = PipelineStateCacheFormat::BGR8;
    if (inputDesc.Format == DXGI_FORMAT_R8G8B8A8_UNORM)
    {
        formatFrom = PipelineStateCacheFormat::RGB8;
    }
    else if (inputDesc.Format == DXGI_FORMAT_R8_UNORM)
    {
        formatFrom = PipelineStateCacheFormat::GRAY8;
    }

    // Set the destination format
    PipelineStateCacheFormat formatTo = PipelineStateCacheFormat::BGR8;
    if (tensorDesc.channelType == IMG_TENSOR_CHANNEL_TYPE_RGB_8)
    {
        formatTo = PipelineStateCacheFormat::RGB8;
    }
    else if (tensorDesc.channelType == IMG_TENSOR_CHANNEL_TYPE_GRAY_8)
    {
        formatTo = PipelineStateCacheFormat::GRAY8;
    }

    _spRootSignature = deviceCache.GetTensorizeRootSignature();
    _spPipelineState = deviceCache.GetCachedPipelineState(type, formatFrom, formatTo, PipelineStateCacheOperation::Tensorize);

    ResetCommandList(deviceCache);

    // Write compute commands into the command list and put it into the queue.
    {
        _spCommandList->SetComputeRootSignature(_spRootSignature.Get());

        ID3D12DescriptorHeap* ppHeaps[] = { _spDescriptorHeap.Get() };
        _spCommandList->SetDescriptorHeaps(_countof(ppHeaps), ppHeaps);

        CD3DX12_GPU_DESCRIPTOR_HANDLE srvHandle(_spDescriptorHeap->GetGPUDescriptorHandleForHeapStart(), SrvBufferIdx, srvUavDescriptorSize);
        CD3DX12_GPU_DESCRIPTOR_HANDLE uavHandle(_spDescriptorHeap->GetGPUDescriptorHandleForHeapStart(), UavBufferIdx, srvUavDescriptorSize);
        {
            ConstantBufferCS constantBufferCS = {};
            constantBufferCS.Height = inputDesc.Height;
            constantBufferCS.Width = (UINT)inputDesc.Width;
            _spCommandList->SetComputeRoot32BitConstants(0, 2, &constantBufferCS, 0);
        }
        _spCommandList->SetComputeRootDescriptorTable(1, srvHandle);
        _spCommandList->SetComputeRootDescriptorTable(2, uavHandle);

        UINT64 dispatchWidth = (inputDesc.Width - 1) / 16 + 1;
        UINT64 dispatchHeight = (inputDesc.Height - 1) / 4 + 1;
        _spCommandList->Dispatch(static_cast<uint32_t>(dispatchWidth), static_cast<uint32_t>(dispatchHeight), 1);

        WINML_THROW_IF_FAILED(_spCommandList->Close());

        ID3D12CommandList* pComputeToGPUCLs[] = { _spCommandList.Get() };

        deviceCache.GetCommandQueue()->ExecuteCommandLists(ARRAYSIZE(pComputeToGPUCLs), pComputeToGPUCLs);
    }

    return S_OK;
}

HRESULT VideoFrameToTensorConverter::ConvertSoftwareBitmapToGPUTensor(
    _In_ UINT32 batchIdx,
    _In_ const IVideoFrame& videoFrame,
    _In_ D3DDeviceCache& deviceCache,
    _In_ const BitmapBounds& inputBounds,
    _In_ const IMG_TENSOR_DESC& tensorDesc,
    _Inout_ ID3D12Resource* pOutputResource
)
{
    assert(pOutputResource != nullptr);
    assert(videoFrame.SoftwareBitmap() != nullptr);

    DX12TextureToGPUTensorTelemetryEvent telemetrylogger(tensorDesc);

    SoftwareBitmap convertedSoftwareBitmap = nullptr;
    BitmapBounds scaledBounds = inputBounds;

    // TODO: Scale during the tensorization phase instead of using the video frame pipeline when the input bounds are not the same size as the tensor
    if (static_cast<UINT>(inputBounds.Width) != tensorDesc.sizes[3] || static_cast<UINT>(inputBounds.Height) != tensorDesc.sizes[2])
    {
        scaledBounds = { 0, 0, tensorDesc.sizes[3], tensorDesc.sizes[2] };

        // Force the VideoFrame to not do a conversion if the format is supported since we do it during the tensorization anyway
        BitmapPixelFormat newPixelFormat = ImageConversionHelpers::SoftwareBitmapFormatSupported(videoFrame.SoftwareBitmap())
            ? videoFrame.SoftwareBitmap().BitmapPixelFormat()
            : ImageConversionHelpers::GetBitmapPixelFormatFromChannelType(tensorDesc.channelType);

        convertedSoftwareBitmap = SoftwareBitmap(newPixelFormat, tensorDesc.sizes[3], tensorDesc.sizes[2]);
        VideoFrame convertedVideoFrame = VideoFrame::CreateWithSoftwareBitmap(convertedSoftwareBitmap);
        videoFrame.as<IVideoFrame2>().CopyToAsync(convertedVideoFrame, inputBounds, scaledBounds).get();

        convertedSoftwareBitmap = convertedVideoFrame.SoftwareBitmap();
    }
    else if (!ImageConversionHelpers::SoftwareBitmapFormatSupported(videoFrame.SoftwareBitmap()))
    {
        convertedSoftwareBitmap = SoftwareBitmap::Convert(videoFrame.SoftwareBitmap(), ImageConversionHelpers::GetBitmapPixelFormatFromChannelType(tensorDesc.channelType));
    }
    else
    {
        // We don't need a conversion
        convertedSoftwareBitmap = videoFrame.SoftwareBitmap();
    }

    assert(convertedSoftwareBitmap != nullptr);

    D3D12_RESOURCE_DESC outputDesc = pOutputResource->GetDesc();

    uint32_t tensorElementSize = tensorDesc.dataType == IMG_TENSOR_DATA_TYPE_FLOAT32 ? 4 : 2;
    uint32_t bufferSize = tensorDesc.sizes[1] * tensorDesc.sizes[2] * tensorDesc.sizes[3] * tensorElementSize;

    // TODO: Make an allocator for upload heaps
    if (!_spUploadHeap || _spUploadHeap->GetDesc().Width < bufferSize)
    {
        THROW_IF_FAILED(deviceCache.GetD3D12Device()->CreateCommittedResource(
            &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD),
            D3D12_HEAP_FLAG_NONE,
            &CD3DX12_RESOURCE_DESC::Buffer(bufferSize),
            D3D12_RESOURCE_STATE_GENERIC_READ,
            nullptr,
            IID_PPV_ARGS(&_spUploadHeap)
        ));
    }

    void* pCPUTensorBuffer = nullptr;
    THROW_IF_FAILED(_spUploadHeap->Map(0, &CD3DX12_RANGE(0, 0), &pCPUTensorBuffer));

    // We avoid the Video Frame pipeline by manually sending the CPU data to the GPU, and we tensorize while we are filling the
    // upload heap. The image may already have been cropped/scaled by the video frame pipeline, so we send the scaled bounds
    // instead of the initial input bounds
    THROW_IF_FAILED(ConvertSoftwareBitmapToCPUTensor(convertedSoftwareBitmap, tensorDesc, scaledBounds, pCPUTensorBuffer));

    _spUploadHeap->Unmap(0, &CD3DX12_RANGE(0, bufferSize));

    ResetCommandList(deviceCache);
    _spCommandList->CopyBufferRegion(pOutputResource, bufferSize * batchIdx, _spUploadHeap.Get(), 0, bufferSize);

    THROW_IF_FAILED(_spCommandList->Close());
    ID3D12CommandList* ppCommandLists[] = { _spCommandList.Get() };
    deviceCache.GetCommandQueue()->ExecuteCommandLists(_countof(ppCommandLists), ppCommandLists);

    return S_OK;
}

D3D12_UNORDERED_ACCESS_VIEW_DESC VideoFrameToTensorConverter::CreateUAVDescription(
    const UINT32 batchIdx,
    const D3D12_RESOURCE_DESC& resourceDesc,
    const IMG_TENSOR_DESC& desc
)
{
    UINT uiTensorElementSize =
        desc.dataType == IMG_TENSOR_DATA_TYPE_FLOAT32 ?
        sizeof(UINT) :
        sizeof(uint16_t);

    D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
    uavDesc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
    UINT singleImageSize = desc.sizes[1] * desc.sizes[2] * desc.sizes[3];
    uavDesc.Buffer.FirstElement = batchIdx * desc.sizes[1] * desc.sizes[2] * desc.sizes[3];
    uavDesc.Buffer.NumElements = singleImageSize;
    uavDesc.Buffer.CounterOffsetInBytes = 0;
    uavDesc.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_NONE;

    if (desc.dataType == IMG_TENSOR_DATA_TYPE_FLOAT32)
    {
        // fp32 uses structured buffers so the format can be set to unknown,
        // and the stride needs to be set.
        uavDesc.Format = DXGI_FORMAT_UNKNOWN;
        uavDesc.Buffer.StructureByteStride = uiTensorElementSize;
    }
    else if (desc.dataType == IMG_TENSOR_DATA_TYPE_FLOAT16)
    {
        // fp16 uses unstructured buffers because structured buffers dont support fp16 on 
        // most hardware. The format can be set to unknown to a specific known format,
        // and the stride must be zeroed.
        uavDesc.Format = DXGI_FORMAT_R16_FLOAT;
        uavDesc.Buffer.StructureByteStride = 0;
    }
    else
    {
        WINML_THROW_HR_IF_FALSE_MSG(
            E_INVALIDARG,
            false,
            "Tensorization conversion is only supported to IMG_TENSOR_DATA_TYPE_FLOAT32, or IMG_TENSOR_DATA_TYPE_FLOAT16.");
    }

    return uavDesc;
}

HRESULT VideoFrameToTensorConverter::ConvertSoftwareBitmapToCPUTensor(
    _In_ const SoftwareBitmap& softwareBitmap,
    _In_ const IMG_TENSOR_DESC& tensorDesc,
    _In_ const BitmapBounds& inputBounds,
    _Inout_ void* pCPUTensor
)
{
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
    WINML_THROW_HR_IF_FALSE_MSG(E_INVALIDARG, tensorDesc.dataType == IMG_TENSOR_DATA_TYPE_FLOAT32 || tensorDesc.dataType == IMG_TENSOR_DATA_TYPE_FLOAT16, "Target tensor description must either be IMG_TENSOR_DATA_TYPE_FLOAT32, or IMG_TENSOR_DATA_TYPE_FLOAT16. %d was supplied.", tensorDesc.dataType);
    WINML_THROW_HR_IF_FALSE_MSG(E_INVALIDARG, tensorDesc.channelType != IMG_TENSOR_CHANNEL_TYPE_RGB_8 || tensorDesc.sizes[1] == 3, "Target tensor description expects IMG_TENSOR_CHANNEL_TYPE_RGB_8, but has %d channels specified instead of 3.", tensorDesc.sizes[1]);
    WINML_THROW_HR_IF_FALSE_MSG(E_INVALIDARG, tensorDesc.channelType != IMG_TENSOR_CHANNEL_TYPE_BGR_8 || tensorDesc.sizes[1] == 3, "Target tensor description expects IMG_TENSOR_CHANNEL_TYPE_BGR_8, but has %d channels specified instead of 3.", tensorDesc.sizes[1]);
    WINML_THROW_HR_IF_FALSE_MSG(E_INVALIDARG, tensorDesc.channelType != IMG_TENSOR_CHANNEL_TYPE_GRAY_8 || tensorDesc.sizes[1] == 1, "Target tensor description expects IMG_TENSOR_CHANNEL_TYPE_GRAY_8, but has %d channels specified instead of 1.", tensorDesc.sizes[1]);
    WINML_THROW_HR_IF_FALSE_MSG(
        E_INVALIDARG,
        tensorDesc.channelType == IMG_TENSOR_CHANNEL_TYPE_GRAY_8 ||
        tensorDesc.channelType == IMG_TENSOR_CHANNEL_TYPE_BGR_8 ||
        tensorDesc.channelType == IMG_TENSOR_CHANNEL_TYPE_RGB_8,
        "Target tensor description expects IMG_TENSOR_CHANNEL_TYPE_GRAY_8, IMG_TENSOR_CHANNEL_TYPE_BGR_8, or IMG_TENSOR_CHANNEL_TYPE_RGB_8 but has %d was specified.",
        tensorDesc.channelType);
     WINML_THROW_HR_IF_FALSE_MSG(E_INVALIDARG, tensorDesc.sizes[2] == (UINT)inputBounds.Height, "Target tensor height (%d) does not match input height (%d).", tensorDesc.sizes[2], (UINT)inputBounds.Height);
     WINML_THROW_HR_IF_FALSE_MSG(E_INVALIDARG, tensorDesc.sizes[3] == (UINT)inputBounds.Width, "Target tensor width (%d) does not match input width (%d).", tensorDesc.sizes[3], (UINT)inputBounds.Width);

    // get the byte buffer out of a softwarebitmap
    BYTE* pData = nullptr;
    UINT32 bufferSize = 0;
    winrt::Windows::Graphics::Imaging::BitmapBuffer spBitmapBuffer(softwareBitmap.LockBuffer(winrt::Windows::Graphics::Imaging::BitmapBufferAccessMode::Read));
    winrt::Windows::Foundation::IMemoryBufferReference reference = spBitmapBuffer.CreateReference();
    auto spByteAccess = reference.as<Windows::Foundation::IMemoryBufferByteAccess>();
    WINML_THROW_IF_FAILED(spByteAccess->GetBuffer(&pData, &bufferSize));

    UINT32 bufferWidth = bufferSize / height;

    IMG_TENSOR_CHANNEL_TYPE channelType = ImageConversionHelpers::GetChannelTypeFromSoftwareBitmap(softwareBitmap);

    if (tensorDesc.dataType == IMG_TENSOR_DATA_TYPE_FLOAT32)
    {
        WINML_THROW_IF_FAILED(CpuTensorizer::TensorizeData<float>(channelType, tensorDesc.channelType, pData, bufferWidth, inputBounds, reinterpret_cast<float*>(pCPUTensor)));
    }
    else if (tensorDesc.dataType == IMG_TENSOR_DATA_TYPE_FLOAT16)
    {
        WINML_THROW_IF_FAILED(CpuTensorizer::TensorizeData<DirectX::PackedVector::HALF>(channelType, tensorDesc.channelType, pData, bufferWidth, inputBounds, reinterpret_cast<DirectX::PackedVector::HALF*>(pCPUTensor)));
    }

    return S_OK;
}