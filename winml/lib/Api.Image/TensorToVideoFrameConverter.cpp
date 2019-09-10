//
//  Copyright (c) Microsoft Corporation.  All rights reserved.
//

#include "pch.h"

#include <winmeta.h> // winmeta needed for TraceLoggingKeyword
#include <TraceLoggingProvider.h>
#include <telemetry\MicrosoftTelemetry.h>
#include <evntrace.h>
#include <MemoryBuffer.h>

#include "inc/D3DDeviceCache.h"
#include "inc/TensorToVideoFrameConverter.h"
#include "CpuDetensorizer.h"

#include "LearningModelDevice.h"

using namespace Microsoft::WRL;
using namespace Windows::AI::MachineLearning::Internal;
using namespace Windows::Graphics::DirectX::Direct3D11;
using namespace winrt::Windows::Graphics::Imaging;
using namespace winrt::Windows::Graphics::DirectX::Direct3D11;
using namespace winrt::Windows::Media;
using namespace winrt::Windows::AI::MachineLearning::implementation;
using namespace winrt::Windows::Graphics::DirectX;

class GPUTensorToDX12TextureTelemetryEvent
{
public:
    GPUTensorToDX12TextureTelemetryEvent(const IMG_TENSOR_DESC &tensorDesc)
    {
#ifndef WINML_TELEMETRY_DISABLED
        TraceLoggingWrite(
            g_hWinMLTraceLoggingProvider,
            "GPUTensorToDX12Texture",
            TraceLoggingKeyword(WINML_PROVIDER_KEYWORD_DEFAULT),
            TraceLoggingOpcode(EVENT_TRACE_TYPE_START),
            TraceLoggingHexInt32(tensorDesc.channelType, "Type"),
            TraceLoggingInt32(tensorDesc.sizes[2], "Height"),
            TraceLoggingInt32(tensorDesc.sizes[3], "Width")
        );
#endif
    }
    ~GPUTensorToDX12TextureTelemetryEvent()
    {
#ifndef WINML_TELEMETRY_DISABLED
        TraceLoggingWrite(
            g_hWinMLTraceLoggingProvider,
            "GPUTensorToDX12Texture",
            TraceLoggingKeyword(WINML_PROVIDER_KEYWORD_DEFAULT),
            TraceLoggingOpcode(EVENT_TRACE_TYPE_STOP),
            TraceLoggingHexInt32(S_OK, "HRESULT")
        );
#endif
    }
};

class ConvertCPUTensorToVideoFrameWithSoftwareBitmapTelemetryEvent
{
public:
    ConvertCPUTensorToVideoFrameWithSoftwareBitmapTelemetryEvent(const IMG_TENSOR_DESC &tensorDesc)
    {
#ifndef WINML_TELEMETRY_DISABLED
        TraceLoggingWrite(
            g_hWinMLTraceLoggingProvider,
            "ConvertCPUTensorToVideoFrameWithSoftwareBitmap",
            TraceLoggingKeyword(WINML_PROVIDER_KEYWORD_DEFAULT),
            TraceLoggingOpcode(EVENT_TRACE_TYPE_START),
            TraceLoggingHexInt32(tensorDesc.channelType, "Type"),
            TraceLoggingInt32(tensorDesc.sizes[2], "Height"),
            TraceLoggingInt32(tensorDesc.sizes[3], "Width")
        );
#endif
    }
    ~ConvertCPUTensorToVideoFrameWithSoftwareBitmapTelemetryEvent()
    {
#ifndef WINML_TELEMETRY_DISABLED
        TraceLoggingWrite(
            g_hWinMLTraceLoggingProvider,
            "ConvertCPUTensorToVideoFrameWithSoftwareBitmap",
            TraceLoggingKeyword(WINML_PROVIDER_KEYWORD_DEFAULT),
            TraceLoggingOpcode(EVENT_TRACE_TYPE_STOP),
            TraceLoggingHexInt32(S_OK, "HRESULT")
        );
#endif
    }
};

HRESULT TensorToVideoFrameConverter::DX12TensorToVideoFrame(
    _In_ UINT32 batchIdx,
    _In_ winrt::Windows::AI::MachineLearning::LearningModelSession& session,
    _In_ ID3D12Resource* pInputTensor,
    _In_ const IMG_TENSOR_DESC& tensorDesc,
    _Inout_ VideoFrame& destVideoFrame
)
{
    CWinML_AutoLock lock(&_Lock);

    auto spDevice = session.Device().as<winmlp::LearningModelDevice>();
    D3DDeviceCache* pDeviceCache = spDevice->GetD3DDeviceCache();

    IDirect3DSurface spDestDirect3DSurface = destVideoFrame.Direct3DSurface();
    SoftwareBitmap softwareBitmap = destVideoFrame.SoftwareBitmap();

    if (softwareBitmap)
    {
        WINML_THROW_IF_FAILED(ConvertGPUTensorToSoftwareBitmap(batchIdx, pInputTensor, *pDeviceCache, tensorDesc, softwareBitmap));
    }
    else if (spDestDirect3DSurface)
    {
        bool isUAVSupportedFormat = ImageConversionHelpers::FormatSupportedForUAV(
            pDeviceCache->GetD3D12Device(),
            ImageConversionHelpers::GetDXGIFormatFromDirectXPixelFormat(spDestDirect3DSurface.Description().Format)
        );

        // UAV support for formats is device dependent
        if (!isUAVSupportedFormat)
        {
            WINML_THROW_IF_FAILED(ConvertDX12TensorToUnsupportedVideoFrameFormat(batchIdx, pInputTensor, *pDeviceCache, tensorDesc, destVideoFrame));
        }
        else
        {
            ComPtr<ID3D11Texture2D> spVideoFrameTexture;
            WINML_THROW_IF_FAILED(ImageConversionHelpers::GetTextureFromDirect3DSurface(destVideoFrame.Direct3DSurface(), &spVideoFrameTexture));

            D3D11_TEXTURE2D_DESC videoFrameTextureDesc;
            spVideoFrameTexture->GetDesc(&videoFrameTextureDesc);
            BitmapBounds bounds = { 0, 0, videoFrameTextureDesc.Width, videoFrameTextureDesc.Height };

            if (ImageConversionHelpers::TextureIsOnDevice(spVideoFrameTexture.Get(), pDeviceCache->GetD3D11Device()))
            {
                // The texture is on our device, so we can just create own texture, share it and cache it
                if (!_spD3D11CachedTexture)
                {
                    WINML_THROW_IF_FAILED(pDeviceCache->GetD3D11Device()->CreateTexture2D(&videoFrameTextureDesc, nullptr, &_spD3D11CachedTexture));
                    WINML_THROW_IF_FAILED(ShareD3D11Texture(_spD3D11CachedTexture.Get(), pDeviceCache->GetD3D12Device(), &_spOutputResource));
                }
                else
                {
                    D3D11_TEXTURE2D_DESC cachedTextureDesc;
                    _spD3D11CachedTexture->GetDesc(&cachedTextureDesc);

                    if (cachedTextureDesc.Width != videoFrameTextureDesc.Width || cachedTextureDesc.Height != videoFrameTextureDesc.Height || cachedTextureDesc.Format != videoFrameTextureDesc.Format)
                    {
                        // The dimensions or format don't match, so we need to re-create our texture
                        WINML_THROW_IF_FAILED(pDeviceCache->GetD3D11Device()->CreateTexture2D(&videoFrameTextureDesc, nullptr, &_spD3D11CachedTexture));
                        WINML_THROW_IF_FAILED(ShareD3D11Texture(_spD3D11CachedTexture.Get(), pDeviceCache->GetD3D12Device(), &_spOutputResource));
                    }
                }

                // Detensorize
                WINML_THROW_IF_FAILED(ConvertGPUTensorToDX12Texture(batchIdx, pInputTensor, *pDeviceCache, tensorDesc, _spOutputResource.Get()));

                // Make sure that detensorization is done
                WINML_THROW_IF_FAILED(SyncD3D12ToD3D11(*pDeviceCache, _spD3D11CachedTexture.Get()));

                // Finally, copy the detensorized texture to the user's device
                WINML_THROW_IF_FAILED(CopyTextureIntoTexture(_spD3D11CachedTexture.Get(), bounds, spVideoFrameTexture.Get()));
            }
            else
            {
                // We are not on the same device, so we can't rely on our own cached texture
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

                    WINML_THROW_IF_FAILED(ShareD3D11Texture(spSharedD3D11Texture.Get(), pDeviceCache->GetD3D12Device(), &_spOutputResource));

                    // Cache the shared texture on the video frame texture in order to tie their lifetime together
                    WINML_THROW_IF_FAILED(spVideoFrameTexture->SetPrivateDataInterface(_d3d11TextureGUID, spSharedD3D11Texture.Get()));
                    WINML_THROW_IF_FAILED(spVideoFrameTexture->SetPrivateData(_handleGUID, sizeof(_sharedHandle), &_sharedHandle));
                }

                // Detensorize
                WINML_THROW_IF_FAILED(ConvertGPUTensorToDX12Texture(batchIdx, pInputTensor, *pDeviceCache, tensorDesc, _spOutputResource.Get()));

                // Make sure that detensorization is done
                WINML_THROW_IF_FAILED(SyncD3D12ToD3D11(*pDeviceCache, spSharedD3D11Texture.Get()));

                // Finally, copy the detensorized texture to the user's device
                WINML_THROW_IF_FAILED(CopyTextureIntoTexture(spSharedD3D11Texture.Get(), bounds, spVideoFrameTexture.Get()));
            }
        }
    }
    else
    {
        // Invalid video frame
        WINML_THROW_IF_FAILED(E_INVALIDARG);
    }

    return S_OK;
}

HRESULT TensorToVideoFrameConverter::ConvertDX12TensorToUnsupportedVideoFrameFormat(
    _In_ UINT32 batchIdx,
    _In_ ID3D12Resource* pInputTensor,
    _In_ D3DDeviceCache& deviceCache,
    _In_ const IMG_TENSOR_DESC& tensorDesc,
    _Inout_ VideoFrame& unsupportedVideoFrame
)
{
    assert(pInputTensor != nullptr);

    ComPtr<ID3D11Texture2D> spVideoFrameTexture;

    // Find the first supported format and convert to it
    auto supportedFormatIter = std::find_if(
        ImageConversionHelpers::supportedWinMLFormats.begin(),
        ImageConversionHelpers::supportedWinMLFormats.end(),
        [&deviceCache](DXGI_FORMAT format) { return ImageConversionHelpers::FormatSupportedForUAV(deviceCache.GetD3D12Device(), format); }
    );

    WINML_THROW_HR_IF_FALSE_MSG(
        E_INVALIDARG,
        supportedFormatIter != ImageConversionHelpers::supportedWinMLFormats.end(),
        "Detensorization for this format is unsupported on the current device."
    );

    // TODO: (@pavignol) Optimize for batch
    _spConvertedVideoFrame = VideoFrame::CreateAsDirect3D11SurfaceBacked(
        ImageConversionHelpers::GetDirectXPixelFormatFromDXGIFormat(*supportedFormatIter),
        unsupportedVideoFrame.Direct3DSurface().Description().Width,
        unsupportedVideoFrame.Direct3DSurface().Description().Height,
        deviceCache.GetWinrtDevice()
    );

    THROW_IF_FAILED(ImageConversionHelpers::GetTextureFromDirect3DSurface(_spConvertedVideoFrame.Direct3DSurface(), &spVideoFrameTexture));
    THROW_IF_FAILED(ShareD3D11Texture(spVideoFrameTexture.Get(), deviceCache.GetD3D12Device(), &_spOutputResource));

    // Detensorize
    THROW_IF_FAILED(ConvertGPUTensorToDX12Texture(batchIdx, pInputTensor, deviceCache, tensorDesc, _spOutputResource.Get()));

    // Wait for the D3D12 work to complete before using the resource
    THROW_IF_FAILED(SyncD3D12ToD3D11(deviceCache, spVideoFrameTexture.Get()));

    // Finally, convert and copy the texture to the destination video frame
    _spConvertedVideoFrame.CopyToAsync(unsupportedVideoFrame).get();

    return S_OK;
}

HRESULT TensorToVideoFrameConverter::SoftwareTensorToVideoFrame(
    _In_ winrt::Windows::AI::MachineLearning::LearningModelSession& session,
    _In_ BYTE* pCPUTensorToConvert,
    _In_ IMG_TENSOR_DESC tensorDesc,
    _Inout_ winrt::Windows::Media::VideoFrame& pDestVideoFrame
)
{
    CWinML_AutoLock lock(&_Lock);
    winrt::Windows::Media::IVideoFrame spTensorFrame;
    UINT32 outputWidth = 0;
    UINT32 outputHeight = 0;


    UINT32 tensorHeight = tensorDesc.sizes[2];
    UINT32 tensorWidth = tensorDesc.sizes[3];
    // create a bitmap bounds for the whole image/tensor
    BitmapBounds inputBounds =
    {
        0,
        0,
        tensorWidth,
        tensorHeight
    };

    winrt::Windows::Graphics::Imaging::SoftwareBitmap spOutputSoftwareBitmap = pDestVideoFrame.SoftwareBitmap();
    winrt::Windows::Graphics::DirectX::Direct3D11::IDirect3DSurface spOutputSurface = pDestVideoFrame.Direct3DSurface();

    // only one of softwarebitmap or direct3Dsurface should be non-null
    if ((spOutputSoftwareBitmap == nullptr && spOutputSurface == nullptr) || (spOutputSoftwareBitmap != nullptr && spOutputSurface != nullptr))
    {
        THROW_IF_FAILED(E_INVALIDARG);
    }
    if (spOutputSoftwareBitmap)
    {
        outputWidth = spOutputSoftwareBitmap.PixelWidth();
        outputHeight = spOutputSoftwareBitmap.PixelHeight();
    }
    else
    {
        Direct3DSurfaceDescription description;
        description = spOutputSurface.Description();
        outputWidth = description.Width;
        outputHeight = description.Height;
    }

    if (ImageConversionHelpers::NeedsVideoFrameConversion(pDestVideoFrame, {}, { 0, 0, (UINT32)tensorWidth, (UINT32)tensorHeight }, tensorWidth, tensorHeight))
    {
        if (_spConvertedVideoFrame == nullptr ||
            ImageConversionHelpers::NeedsVideoFrameConversion(_spConvertedVideoFrame, {}, { 0, 0, (UINT32)tensorWidth, (UINT32)tensorHeight }, tensorWidth, tensorHeight))
        {
            _spConvertedVideoFrame = VideoFrame::CreateWithSoftwareBitmap(SoftwareBitmap(BitmapPixelFormat::Bgra8, tensorWidth, tensorHeight));
        }

        spTensorFrame = _spConvertedVideoFrame;
    }
    else
    {
        spTensorFrame = pDestVideoFrame;
        _spConvertedVideoFrame = nullptr;
    }
    auto bitmap = spTensorFrame.SoftwareBitmap();
    THROW_IF_FAILED(ConvertCPUTensorToSoftwareBitmap(
        pCPUTensorToConvert,
        tensorDesc,
        bitmap
    ));

    if (_spConvertedVideoFrame)
    {
        THROW_IF_FAILED(ImageConversionHelpers::ConvertVideoFrameToVideoFrame(
            _spConvertedVideoFrame,
            inputBounds,
            outputWidth,
            outputHeight,
            pDestVideoFrame));
    }

    return S_OK;
}

HRESULT TensorToVideoFrameConverter::ConvertGPUTensorToDX12Texture(
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
    CD3DX12_VIEWPORT viewport((float)0, (float)0, (float)outputDesc.Width, (float)outputDesc.Height);
    CD3DX12_RECT scissorRect(0, 0, (LONG)outputDesc.Width, outputDesc.Height);
    ComPtr<ID3D12Device> spDx12Device = deviceCache.GetD3D12Device();

    GPUTensorToDX12TextureTelemetryEvent telemetrylogger(tensorDesc);

    WINML_THROW_HR_IF_FALSE_MSG(
        E_INVALIDARG,
        outputDesc.Format == DXGI_FORMAT_B8G8R8A8_UNORM || outputDesc.Format == DXGI_FORMAT_R8G8B8A8_UNORM || outputDesc.Format == DXGI_FORMAT_R8_UNORM,
        "Format was output image %d. Output image format must be Bgra8, Rgba8 or Gray8.",
        outputDesc.Format
    );

    // Validate input description
    WINML_THROW_HR_IF_FALSE_MSG(E_INVALIDARG, inputDesc.Height != 0, "Invalid input image height provided. Height is set to zero.");
    WINML_THROW_HR_IF_FALSE_MSG(E_INVALIDARG, inputDesc.Width != 0, "Invalid input image height provided. Height is set to zero.");

    // Validate output description
    WINML_THROW_HR_IF_FALSE_MSG(E_INVALIDARG, outputDesc.Height != 0, "Invalid input image height provided. Height is set to zero.");
    WINML_THROW_HR_IF_FALSE_MSG(E_INVALIDARG, outputDesc.Width != 0, "Invalid input image height provided. Height is set to zero.");

    // Validate Tensor description
    WINML_THROW_HR_IF_FALSE_MSG(E_INVALIDARG, tensorDesc.dataType == IMG_TENSOR_DATA_TYPE_FLOAT32 || tensorDesc.dataType == IMG_TENSOR_DATA_TYPE_FLOAT16, "Target tensor description must either be IMG_TENSOR_DATA_TYPE_FLOAT32, or IMG_TENSOR_DATA_TYPE_FLOAT16. %d was supplied.", tensorDesc.dataType);
    WINML_THROW_HR_IF_FALSE_MSG(E_INVALIDARG, tensorDesc.channelType != IMG_TENSOR_CHANNEL_TYPE_RGB_8 || tensorDesc.sizes[1] == 3, "Target tensor description expects IMG_TENSOR_CHANNEL_TYPE_RGB_8, but has %d channels specified instead of 3.", tensorDesc.sizes[1]);
    WINML_THROW_HR_IF_FALSE_MSG(E_INVALIDARG, tensorDesc.channelType != IMG_TENSOR_CHANNEL_TYPE_BGR_8 || tensorDesc.sizes[1] == 3, "Target tensor description expects IMG_TENSOR_CHANNEL_TYPE_BGR_8, but has %d channels specified instead of 3.", tensorDesc.sizes[1]);
    WINML_THROW_HR_IF_FALSE_MSG(E_INVALIDARG, tensorDesc.channelType != IMG_TENSOR_CHANNEL_TYPE_GRAY_8 || tensorDesc.sizes[1] == 1, "Target tensor description expects IMG_TENSOR_CHANNEL_TYPE_GRAY_8, but has %d channels specified instead of 1.", tensorDesc.sizes[1]);
    WINML_THROW_HR_IF_FALSE_MSG(E_INVALIDARG, tensorDesc.sizes[2] == outputDesc.Height, "Target tensor height (%d) does not match input height (%d).", tensorDesc.sizes[2], outputDesc.Height);
    WINML_THROW_HR_IF_FALSE_MSG(E_INVALIDARG, tensorDesc.sizes[3] == (UINT)outputDesc.Width, "Target tensor width (%d) does not match input width (%d).", tensorDesc.sizes[3], (UINT)outputDesc.Width);

    // Create descriptor heaps
    UINT srvUavDescriptorSize = spDx12Device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

    // Create a UAV resource for the shader
    D3D12_RESOURCE_DESC outputResourceDesc = _spOutputResource->GetDesc();
    outputResourceDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

    if (!_spUAVResource
        || outputDesc.Format != _spUAVResource->GetDesc().Format
        || outputDesc.Width != _spUAVResource->GetDesc().Width
        || outputDesc.Height != _spUAVResource->GetDesc().Height)
    {
        WINML_THROW_IF_FAILED(deviceCache.GetD3D12Device()->CreateCommittedResource(
            &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT),
            D3D12_HEAP_FLAG_NONE,
            &outputResourceDesc,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            nullptr,
            IID_PPV_ARGS(&_spUAVResource))
        );
    }

    if (_spDescriptorHeap == nullptr)
    {
        // Describe and create a shader resource view (SRV) and unordered access view (UAV) descriptor heap.
        D3D12_DESCRIPTOR_HEAP_DESC srvUavHeapDesc = {};
        srvUavHeapDesc.NumDescriptors = DescriptorCount;
        srvUavHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        srvUavHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
        WINML_THROW_IF_FAILED(spDx12Device->CreateDescriptorHeap(&srvUavHeapDesc, IID_PPV_ARGS(&_spDescriptorHeap)));
        _spDescriptorHeap->SetName(L"Detensorize Descriptor Heap");
    }

    // Create SRV and UAV for input and output respectively
    {
        D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = CreateSRVDescriptor(batchIdx, inputDesc, tensorDesc);
        CD3DX12_CPU_DESCRIPTOR_HANDLE srvHandle(_spDescriptorHeap->GetCPUDescriptorHandleForHeapStart(), SrvBufferIdx, srvUavDescriptorSize);
        spDx12Device->CreateShaderResourceView(pInputResource, &srvDesc, srvHandle);

        D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
        uavDesc.Format = outputResourceDesc.Format;
        uavDesc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
        CD3DX12_CPU_DESCRIPTOR_HANDLE uavHandle(_spDescriptorHeap->GetCPUDescriptorHandleForHeapStart(), UavBufferIdx, srvUavDescriptorSize);
        spDx12Device->CreateUnorderedAccessView(_spUAVResource.Get(), nullptr, &uavDesc, uavHandle);
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
    if (tensorDesc.channelType == IMG_TENSOR_CHANNEL_TYPE_RGB_8)
    {
        formatFrom = PipelineStateCacheFormat::RGB8;
    }
    else if (inputDesc.Format == IMG_TENSOR_CHANNEL_TYPE_GRAY_8)
    {
        formatFrom = PipelineStateCacheFormat::GRAY8;
    }

    // Set the destination format
    PipelineStateCacheFormat formatTo = PipelineStateCacheFormat::BGR8;
    if (outputDesc.Format == DXGI_FORMAT_R8G8B8A8_UNORM)
    {
        formatTo = PipelineStateCacheFormat::RGB8;
    }
    else if (outputDesc.Format == DXGI_FORMAT_R8_UNORM)
    {
        formatTo = PipelineStateCacheFormat::GRAY8;
    }

    _spRootSignature = deviceCache.GetDetensorizeRootSignature();
    _spPipelineState = deviceCache.GetCachedPipelineState(type, formatFrom, formatTo, PipelineStateCacheOperation::Detensorize);

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
            constantBufferCS.Height = tensorDesc.sizes[2];
            constantBufferCS.Width = tensorDesc.sizes[3];
            _spCommandList->SetComputeRoot32BitConstants(0, 2, &constantBufferCS, 0);
        }
        _spCommandList->SetComputeRootDescriptorTable(1, srvHandle);
        _spCommandList->SetComputeRootDescriptorTable(2, uavHandle);

        auto dispatchWidth = static_cast<UINT>((tensorDesc.sizes[3] - 1) / 16 + 1);
        auto dispatchHeight = static_cast<UINT>((tensorDesc.sizes[2] - 1) / 4 + 1);
        _spCommandList->Dispatch(dispatchWidth, dispatchHeight, 1);

        // Copy the UAV data to the output resource after detensorization
        _spCommandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(_spUAVResource.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE));
        _spCommandList->CopyResource(pOutputResource, _spUAVResource.Get());
        _spCommandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(_spUAVResource.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS));

        WINML_THROW_IF_FAILED(_spCommandList->Close());
        ID3D12CommandList* pComputeToGPUCLs[] = { _spCommandList.Get() };
        deviceCache.GetCommandQueue()->ExecuteCommandLists(ARRAYSIZE(pComputeToGPUCLs), pComputeToGPUCLs);
    }

    return S_OK;
}

HRESULT TensorToVideoFrameConverter::ConvertGPUTensorToSoftwareBitmap(
    _In_ UINT32 batchIdx,
    _In_ ID3D12Resource* pInputTensor,
    _In_ D3DDeviceCache& deviceCache,
    _In_ const IMG_TENSOR_DESC& tensorDesc,
    _Inout_ SoftwareBitmap& softwareBitmap
)
{
    assert(pInputTensor != nullptr);
    assert(softwareBitmap != nullptr);

    GPUTensorToDX12TextureTelemetryEvent telemetrylogger(tensorDesc);

    uint32_t tensorElementSize = tensorDesc.dataType == IMG_TENSOR_DATA_TYPE_FLOAT32 ? 4 : 2;
    uint32_t singleVideoFramebufferSize = tensorDesc.sizes[1] * tensorDesc.sizes[2] * tensorDesc.sizes[3] * tensorElementSize;

    // TODO: Make an allocator for readback heaps
    if (!_spReadbackHeap || _spReadbackHeap->GetDesc().Width < singleVideoFramebufferSize)
    {
        THROW_IF_FAILED(deviceCache.GetD3D12Device()->CreateCommittedResource(
            &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_READBACK),
            D3D12_HEAP_FLAG_NONE,
            &CD3DX12_RESOURCE_DESC::Buffer(singleVideoFramebufferSize),
            D3D12_RESOURCE_STATE_COPY_DEST,
            nullptr,
            IID_PPV_ARGS(&_spReadbackHeap)
        ));
    }

    ResetCommandList(deviceCache);
    _spCommandList->CopyBufferRegion(_spReadbackHeap.Get(), 0, pInputTensor, singleVideoFramebufferSize * batchIdx, singleVideoFramebufferSize);

    THROW_IF_FAILED(_spCommandList->Close());
    ID3D12CommandList* ppCommandLists[] = { _spCommandList.Get() };
    deviceCache.GetCommandQueue()->ExecuteCommandLists(_countof(ppCommandLists), ppCommandLists);

    // Sync to make sure the the heap received all the data
    deviceCache.SyncD3D12ToCPU();

    void* pCPUTensorBuffer = nullptr;
    THROW_IF_FAILED(_spReadbackHeap->Map(0, &CD3DX12_RANGE(0, singleVideoFramebufferSize), &pCPUTensorBuffer));

    // We avoid the Video Frame pipeline by manually downloading the GPU data to the CPU and detensorize while we are filling the readback heap
    THROW_IF_FAILED(ConvertCPUTensorToSoftwareBitmap(pCPUTensorBuffer, tensorDesc, softwareBitmap));

    _spReadbackHeap->Unmap(0, &CD3DX12_RANGE(0, 0));

    return S_OK;
}

D3D12_SHADER_RESOURCE_VIEW_DESC TensorToVideoFrameConverter::CreateSRVDescriptor(
    const UINT32 batchIdx,
    const D3D12_RESOURCE_DESC& resourceDesc,
    const IMG_TENSOR_DESC& desc
)
{
    UINT uiTensorElementSize =
        desc.dataType == IMG_TENSOR_DATA_TYPE_FLOAT32 ?
        sizeof(UINT) :
        sizeof(uint16_t);

    D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
    srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srvDesc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
    UINT singleImageSize = desc.sizes[1] * desc.sizes[2] * desc.sizes[3];
    srvDesc.Buffer.FirstElement = batchIdx * desc.sizes[1] * desc.sizes[2] * desc.sizes[3];
    srvDesc.Buffer.NumElements = singleImageSize;
    srvDesc.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_NONE;

    if (desc.dataType == IMG_TENSOR_DATA_TYPE_FLOAT32)
    {
        // fp32 uses structured buffers so the format can be set to unknown,
        // and the stride needs to be set.
        srvDesc.Format = resourceDesc.Format;
        srvDesc.Buffer.StructureByteStride = uiTensorElementSize;
    }
    else if (desc.dataType == IMG_TENSOR_DATA_TYPE_FLOAT16)
    {
        // fp16 uses unstructured buffers because structured buffers dont support fp16 on 
        // most hardware. The format can be set to unknown to a specific known format,
        // and the stride must be zeroed.
        srvDesc.Format = DXGI_FORMAT_R16_FLOAT;
        srvDesc.Buffer.StructureByteStride = 0;
    }
    else
    {
        WINML_THROW_HR_IF_FALSE_MSG(
            E_INVALIDARG,
            false,
            "Tensorization conversion is only supported to IMG_TENSOR_DATA_TYPE_FLOAT32, or IMG_TENSOR_DATA_TYPE_FLOAT16.");
    }

    return srvDesc;
}

HRESULT TensorToVideoFrameConverter::ConvertCPUTensorToSoftwareBitmap(
    _In_ void* pCPUTensor,
    _In_ const IMG_TENSOR_DESC& tensorDesc,
    _Inout_ SoftwareBitmap& softwareBitmap
)
{
    ConvertCPUTensorToVideoFrameWithSoftwareBitmapTelemetryEvent telemetrylogger(tensorDesc);

    auto height = softwareBitmap.PixelHeight();
    auto width = softwareBitmap.PixelWidth();
    auto format = softwareBitmap.BitmapPixelFormat();

    // Validate input description
    WINML_THROW_HR_IF_FALSE_MSG(
        E_INVALIDARG,
        format == BitmapPixelFormat::Bgra8 || format == BitmapPixelFormat::Rgba8 || format == BitmapPixelFormat::Gray8,
        "Format was input image %d. Input image format must Bgra8, Rgba8 or Gray8.",
        format);
    WINML_THROW_HR_IF_FALSE_MSG(E_INVALIDARG, height > 0, "Output input image height provided. Height is set to zero.");
    WINML_THROW_HR_IF_FALSE_MSG(E_INVALIDARG, width > 0, "Output input image width provided. Width is set to zero.");

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
    WINML_THROW_HR_IF_FALSE_MSG(E_INVALIDARG, tensorDesc.sizes[2] == (UINT)height, "Target tensor height (%d) does not match input height (%d).", tensorDesc.sizes[2], (UINT)height);
    WINML_THROW_HR_IF_FALSE_MSG(E_INVALIDARG, tensorDesc.sizes[3] == (UINT)width, "Target tensor width (%d) does not match input width (%d).", tensorDesc.sizes[3], (UINT)width);

    // get the byte buffer out of a softwarebitmap
    BYTE* pData = nullptr;
    UINT32 uiCapacity = 0;

    winrt::Windows::Graphics::Imaging::BitmapBuffer spBitmapBuffer(softwareBitmap.LockBuffer(winrt::Windows::Graphics::Imaging::BitmapBufferAccessMode::Write));
    winrt::Windows::Foundation::IMemoryBufferReference reference = spBitmapBuffer.CreateReference();
    auto spByteAccess = reference.as<Windows::Foundation::IMemoryBufferByteAccess>();
    WINML_THROW_IF_FAILED(spByteAccess->GetBuffer(&pData, &uiCapacity));

    uint32_t bufferWidth = uiCapacity / height;

    IMG_TENSOR_CHANNEL_TYPE targetChannelType = ImageConversionHelpers::GetChannelTypeFromSoftwareBitmap(softwareBitmap);

    if (tensorDesc.dataType == IMG_TENSOR_DATA_TYPE_FLOAT32)
    {
        WINML_THROW_IF_FAILED(CpuDetensorizer::Detensorize<float>(tensorDesc.channelType, targetChannelType, static_cast<float*>(pCPUTensor), bufferWidth, height, width, pData));
    }
    else if (tensorDesc.dataType == IMG_TENSOR_DATA_TYPE_FLOAT16)
    {
        WINML_THROW_IF_FAILED(CpuDetensorizer::Detensorize<DirectX::PackedVector::HALF>(tensorDesc.channelType, targetChannelType, static_cast<DirectX::PackedVector::HALF*>(pCPUTensor), bufferWidth, height, width, pData));
    }

    return S_OK;
}