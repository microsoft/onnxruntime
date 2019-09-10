//
//  Copyright (c) Microsoft Corporation.  All rights reserved.
//

#pragma once

#include "ImageConverter.h"
#include "ImageConversionHelpers.h"
#include "ImageConversionTypes.h"

namespace Windows::AI::MachineLearning::Internal
{
    class IVideoFrameToTensorConverter
    {
    public:
        virtual HRESULT VideoFrameToDX12Tensor(
            _In_ const UINT32 batchIdx,
            _In_ winrt::Windows::AI::MachineLearning::LearningModelSession& session,
            _In_ const winrt::Windows::Media::IVideoFrame& inputVideoFrame,
            _In_ const winrt::Windows::Graphics::Imaging::BitmapBounds& inputBounds,
            _In_ const IMG_TENSOR_DESC& tensorDesc,
            _Inout_ ID3D12Resource* pOutputTensor
        ) = 0;

        virtual HRESULT VideoFrameToSoftwareTensor(
            _In_ const winrt::Windows::Media::IVideoFrame& inputVideoFrame,
            _In_ const winrt::Windows::Graphics::Imaging::BitmapBounds& inputBounds,
            _In_ const IMG_TENSOR_DESC& tensorDesc,
            _Out_ BYTE* pOutputCPUTensor
        ) = 0;
    };

    class VideoFrameToTensorConverter : IVideoFrameToTensorConverter, public ImageConverter
    {
    public:
        // Function takes in a VideoFrame backed by either a SoftwareBitmap or D3DSurface, 
        // and converts to a tensor DX12 Resource. 
        // CommandQueue and commandlist should be a compute resource, 
        // commandlist will be passed in open, closed and executing when function exits
        // User should pass in a BitmapBounds describing the region of interest, in the form of 
        // {upperleft X, upperleft Y, width, height} to be turned into a tensor.
        // If the region of interest is the entire VideoFrame, the input BitmapBounds should describe the entire image. 
        HRESULT VideoFrameToDX12Tensor(
            _In_ const UINT32 batchIdx,
            _In_ winrt::Windows::AI::MachineLearning::LearningModelSession& session,
            _In_ const winrt::Windows::Media::IVideoFrame& inputVideoFrame,
            _In_ const winrt::Windows::Graphics::Imaging::BitmapBounds& inputBounds,
            _In_ const IMG_TENSOR_DESC& tensorDesc,
            _Inout_ ID3D12Resource* pOutputTensor
        );

        // Function takes in a VideoFrame backed by either a SoftwareBitmap or D3DSurface, 
        // and converts to a tensor returned in a buffer. 
        // User should pass in a BitmapBounds describing the region of interest, in the form of 
        // {upperleft X, upperleft Y, width, height} to be turned into a tensor.
        // If the region of interest is the entire VideoFrame, the input BitmapBounds should describe the entire image. 
        HRESULT VideoFrameToSoftwareTensor(
            _In_ const winrt::Windows::Media::IVideoFrame& inputVideoFrame,
            _In_ const winrt::Windows::Graphics::Imaging::BitmapBounds& inputBounds,
            _In_ const IMG_TENSOR_DESC &tensorDesc,
            _Out_ BYTE* pOuputCPUTensor
        );

    private:
        GUID _d3d11TextureGUID = { 0x485e4bb3, 0x3fe8, 0x497b,{ 0x85, 0x9e, 0xc7, 0x5, 0x18, 0xdb, 0x11, 0x2a } };  // {485E4BB3-3FE8-497B-859E-C70518DB112A}
        GUID _handleGUID = { 0xce43264e, 0x41f7, 0x4882,{ 0x9e, 0x20, 0xfa, 0xa5, 0x1e, 0x37, 0x64, 0xfc } };; // CE43264E-41F7-4882-9E20-FAA51E3764FC
        Microsoft::WRL::ComPtr<ID3D12Resource> _spUploadHeap;
        Microsoft::WRL::ComPtr<ID3D12Resource> _spInputD3D12Resource;

        HRESULT ConvertSoftwareBitmapToGPUTensor(
            _In_ const UINT32 batchIdx,
            _In_ const winrt::Windows::Media::IVideoFrame& videoFrame,
            _In_ winrt::Windows::AI::MachineLearning::implementation::D3DDeviceCache& deviceCache,
            _In_ const winrt::Windows::Graphics::Imaging::BitmapBounds& inputBounds,
            _In_ const IMG_TENSOR_DESC& tensorDesc,
            _Inout_ ID3D12Resource* pOutputResource
        );

        HRESULT ConvertDX12TextureToGPUTensor(
            _In_ const UINT32 batchIdx,
            _In_ ID3D12Resource* pInputResource,
            _In_ winrt::Windows::AI::MachineLearning::implementation::D3DDeviceCache& deviceCache,
            _In_ const IMG_TENSOR_DESC& tensorDesc,
            _Inout_ ID3D12Resource* pOutputResource
        );

        static D3D12_UNORDERED_ACCESS_VIEW_DESC CreateUAVDescription(
            const UINT32 batchIdx,
            const D3D12_RESOURCE_DESC& resourceDesc,
            const IMG_TENSOR_DESC& desc
        );

        static HRESULT VideoFrameToTensorConverter::ConvertSoftwareBitmapToCPUTensor(
            _In_ const winrt::Windows::Graphics::Imaging::SoftwareBitmap& spSoftwareBitmap,
            _In_ const IMG_TENSOR_DESC& tensorDesc,
            _In_ const winrt::Windows::Graphics::Imaging::BitmapBounds& inputBounds,
            _Inout_ void* pCPUTensor
        );
    };
}
