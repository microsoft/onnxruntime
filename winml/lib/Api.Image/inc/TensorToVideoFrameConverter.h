//
//  Copyright (c) Microsoft Corporation.  All rights reserved.
//

#pragma once

#include "ImageConverter.h"
#include "ImageConversionTypes.h"

namespace Windows::AI::MachineLearning::Internal
{
    class ITensorToVideoFrameConverter
    {
    public:
        virtual HRESULT DX12TensorToVideoFrame(
            _In_ UINT32 batchIdx,
            _In_ winrt::Windows::AI::MachineLearning::LearningModelSession& session,
            _In_ ID3D12Resource* pInputTensor,
            _In_ const IMG_TENSOR_DESC& tensorDesc,
            _Inout_ winrt::Windows::Media::VideoFrame& destVideoFrame
        ) = 0;

        virtual HRESULT SoftwareTensorToVideoFrame(
            _In_ winrt::Windows::AI::MachineLearning::LearningModelSession& session,
            _In_ BYTE* pCPUTensorToConvert,
            _In_ IMG_TENSOR_DESC tensorDesc,
            _Inout_ winrt::Windows::Media::VideoFrame& destVideoFrame
        ) = 0;
    };

    class TensorToVideoFrameConverter : ITensorToVideoFrameConverter, public ImageConverter
    {
    public:
        // Function takes in a tensor DX12 Resource all compute ops should be completed
        // converts it to a VideoFrame backed by either a SoftwareBitmap or D3DSurface
        HRESULT DX12TensorToVideoFrame(
            _In_ UINT32 batchIdx,
            _In_ winrt::Windows::AI::MachineLearning::LearningModelSession& session,
            _In_ ID3D12Resource* pInputTensor,
            _In_ const IMG_TENSOR_DESC& tensorDesc,
            _Inout_ winrt::Windows::Media::VideoFrame& destVideoFrame
        );

        // Function takes in a byte pointer to a CPUTensor 
        // converts it to VideoFrame backed by either a SoftwareBitmap or D3DSurface,
        HRESULT SoftwareTensorToVideoFrame(
            _In_ winrt::Windows::AI::MachineLearning::LearningModelSession& session,
            _In_ BYTE* pCPUTensorToConvert,
            _In_ IMG_TENSOR_DESC tensorDesc,
            _Inout_ winrt::Windows::Media::VideoFrame& destVideoFrame
        );

    private:
        GUID _d3d11TextureGUID = { 0x14bf1054, 0x6ce7, 0x4c00,{ 0xa1, 0x32, 0xb0, 0xf2, 0x11, 0x5D, 0xE0, 0x7f } }; // {14BF1054-6CE7-4C00-A132-B0F2115DE07F}
        GUID _handleGUID = { 0x700148fc, 0xc0cb, 0x4a7e,{ 0xa7, 0xc0, 0xe7, 0x43, 0xc1, 0x9, 0x9d, 0x62 } };; // {700148FC-C0CB-4A7E-A7C0-E743C1099D62}
        Microsoft::WRL::ComPtr<ID3D12Resource> _spReadbackHeap;
        Microsoft::WRL::ComPtr<ID3D12Resource> _spOutputResource;
        Microsoft::WRL::ComPtr<ID3D12Resource> _spUAVResource;

        HRESULT ConvertGPUTensorToSoftwareBitmap(
            _In_ UINT32 batchIdx,
            _In_ ID3D12Resource* pInputTensor,
            _In_ winrt::Windows::AI::MachineLearning::implementation::D3DDeviceCache& deviceCache,
            _In_ const IMG_TENSOR_DESC& tensorDesc,
            _Inout_ winrt::Windows::Graphics::Imaging::SoftwareBitmap& softwareBitmap
        );

        HRESULT ConvertGPUTensorToDX12Texture(
            _In_ UINT32 batchIdx,
            _In_ ID3D12Resource* pInputResource,
            _In_ winrt::Windows::AI::MachineLearning::implementation::D3DDeviceCache& deviceCache,
            _In_ const IMG_TENSOR_DESC& tensorDesc,
            _Inout_ ID3D12Resource* pOutputResource
        );

        HRESULT ConvertDX12TensorToUnsupportedVideoFrameFormat(
            _In_ UINT32 batchIdx,
            _In_ ID3D12Resource* pInputTensor,
            _In_ winrt::Windows::AI::MachineLearning::implementation::D3DDeviceCache& deviceCache,
            _In_ const IMG_TENSOR_DESC& tensorDesc,
            _Inout_ winrt::Windows::Media::VideoFrame& unsupportedVideoFrame
        );

        static D3D12_SHADER_RESOURCE_VIEW_DESC TensorToVideoFrameConverter::CreateSRVDescriptor(
            const UINT32 batchIdx,
            const D3D12_RESOURCE_DESC& resourceDesc,
            const IMG_TENSOR_DESC& desc
        );

        static HRESULT ConvertCPUTensorToSoftwareBitmap(
            _In_ void* pCPUTensor,
            _In_ const IMG_TENSOR_DESC& tensorDesc,
            _Inout_ winrt::Windows::Graphics::Imaging::SoftwareBitmap& softwareBitmap
        );
    };
}