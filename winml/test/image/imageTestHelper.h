//-----------------------------------------------------------------------------
//
//  Copyright (c) Microsoft Corporation. All rights reserved.
//
//-----------------------------------------------------------------------------

#pragma once

#include <winrt/Windows.Media.h>
#include <winrt/Windows.Graphics.Imaging.h>

enum OutputBindingStrategy { Bound, Unbound, Empty };
enum EvaluationStrategy { Async, Sync };
enum ModelInputOutputType { Image, Tensor };
enum InputImageSource { FromVideoFrame, FromImageFeatureValue, FromCPUResource, FromGPUResource };
enum VideoFrameSource { FromSoftwareBitmap, FromDirect3DSurface, FromUnsupportedD3DSurface };

namespace ImageTestHelper
{
    winrt::Windows::Graphics::Imaging::BitmapPixelFormat GetPixelFormat(const std::wstring& inputPixelFormat);

    winrt::Windows::AI::MachineLearning::TensorFloat LoadInputImageFromCPU(
                winrt::Windows::Graphics::Imaging::SoftwareBitmap softwareBitmap,
                const std::wstring& modelPixelFormat);

    winrt::Windows::AI::MachineLearning::TensorFloat LoadInputImageFromGPU(
                winrt::Windows::Graphics::Imaging::SoftwareBitmap softwareBitmap,
                const std::wstring& modelPixelFormat);

    bool VerifyHelper(
                winrt::Windows::Media::VideoFrame actual,
                winrt::Windows::Media::VideoFrame expected);

}
