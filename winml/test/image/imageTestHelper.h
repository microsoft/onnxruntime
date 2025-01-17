//-----------------------------------------------------------------------------
//
//  Copyright (c) Microsoft Corporation. All rights reserved.
//
//-----------------------------------------------------------------------------

#pragma once

#include <winrt/Windows.Media.h>
#include <winrt/Windows.Graphics.Imaging.h>

enum OutputBindingStrategy {
  Bound,
  Unbound,
  Empty
};
enum EvaluationStrategy {
  Async,
  Sync
};
enum ModelInputOutputType {
  Image,
  Tensor
};
enum InputImageSource {
  FromVideoFrame,
  FromImageFeatureValue,
  FromCPUResource,
  FromGPUResource
};
enum VideoFrameSource {
  FromSoftwareBitmap,
  FromDirect3DSurface,
  FromUnsupportedD3DSurface
};

namespace ImageTestHelper {
wgi::BitmapPixelFormat GetPixelFormat(const std::wstring& inputPixelFormat);

winml::TensorFloat LoadInputImageFromCPU(wgi::SoftwareBitmap softwareBitmap, const std::wstring& modelPixelFormat);

winml::TensorFloat LoadInputImageFromGPU(wgi::SoftwareBitmap softwareBitmap, const std::wstring& modelPixelFormat);

bool VerifyHelper(wm::VideoFrame actual, wm::VideoFrame expected);

}  // namespace ImageTestHelper
