// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "ImageFeatureDescriptor.g.h"
#include "iengine.h"

namespace WINMLP {

enum class ImageColorSpaceGamma {
  ImageColorSpaceGamma_Linear,
  ImageColorSpaceGamma_SRGB,
};

struct ImageFeatureDescriptor : ImageFeatureDescriptorT<
                                  ImageFeatureDescriptor,
                                  ILearningModelFeatureDescriptorNative,
                                  _winml::IDescriptorInfoProvider> {
  ImageFeatureDescriptor() = delete;
  ImageFeatureDescriptor(
    const char* name,
    const char* description,
    winml::TensorKind tensor_kind,
    const std::vector<int64_t>& shape,
    bool is_required,
    wgi::BitmapPixelFormat pixelformat,
    wgi::BitmapAlphaMode alphamode,
    uint32_t width,
    uint32_t height,
    winml::LearningModelPixelRange pixelRange,
    ImageColorSpaceGamma colorSpaceGamma
  );

  wgi::BitmapPixelFormat BitmapPixelFormat();

  wgi::BitmapAlphaMode BitmapAlphaMode();

  uint32_t Width();

  uint32_t Height();

  hstring Name();

  hstring Description();

  winml::LearningModelFeatureKind Kind();

  bool IsRequired();

  winml::TensorKind TensorKind();

  wfc::IVectorView<int64_t> Shape();

  winml::LearningModelPixelRange PixelRange();

  ImageColorSpaceGamma GetColorSpaceGamma();

  STDMETHOD(GetName)
  (const wchar_t** name, uint32_t* cchName) override;

  STDMETHOD(GetDescription)
  (const wchar_t** description, uint32_t* cchDescription) override;

  STDMETHOD(GetDescriptorInfo)
  (_winml::IEngineFactory* engine_factory, _winml::IDescriptorInfo** info) override;

 private:
  winrt::hstring name_;
  winrt::hstring description_;
  winml::TensorKind tensor_kind_;
  std::vector<int64_t> shape_;
  bool is_required_;
  wgi::BitmapPixelFormat pixel_format_;
  wgi::BitmapAlphaMode alpha_mode_;
  uint32_t width_;
  uint32_t height_;
  winml::LearningModelPixelRange pixel_range_;
  ImageColorSpaceGamma color_space_gamma_;
};

}  // namespace WINMLP
