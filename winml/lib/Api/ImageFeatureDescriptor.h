// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include "ImageFeatureDescriptor.g.h"

namespace winrt::Windows::AI::MachineLearning::implementation {
enum class ImageNominalPixelRange {
  ImageNominalPixelRange_NominalRange_0_255,
  ImageNominalPixelRange_Normalized_0_1,
  ImageNominalPixelRange_Normalized_1_1,
  ImageNominalPixelRange_NominalRange_16_235,
};
enum class ImageColorSpaceGamma {
  ImageColorSpaceGamma_Linear,
  ImageColorSpaceGamma_SRGB,
};

struct ImageFeatureDescriptor : ImageFeatureDescriptorT<
                                    ImageFeatureDescriptor,
                                    ILearningModelFeatureDescriptorNative> {
  ImageFeatureDescriptor() = delete;
  ImageFeatureDescriptor(
      const char* name,
      const char* description,
      bool is_required,
      winml::TensorKind tensor_kind,
      const std::vector<int64_t>& shape,
      wgi::BitmapPixelFormat pixelformat,
      wgi::BitmapAlphaMode alphamode,
      uint32_t width,
      uint32_t height,
      ImageNominalPixelRange nominalPixelRange,
      ImageColorSpaceGamma colorSpaceGamma);

  ImageFeatureDescriptor(
      hstring const& Name,
      hstring const& Description,
      bool IsRequired,
      Windows::AI::MachineLearning::TensorKind const& TensorKind, 
      array_view<int64_t const> Shape,
      Windows::Graphics::Imaging::BitmapPixelFormat const& BitmapPixelFormat,
      Windows::Graphics::Imaging::BitmapAlphaMode const& BitmapAlphaMode,
      uint32_t Width,
      uint32_t Height);

  wgi::BitmapPixelFormat
  BitmapPixelFormat();

  wgi::BitmapAlphaMode
  BitmapAlphaMode();

  uint32_t
  Width();

  uint32_t
  Height();

  hstring
  Name();

  hstring
  Description();

  winml::LearningModelFeatureKind
  Kind();

  bool
  IsRequired();

  winml::TensorKind
  TensorKind();

  wfc::IVectorView<int64_t>
  Shape();

  ImageNominalPixelRange
  GetNominalPixelRange();

  ImageColorSpaceGamma
  GetColorSpaceGamma();

  STDMETHOD(GetName)
  (
      const wchar_t** name,
      uint32_t* cchName) override;

  STDMETHOD(GetDescription)
  (
      const wchar_t** description,
      uint32_t* cchDescription) override;

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
  ImageNominalPixelRange nominal_pixel_range_;
  ImageColorSpaceGamma color_space_gamma_;
};
}  // namespace winrt::Windows::AI::MachineLearning::implementation

namespace winrt::Windows::AI::MachineLearning::factory_implementation {
    struct ImageFeatureDescriptor : ImageFeatureDescriptorT<ImageFeatureDescriptor, implementation::ImageFeatureDescriptor> {

    };
}  // namespace winrt::Windows::AI::MachineLearning::factory_implementation
