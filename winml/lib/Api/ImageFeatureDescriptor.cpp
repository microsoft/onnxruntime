// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "pch.h"

#include "ImageFeatureDescriptor.h"

#include <windows.graphics.imaging.h>

namespace winrt::Windows::AI::MachineLearning::implementation {
ImageFeatureDescriptor::ImageFeatureDescriptor(
    const char* name,
    const char* description,
    bool is_required,
    winml::TensorKind tensor_kind,
    const std::vector<int64_t>& shape,
    wgi::BitmapPixelFormat pixel_format,
    wgi::BitmapAlphaMode alpha_mode,
    uint32_t width,
    uint32_t height,
    ImageNominalPixelRange nominal_pixel_range,
    ImageColorSpaceGamma color_space_gamma) : name_(WinML::Strings::HStringFromUTF8(name)),
                                              description_(WinML::Strings::HStringFromUTF8(description)),
                                              tensor_kind_(tensor_kind),
                                              shape_(shape),
                                              is_required_(is_required),
                                              pixel_format_(pixel_format),
                                              alpha_mode_(alpha_mode),
                                              width_(width),
                                              height_(height),
                                              nominal_pixel_range_(nominal_pixel_range),
                                              color_space_gamma_(color_space_gamma) {
}

ImageFeatureDescriptor::ImageFeatureDescriptor(
        hstring const& Name,
        hstring const& Description,
        bool IsRequired,
        Windows::AI::MachineLearning::TensorKind const& TensorKind,
        array_view<int64_t const> Shape,
        Windows::Graphics::Imaging::BitmapPixelFormat const& BitmapPixelFormat,
        Windows::Graphics::Imaging::BitmapAlphaMode const& BitmapAlphaMode,
        uint32_t Width,
        uint32_t Height) : name_(Name),
                           description_(Description),
                            tensor_kind_(TensorKind),
                            shape_(Shape.begin(), Shape.end()),
                            is_required_(IsRequired),
                            pixel_format_(BitmapPixelFormat),
                            alpha_mode_(BitmapAlphaMode),
                            width_(Width),
                            height_(Height),
                            nominal_pixel_range_(ImageNominalPixelRange::ImageNominalPixelRange_NominalRange_0_255),
                            color_space_gamma_(ImageColorSpaceGamma::ImageColorSpaceGamma_SRGB) {
}

wgi::BitmapPixelFormat
ImageFeatureDescriptor::BitmapPixelFormat() try {
  return pixel_format_;
}
WINML_CATCH_ALL

wgi::BitmapAlphaMode
ImageFeatureDescriptor::BitmapAlphaMode() try {
  return alpha_mode_;
}
WINML_CATCH_ALL

uint32_t
ImageFeatureDescriptor::Width() try {
  return width_;
}
WINML_CATCH_ALL

uint32_t
ImageFeatureDescriptor::Height() try {
  return height_;
}
WINML_CATCH_ALL

hstring
ImageFeatureDescriptor::Name() try {
  return name_;
}
WINML_CATCH_ALL

hstring
ImageFeatureDescriptor::Description() try {
  return description_;
}
WINML_CATCH_ALL

winml::LearningModelFeatureKind
ImageFeatureDescriptor::Kind() try {
  return LearningModelFeatureKind::Image;
}
WINML_CATCH_ALL

bool ImageFeatureDescriptor::IsRequired() try {
  return is_required_;
}
WINML_CATCH_ALL

winml::TensorKind
ImageFeatureDescriptor::TensorKind() {
  return tensor_kind_;
}

wfc::IVectorView<int64_t>
ImageFeatureDescriptor::Shape() {
  return winrt::single_threaded_vector<int64_t>(
             std::vector<int64_t>(
                 std::begin(shape_),
                 std::end(shape_)))
      .GetView();
}

HRESULT
ImageFeatureDescriptor::GetName(
    const wchar_t** name,
    uint32_t* cchName) {
  *name = name_.data();
  *cchName = static_cast<uint32_t>(name_.size());
  return S_OK;
}

HRESULT
ImageFeatureDescriptor::GetDescription(
    const wchar_t** description,
    uint32_t* cchDescription) {
  *description = description_.data();
  *cchDescription = static_cast<uint32_t>(description_.size());
  return S_OK;
}

ImageNominalPixelRange
ImageFeatureDescriptor::GetNominalPixelRange() {
  return nominal_pixel_range_;
}

ImageColorSpaceGamma
ImageFeatureDescriptor::GetColorSpaceGamma() {
  return color_space_gamma_;
}
}  // namespace winrt::Windows::AI::MachineLearning::implementation
