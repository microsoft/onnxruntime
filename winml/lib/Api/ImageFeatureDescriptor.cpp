// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "pch.h"

#include "ImageFeatureDescriptor.h"

#include <windows.graphics.imaging.h>

namespace WINMLP {
ImageFeatureDescriptor::ImageFeatureDescriptor(
    const char* name,
    const char* description,
    winml::TensorKind tensor_kind,
    const std::vector<int64_t>& shape,
    bool is_required,
    wgi::BitmapPixelFormat pixel_format,
    wgi::BitmapAlphaMode alpha_mode,
    uint32_t width,
    uint32_t height,
    winml::LearningModelPixelRange pixel_range,
    ImageColorSpaceGamma color_space_gamma) : name_(_winml::Strings::HStringFromUTF8(name)),
                                              description_(_winml::Strings::HStringFromUTF8(description)),
                                              tensor_kind_(tensor_kind),
                                              shape_(shape),
                                              is_required_(is_required),
                                              pixel_format_(pixel_format),
                                              alpha_mode_(alpha_mode),
                                              width_(width),
                                              height_(height),
                                              pixel_range_(pixel_range),
                                              color_space_gamma_(color_space_gamma) {
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

winml::LearningModelPixelRange
ImageFeatureDescriptor::PixelRange() {
  return pixel_range_;
}

ImageColorSpaceGamma
ImageFeatureDescriptor::GetColorSpaceGamma() {
  return color_space_gamma_;
}

HRESULT
ImageFeatureDescriptor::GetDescriptorInfo(
    _winml::IEngineFactory* engine_factory,
    _winml::IDescriptorInfo** info) {
    // TODO: Need to add denotations here
  engine_factory->CreateTensorDescriptorInfo(tensor_kind_, shape_.data(), shape_.size(), info);
  return S_OK;
}

}  // namespace WINMLP
