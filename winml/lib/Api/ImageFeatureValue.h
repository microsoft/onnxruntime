// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "ImageFeatureValue.g.h"

#include "inc/ILotusValueProviderPrivate.h"

namespace WINMLP {

struct ImageFeatureValue : ImageFeatureValueT<ImageFeatureValue, _winml::ILotusValueProviderPrivate> {
  // Metadata about the resource which helps in finding
  // compatible cached resources
  struct ImageResourceMetadata;

  ImageFeatureValue() = delete;
  ImageFeatureValue(Windows::Media::VideoFrame const& image);
  ImageFeatureValue(wfc::IVector<Windows::Media::VideoFrame> const& images);
  ImageFeatureValue(wfc::IVectorView<Windows::Media::VideoFrame> const& images);

  Windows::Media::VideoFrame VideoFrame();
  wfc::IIterable<Windows::Media::VideoFrame> VideoFrames();
  winml::LearningModelFeatureKind Kind();

  static winml::ImageFeatureValue ImageFeatureValue::Create(
    uint32_t batchSize, Windows::Graphics::Imaging::BitmapPixelFormat format, uint32_t width, uint32_t height
  );
  static winml::ImageFeatureValue CreateFromVideoFrame(Windows::Media::VideoFrame const& image);

  std::optional<ImageResourceMetadata> GetInputMetadata(const _winml::BindingContext& context);

  // ILotusValueProviderPrivate implementation
  STDMETHOD(GetValue)
  (_winml::BindingContext& context, _winml::IValue** out);
  STDMETHOD(IsPlaceholder)
  (bool* pIsPlaceHolder);
  STDMETHOD(UpdateSourceResourceData)
  (_winml::BindingContext& context, _winml::IValue* value);
  STDMETHOD(AbiRepresentation)
  (wf::IInspectable& abiRepresentation);

  std::vector<uint32_t> Widths() { return m_widths; }
  std::vector<uint32_t> Heights() { return m_heights; }
  bool IsBatch() { return m_batchSize > 1; }

 private:
  wfc::IVector<wm::VideoFrame> m_videoFrames;
  std::vector<uint32_t> m_widths = {};
  std::vector<uint32_t> m_heights = {};
  uint32_t m_batchSize = 1;
  // Crop the image with desired aspect ratio.
  // This function does not crop image to desried width and height, but crops to center for desired ratio
  wgi::BitmapBounds CenterAndCropBounds(uint32_t idx, uint32_t desiredWidth, uint32_t desiredHeight);
  void Initialize();
};
}  // namespace WINMLP

namespace WINML::factory_implementation {
struct ImageFeatureValue : ImageFeatureValueT<ImageFeatureValue, implementation::ImageFeatureValue> {};
}  // namespace WINML::factory_implementation
