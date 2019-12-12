// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include "ImageFeatureValue.g.h"

#include "inc/ILotusValueProviderPrivate.h"

namespace winrt::Windows::AI::MachineLearning::implementation {
struct ImageFeatureValue : ImageFeatureValueT<ImageFeatureValue, WinML::ILotusValueProviderPrivate> {
  // Metadata about the resource which helps in finding
  // compatible cached resources
  struct ImageResourceMetadata;

  ImageFeatureValue() = delete;
  ~ImageFeatureValue();
  ImageFeatureValue(Windows::Media::VideoFrame const& image);
  ImageFeatureValue(winrt::Windows::Foundation::Collections::IVector<Windows::Media::VideoFrame> const& images);
  ImageFeatureValue(winrt::Windows::Foundation::Collections::IVectorView<Windows::Media::VideoFrame> const& images);

  Windows::Media::VideoFrame VideoFrame();
  winrt::Windows::Foundation::Collections::IIterable<Windows::Media::VideoFrame> VideoFrames();
  Windows::AI::MachineLearning::LearningModelFeatureKind Kind();

  static Windows::AI::MachineLearning::ImageFeatureValue ImageFeatureValue::Create(
      uint32_t batchSize,
      Windows::Graphics::Imaging::BitmapPixelFormat format,
      uint32_t width,
      uint32_t height);
  static Windows::AI::MachineLearning::ImageFeatureValue CreateFromVideoFrame(Windows::Media::VideoFrame const& image);

  std::optional<ImageResourceMetadata> GetInputMetadata(const WinML::BindingContext& context);

  // ILotusValueProviderPrivate implementation
  STDMETHOD(GetOrtValue)
  (WinML::BindingContext& context, OrtValue** ort_value);
  STDMETHOD(IsPlaceholder)
  (bool* pIsPlaceHolder);
  STDMETHOD(UpdateSourceResourceData)
  (WinML::BindingContext& context, OrtValue* ort_value);
  STDMETHOD(AbiRepresentation)
  (winrt::Windows::Foundation::IInspectable& abiRepresentation);

  std::vector<uint32_t> Widths() { return m_widths; }
  std::vector<uint32_t> Heights() { return m_heights; }
  bool IsBatch() { return m_batchSize > 1; }

 private:
  com_ptr<winmla::IWinMLAdapter> m_adapter;
  winrt::Windows::Foundation::Collections::IVector<Windows::Media::VideoFrame> m_videoFrames;
  std::vector<uint32_t> m_widths = {};
  std::vector<uint32_t> m_heights = {};
  std::vector<OrtAllocator *> m_tensorAllocators;
  uint32_t m_batchSize = 1;
  // Crop the image with desired aspect ratio.
  // This function does not crop image to desried width and height, but crops to center for desired ratio
  Windows::Graphics::Imaging::BitmapBounds CenterAndCropBounds(
      uint32_t idx,
      uint32_t desiredWidth,
      uint32_t desiredHeight);
  void Initialize();
};
}  // namespace winrt::Windows::AI::MachineLearning::implementation

namespace winrt::Windows::AI::MachineLearning::factory_implementation {
struct ImageFeatureValue : ImageFeatureValueT<ImageFeatureValue, implementation::ImageFeatureValue> {
};
}  // namespace winrt::Windows::AI::MachineLearning::factory_implementation
