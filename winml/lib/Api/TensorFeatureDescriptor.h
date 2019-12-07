// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include "TensorFeatureDescriptor.g.h"

namespace winrt::Windows::AI::MachineLearning::implementation {
struct TensorFeatureDescriptor : TensorFeatureDescriptorT<
                                     TensorFeatureDescriptor,
                                     ILearningModelFeatureDescriptorNative> {
  TensorFeatureDescriptor() = delete;
  TensorFeatureDescriptor(
      const char* name,
      const char* description,
      bool is_required,
      winml::TensorKind tensor_kind,
      const std::vector<int64_t>& shape,
      bool has_unsuppored_image_metadata);

  TensorFeatureDescriptor(
      hstring const& Name,
      hstring const& Description,
      bool IsRequired,
      TensorKind const& TensorKind,
      array_view<int64_t const> Shape,
      bool HasUnsupportedImageMetadata);

  // ITensorDescriptor
  winml::TensorKind TensorKind();
  wfc::IVectorView<int64_t> Shape();
  bool HasUnsupportedImageMetadata();

  // IFeatureDescriptor
  winrt::hstring
  Name();

  winrt::hstring
  Description();

  winml::LearningModelFeatureKind
  Kind();

  bool
  IsRequired();

  bool
  IsUnsupportedMetaData();

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
  bool has_unsupported_image_metadata_;
};
}  // namespace winrt::Windows::AI::MachineLearning::implementation

namespace winrt::Windows::AI::MachineLearning::factory_implementation {
    struct TensorFeatureDescriptor : TensorFeatureDescriptorT<TensorFeatureDescriptor, implementation::TensorFeatureDescriptor> {

    };
}  // namespace winrt::Windows::AI::MachineLearning::factory_implementation
