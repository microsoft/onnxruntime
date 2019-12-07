// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include "SequenceFeatureDescriptor.g.h"

namespace winrt::Windows::AI::MachineLearning::implementation {
struct SequenceFeatureDescriptor : SequenceFeatureDescriptorT<
                                       SequenceFeatureDescriptor,
                                       ILearningModelFeatureDescriptorNative> {
  SequenceFeatureDescriptor() = delete;
  SequenceFeatureDescriptor(
      const char* name,
      const char* description,
      bool is_required,
      winml::ILearningModelFeatureDescriptor element_descriptor);
  SequenceFeatureDescriptor(
      hstring const& Name,
      hstring const& Description,
      bool IsRequired,
      Windows::AI::MachineLearning::ILearningModelFeatureDescriptor const& ElementDescriptor);

  winml::ILearningModelFeatureDescriptor
  ElementDescriptor();

  // IFeatureDescriptor
  hstring
  Name();

  hstring
  Description();

  winml::LearningModelFeatureKind
  Kind();

  bool
  IsRequired();

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
  bool is_required_;
  winml::ILearningModelFeatureDescriptor element_descriptor_;
};
}  // namespace winrt::Windows::AI::MachineLearning::implementation

namespace winrt::Windows::AI::MachineLearning::factory_implementation {
    struct SequenceFeatureDescriptor : SequenceFeatureDescriptorT<SequenceFeatureDescriptor, implementation::SequenceFeatureDescriptor> {

    };
}  // namespace winrt::Windows::AI::MachineLearning::factory_implementation
