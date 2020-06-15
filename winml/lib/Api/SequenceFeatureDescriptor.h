// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "SequenceFeatureDescriptor.g.h"

namespace WINMLP {
struct SequenceFeatureDescriptor : SequenceFeatureDescriptorT<
                                       SequenceFeatureDescriptor,
                                       ILearningModelFeatureDescriptorNative> {
  SequenceFeatureDescriptor() = delete;
  SequenceFeatureDescriptor(
      const char* name,
      const char* description,
      bool is_required,
      winml::ILearningModelFeatureDescriptor element_descriptor);

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
}  // namespace WINMLP