// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "pch.h"

#include "SequenceFeatureDescriptor.h"

namespace WINMLP {
SequenceFeatureDescriptor::SequenceFeatureDescriptor(
    const char* name,
    const char* description,
    bool is_required,
    winml::ILearningModelFeatureDescriptor descriptor) : name_(_winml::Strings::HStringFromUTF8(name)),
                                                         description_(_winml::Strings::HStringFromUTF8(description)),
                                                         is_required_(is_required),
                                                         element_descriptor_(descriptor) {}


winml::ILearningModelFeatureDescriptor
SequenceFeatureDescriptor::ElementDescriptor() try {
  return element_descriptor_;
}
WINML_CATCH_ALL

hstring
SequenceFeatureDescriptor::Name() try {
  return name_;
}
WINML_CATCH_ALL

hstring
SequenceFeatureDescriptor::Description() try {
  return description_;
}
WINML_CATCH_ALL

winml::LearningModelFeatureKind
SequenceFeatureDescriptor::Kind() try {
  return LearningModelFeatureKind::Sequence;
}
WINML_CATCH_ALL

bool SequenceFeatureDescriptor::IsRequired() try {
  return is_required_;
}
WINML_CATCH_ALL

HRESULT
SequenceFeatureDescriptor::GetName(
    const wchar_t** name,
    uint32_t* cchName) {
  *name = name_.data();
  *cchName = static_cast<uint32_t>(name_.size());
  return S_OK;
}

HRESULT
SequenceFeatureDescriptor::GetDescription(
    const wchar_t** description,
    uint32_t* cchDescription) {
  *description = description_.data();
  *cchDescription = static_cast<uint32_t>(description_.size());
  return S_OK;
}

HRESULT
SequenceFeatureDescriptor::GetDescriptorInfo(
    _winml::IEngineFactory* engine_factory,
    _winml::IDescriptorInfo** info) {
  engine_factory->CreateSequenceDescriptorInfo(info);
  return S_OK;
};


}  // namespace WINMLP