// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

namespace _winml {

// clang-format off
/* [uuid("529d0bca-4c6c-48c1-9bd3-e1ea2e816348"), feature, contract, object, exclusiveto] */
MIDL_INTERFACE("529d0bca-4c6c-48c1-9bd3-e1ea2e816348")
ISequenceFeatureValue : public ::IUnknown {
 public:
  /* [propget] */ virtual HRESULT STDMETHODCALLTYPE get_ElementDescriptor(
      /* [out, retval] */ winml::ILearningModelFeatureDescriptor* result) = 0;
};
// clang-format on

}  // namespace _winml
