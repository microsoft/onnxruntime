// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

namespace _winml {

// clang-format off
/* [uuid("3e4d4350-0b61-4517-aa6d-79d49bf164b4"), feature, contract, object, exclusiveto] */
MIDL_INTERFACE("3e4d4350-0b61-4517-aa6d-79d49bf164b4")
IMapFeatureValue : public ::IUnknown {
 public:
  /* [propget] */ virtual HRESULT STDMETHODCALLTYPE get_KeyKind(
      /* [out, retval] */ winml::TensorKind* kind) = 0;

  /* [propget] */ virtual HRESULT STDMETHODCALLTYPE get_ValueDescriptor(
      /* [out, retval] */ winml::ILearningModelFeatureDescriptor* result) = 0;
};
// clang-format on

}  // namespace _winml
