// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include "iengine.h"
#include "adapter/winml_adapter_c_api.h"

using UniqueOrtSessionOptions = std::unique_ptr<OrtSessionOptions, void (*)(OrtSessionOptions*)>;
using UniqueOrtSession = std::unique_ptr<OrtSession, void (*)(OrtSession*)>;
using UniqueOrtExecutionProvider = std::unique_ptr<OrtExecutionProvider, void (*)(OrtExecutionProvider*)>;

// The IOrtSessionBuilder offers an abstraction over the creation of
// InferenceSession, that enables the creation of the session based on a device (CPU/DML).
MIDL_INTERFACE("2746f03a-7e08-4564-b5d0-c670fef116ee")
IOrtSessionBuilder : IUnknown {
  virtual HRESULT STDMETHODCALLTYPE CreateSessionOptions(
      OrtSessionOptions * *options) = 0;

  virtual HRESULT STDMETHODCALLTYPE CreateSession(
      OrtSessionOptions * options,
      OrtSession** session,
      OrtExecutionProvider** provider) = 0;

  virtual HRESULT STDMETHODCALLTYPE Initialize(
      OrtSession* session,
      OrtExecutionProvider* provider) = 0;
};