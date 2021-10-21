// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

namespace _winml {

// The IOrtSessionBuilder offers an abstraction over the creation of
// InferenceSession, that enables the creation of the session based on a device (CPU/DML).
MIDL_INTERFACE("2746f03a-7e08-4564-b5d0-c670fef116ee")
IOrtSessionBuilder : IUnknown {
  virtual HRESULT STDMETHODCALLTYPE CreateSessionOptions(
      OrtSessionOptions * *options) = 0;

  virtual HRESULT STDMETHODCALLTYPE CreateSession(
      OrtSessionOptions * options,
      OrtSession * *session) = 0;

  virtual HRESULT STDMETHODCALLTYPE Initialize(
      OrtSession * session) = 0;
};

}  // namespace _winml