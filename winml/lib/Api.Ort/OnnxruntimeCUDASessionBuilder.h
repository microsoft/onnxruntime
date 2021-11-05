// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "OnnxruntimeSessionBuilder.h"
#include "iengine.h"

namespace _winml {

class OnnxruntimeEngineFactory;

class OnnxruntimeCUDASessionBuilder : public Microsoft::WRL::RuntimeClass<
                                         Microsoft::WRL::RuntimeClassFlags<Microsoft::WRL::ClassicCom>,
                                         IOrtSessionBuilder> {
 public:
  HRESULT RuntimeClassInitialize(OnnxruntimeEngineFactory* engine_factory, IExecutionProviderOptions* options);

  HRESULT STDMETHODCALLTYPE CreateSessionOptions(
      OrtSessionOptions** options) override;

  HRESULT STDMETHODCALLTYPE CreateSession(
      OrtSessionOptions* options,
      OrtSession** session) override;

  HRESULT STDMETHODCALLTYPE Initialize(
      OrtSession* session) override;

 private:
  Microsoft::WRL::ComPtr<OnnxruntimeEngineFactory> engine_factory_;
  winrt::com_ptr<IExecutionProviderOptions> options_;
};

}  // namespace _winml