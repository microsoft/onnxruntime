// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "OnnxruntimeSessionBuilder.h"

namespace _winml {

class OnnxruntimeEngineFactory;

class OnnxruntimeCpuSessionBuilder : public Microsoft::WRL::RuntimeClass<
                                         Microsoft::WRL::RuntimeClassFlags<Microsoft::WRL::ClassicCom>,
                                         IOrtSessionBuilder> {
 public:
  HRESULT RuntimeClassInitialize(OnnxruntimeEngineFactory* engine_factory);

  HRESULT STDMETHODCALLTYPE CreateSessionOptions(
      OrtSessionOptions** options) override;

  HRESULT STDMETHODCALLTYPE CreateSession(
      OrtSessionOptions* options,
      OrtSession** session) override;

  HRESULT STDMETHODCALLTYPE Initialize(
      OrtSession* session) override;

 private:
  Microsoft::WRL::ComPtr<OnnxruntimeEngineFactory> engine_factory_;
};

}  // namespace _winml