// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include "OnnxruntimeSessionBuilder.h"

namespace Windows::AI::MachineLearning {

class OnnxruntimeEngineFactory;

class OnnxruntimeCpuSessionBuilder : public Microsoft::WRL::RuntimeClass <
    Microsoft::WRL::RuntimeClassFlags<Microsoft::WRL::ClassicCom>,
    IOrtSessionBuilder> {

 public:
  HRESULT RuntimeClassInitialize(OnnxruntimeEngineFactory* engine_factory);

  HRESULT STDMETHODCALLTYPE CreateSessionOptions(
      OrtSessionOptions** options) override;

  HRESULT STDMETHODCALLTYPE CreateSession(
      OrtSessionOptions* options,
      OrtSession** session,
      OrtExecutionProvider** provider) override;

  HRESULT STDMETHODCALLTYPE Initialize(
      OrtSession* session,
      OrtExecutionProvider* provider) override;

  private:
  Microsoft::WRL::ComPtr<OnnxruntimeEngineFactory> engine_factory_;
};

}  // namespace Windows::AI::MachineLearning::Adapter