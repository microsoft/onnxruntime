// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include "WinMLAdapter.h"

namespace Windows::AI::MachineLearning::Adapter {

class CpuOrtSessionBuilder : public Microsoft::WRL::RuntimeClass <
    Microsoft::WRL::RuntimeClassFlags<Microsoft::WRL::ClassicCom>,
    winmla::IOrtSessionBuilder> {

 public:
     CpuOrtSessionBuilder();

  HRESULT STDMETHODCALLTYPE CreateSessionOptions(
      OrtSessionOptions** options) override;

  HRESULT STDMETHODCALLTYPE CreateSession(
      OrtSessionOptions* options,
      winmla::IInferenceSession** p_session,
      onnxruntime::IExecutionProvider** pp_provider) override;

  HRESULT STDMETHODCALLTYPE Initialize(
      winmla::IInferenceSession* p_session,
      onnxruntime::IExecutionProvider* p_provider) override;
};

}  // namespace Windows::AI::MachineLearning::Adapter