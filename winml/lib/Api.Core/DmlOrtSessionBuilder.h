// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include "inc/WinMLAdapter.h"

namespace Windows::AI::MachineLearning::Adapter {

class DmlOrtSessionBuilder : public Microsoft::WRL::RuntimeClass <
    Microsoft::WRL::RuntimeClassFlags<Microsoft::WRL::ClassicCom>,
    _winmla::IOrtSessionBuilder> {

 public:
  DmlOrtSessionBuilder(ID3D12Device* device, ID3D12CommandQueue*  queue);

  HRESULT STDMETHODCALLTYPE CreateSessionOptions(
      ISessionOptions** p_options) override;

  HRESULT STDMETHODCALLTYPE CreateSession(
      ISessionOptions* options,
      _winmla::IInferenceSession** p_session,
      onnxruntime::IExecutionProvider** pp_provider) override;

  HRESULT STDMETHODCALLTYPE Initialize(
      _winmla::IInferenceSession* p_session,
      onnxruntime::IExecutionProvider* p_provider) override;

 private:
  winrt::com_ptr<ID3D12Device> device_;
  winrt::com_ptr<ID3D12CommandQueue> queue_;
};

}  // namespace Windows::AI::MachineLearning::Adapter