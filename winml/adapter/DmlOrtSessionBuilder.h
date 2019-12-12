// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include "WinMLAdapter.h"

namespace Windows::AI::MachineLearning::Adapter {

class DmlOrtSessionBuilder : public Microsoft::WRL::RuntimeClass <
    Microsoft::WRL::RuntimeClassFlags<Microsoft::WRL::ClassicCom>,
    winmla::IOrtSessionBuilder> {

 public:
  DmlOrtSessionBuilder(ID3D12Device* device, ID3D12CommandQueue*  queue);

  HRESULT STDMETHODCALLTYPE CreateSessionOptions(
      OrtSessionOptions** options) override;

  HRESULT STDMETHODCALLTYPE CreateSession(
      OrtSessionOptions* options,
      winmla::IInferenceSession** p_session,
      onnxruntime::IExecutionProvider** pp_provider) override;

  HRESULT STDMETHODCALLTYPE Initialize(
      winmla::IInferenceSession* p_session,
      onnxruntime::IExecutionProvider* p_provider) override;

 private:
  winrt::com_ptr<ID3D12Device> device_;
  winrt::com_ptr<ID3D12CommandQueue> queue_;
  bool enableMetacommands_ = true;
};

}  // namespace Windows::AI::MachineLearning::Adapter