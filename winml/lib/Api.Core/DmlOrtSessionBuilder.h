// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include "inc/IOrtSessionBuilder.h"

namespace Windows::AI::MachineLearning {

class DmlOrtSessionBuilder : public IOrtSessionBuilder {
 public:
  DmlOrtSessionBuilder(ID3D12Device* device, ID3D12CommandQueue*  queue);

  HRESULT __stdcall CreateSessionOptions(
      onnxruntime::SessionOptions* p_options);

  HRESULT __stdcall CreateSession(
      const onnxruntime::SessionOptions& options,
      _winmla::InferenceSession** p_session,
      onnxruntime::IExecutionProvider** pp_provider);

  HRESULT __stdcall Initialize(
      _winmla::InferenceSession* p_session,
      onnxruntime::IExecutionProvider* p_provider);

 private:
  winrt::com_ptr<ID3D12Device> device_;
  winrt::com_ptr<ID3D12CommandQueue> queue_;
};

}  // namespace Windows::AI::MachineLearning