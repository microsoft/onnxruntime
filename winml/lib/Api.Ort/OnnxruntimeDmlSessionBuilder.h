// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "OnnxruntimeSessionBuilder.h"

namespace _winml {

class OnnxruntimeEngineFactory;

class OnnxruntimeDmlSessionBuilder
  : public Microsoft::WRL::
      RuntimeClass<Microsoft::WRL::RuntimeClassFlags<Microsoft::WRL::ClassicCom>, IOrtSessionBuilder> {
 public:
  HRESULT RuntimeClassInitialize(
    OnnxruntimeEngineFactory* engine_factory,
    ID3D12Device* device,
    ID3D12CommandQueue* queue,
    bool metacommands_enabled_
  );

  HRESULT STDMETHODCALLTYPE CreateSessionOptions(OrtSessionOptions** options) override;

  HRESULT STDMETHODCALLTYPE CreateSession(
    OrtSessionOptions* options,
    OrtThreadPool* inter_op_thread_pool,
    OrtThreadPool* intra_op_thread_pool,
    OrtSession** session
  ) override;

  HRESULT STDMETHODCALLTYPE Initialize(OrtSession* session) override;

 private:
  Microsoft::WRL::ComPtr<OnnxruntimeEngineFactory> engine_factory_;
  winrt::com_ptr<ID3D12Device> device_;
  winrt::com_ptr<ID3D12CommandQueue> queue_;
  bool metacommands_enabled_ = true;
};

}  // namespace _winml
