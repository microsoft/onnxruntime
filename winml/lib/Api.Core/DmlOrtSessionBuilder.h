#pragma once

#include "inc/IOrtSessionBuilder.h"

namespace Windows::AI::MachineLearning {

class DmlOrtSessionBuilder : public IOrtSessionBuilder {
 public:
  DmlOrtSessionBuilder(winml::LearningModelDevice const& device);

  HRESULT __stdcall CreateSessionOptions(
      onnxruntime::SessionOptions* p_options);

  HRESULT __stdcall CreateSession(
      const onnxruntime::SessionOptions& options,
      std::unique_ptr<onnxruntime::InferenceSession>* p_session,
      onnxruntime::IExecutionProvider** pp_provider);

  HRESULT __stdcall Initialize(
      onnxruntime::InferenceSession* p_session,
      onnxruntime::IExecutionProvider* p_provider);

 private:
  winml::LearningModelDevice device_;
};

}  // namespace Windows::AI::MachineLearning