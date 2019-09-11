// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

// Forward declarations
namespace onnxruntime {
struct SessionOptions;
class IExecutionProvider;
class InferenceSession;
}  // namespace onnxruntime

namespace Windows::AI::MachineLearning {

// The IOrtSessionBuilder offers an abstraction over the creation of
// InferenceSession, that enables the creation of the session based on a device (CPU/DML).
struct __declspec(novtable) IOrtSessionBuilder {
  virtual ~IOrtSessionBuilder(){};

  virtual HRESULT __stdcall CreateSessionOptions(
      onnxruntime::SessionOptions* options) = 0;

  virtual HRESULT __stdcall CreateSession(
      const onnxruntime::SessionOptions& options,
      std::unique_ptr<onnxruntime::InferenceSession>* session,
      onnxruntime::IExecutionProvider** provider) = 0;

  virtual HRESULT __stdcall Initialize(
      onnxruntime::InferenceSession* session,
      onnxruntime::IExecutionProvider* provider) = 0;
};

// factory method for creating an ortsessionbuilder from a device
std::unique_ptr<WinML::IOrtSessionBuilder>
CreateOrtSessionBuilder(
    winml::LearningModelDevice const& device);

}  // namespace Windows::AI::MachineLearning