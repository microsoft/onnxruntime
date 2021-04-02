// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/core/agent/training_agent.h"
#include "core/session/IOBinding.h"

namespace onnxruntime {
namespace training {

TrainingAgent::TrainingAgent(InferenceSession& session) : inference_session_(session) {}

TrainingAgent::~TrainingAgent() {
};

common::Status TrainingAgent::RunForward(const onnxruntime::RunOptions& run_options, onnxruntime::IOBinding& io_binding) {
  return inference_session_.Run(run_options, io_binding);
}

common::Status TrainingAgent::RunBackward(const onnxruntime::RunOptions& run_options, onnxruntime::IOBinding& io_binding) {
  return inference_session_.Run(run_options, io_binding);
}

}  // namespace training
}  // namespace onnxruntime
