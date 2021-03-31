// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/core/agent/training_agent.h"
#include "core/session/IOBinding.h"

namespace onnxruntime {
namespace training {

TrainingAgent::TrainingAgent(InferenceSession& session) : inference_session_(session) {}

TrainingAgent::~TrainingAgent() {
};

common::Status TrainingAgent::RunForward(onnxruntime::IOBinding& io_binding, int64_t& run_id) {
  run_id = inference_session_.CreatePartialRun();
  return inference_session_.PartialRun(io_binding, run_id);
}

common::Status TrainingAgent::RunBackward(onnxruntime::IOBinding& io_binding, int64_t run_id) {
  LOGS(*inference_session_.GetLogger(), VERBOSE) << "Running TrainingAgent::Backward() with run_id " << run_id;
  return inference_session_.PartialRun(io_binding, run_id);
}

void TrainingAgent::CancelPendingBackwardRun(int64_t run_id) {
  LOGS(*inference_session_.GetLogger(), WARNING) << "Canceling backward run with run_id " << run_id;
  inference_session_.CancelPartialRun(run_id);
}
}  // namespace training
}  // namespace onnxruntime
