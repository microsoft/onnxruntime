// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/core/agent/training_agent.h"
#include "core/session/IOBinding.h"

namespace onnxruntime {
namespace training {

TrainingAgent::TrainingAgent(InferenceSession& session) : inference_session_(session) {}

TrainingAgent::~TrainingAgent(){};

common::Status TrainingAgent::RunForward(onnxruntime::RunOptions& run_options, onnxruntime::IOBinding& io_binding, std::vector<OrtValue>* ort_values) {
  run_options.program_counter_start = 0;
  run_options.program_counter_end = inference_session_.GetBreakpointAndEndPoint().first - 1;
  return inference_session_.Run(run_options, io_binding.GetInputNames(), io_binding.GetInputs(), io_binding.GetOutputNames(),
                                &io_binding.GetOutputs(), &io_binding.GetOutputsDeviceInfo(), ort_values);
}

common::Status TrainingAgent::RunBackward(onnxruntime::RunOptions& run_options, onnxruntime::IOBinding& io_binding, std::vector<OrtValue>* ort_values) {
  auto bp = inference_session_.GetBreakpointAndEndPoint();
  run_options.program_counter_start = bp.first;
  run_options.program_counter_end = bp.second;
  return inference_session_.Run(run_options, io_binding.GetInputNames(), io_binding.GetInputs(), io_binding.GetOutputNames(),
                                &io_binding.GetOutputs(), &io_binding.GetOutputsDeviceInfo(), ort_values);
}

}  // namespace training
}  // namespace onnxruntime
