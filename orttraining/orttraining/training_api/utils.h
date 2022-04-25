// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if defined(ENABLE_TRAINING) && defined(ENABLE_TRAINING_ON_DEVICE)

#pragma once
#include "core/session/inference_session.h"
#include "orttraining/training_api/interfaces.h"

namespace onnxruntime {
namespace training {
namespace api_test {
namespace utils {
struct CheckpointProperty {
  int value;
  // Support primitive types like int, float, string leveraging type trait.
};

struct CheckpointStates {
  CheckpointStates() {
    ORT_NOT_IMPLEMENTED("Not implemented.");
  }
  std::map<std::string, std::shared_ptr<Parameter>> named_parameters;
  OptimizerState optimizer_states;
  std::unordered_map<std::string, CheckpointProperty> named_properties;
};

// Save properties into a checkpoint property file (with postfix .prop).
Status Ort_Save(CheckpointStates& /*state_dicts*/, const PathString& /*checkpoint_path*/);

// Load properties file having postfix being '.prop'.
Status Ort_Load(const PathString& /*checkpoint_path*/, CheckpointStates& /*state_dicts*/);

/*
  module.train_sess.RegisterExecutionProvider(provider);
  module.eval_sess.RegisterExecutionProvider(provider);
  optimizer.optim_sess.RegisterExecutionProvider(provider);
*/
void SetExecutionProvider(const Module& /*module*/, const Optimizer& /*optimizer*/, IExecutionProvider* /*provider*/);

}  // namespace utils
}  // namespace api_test
}  // namespace training
}  // namespace onnxruntime

#endif