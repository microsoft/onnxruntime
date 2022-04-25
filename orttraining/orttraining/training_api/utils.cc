// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if defined(ENABLE_TRAINING) && defined(ENABLE_TRAINING_ON_DEVICE)

#include "core/session/inference_session.h"
#include "orttraining/training_api/utils.h"

namespace onnxruntime {
namespace training {
namespace api_test {
namespace utils {

// Save properties into a checkpoint property file (with postfix .prop).
Status Ort_Save(CheckpointStates& /*state_dicts*/, const PathString& /*checkpoint_path*/) {
  ORT_NOT_IMPLEMENTED("Not implemented.");
  return Status::OK();
}

// Load properties file having postfix being '.prop'.
Status Ort_Load(const PathString& /*checkpoint_path*/, CheckpointStates& /*state_dicts*/) {
  ORT_NOT_IMPLEMENTED("Not implemented.");
  return Status::OK();
}

/*
  module.train_sess.RegisterExecutionProvider(provider);
  module.eval_sess.RegisterExecutionProvider(provider);
  optimizer.optim_sess.RegisterExecutionProvider(provider);
*/
void SetExecutionProvider(const Module& /*module*/, const Optimizer& /*optimizer*/, IExecutionProvider* /*provider*/) {
  ORT_NOT_IMPLEMENTED("Not implemented.");
}

}  // namespace utils
}  // namespace api_test
}  // namespace training
}  // namespace onnxruntime

#endif