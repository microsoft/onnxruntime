// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/session/inference_session.h"
#include "orttraining/training_api/interfaces.h"

namespace onnxruntime {
namespace training {
namespace api_test {
namespace utils {
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
