// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/training_api/include/interfaces.h"

namespace onnxruntime {
namespace training {
namespace api {

/*
  module.train_sess.RegisterExecutionProvider(provider);
  module.eval_sess.RegisterExecutionProvider(provider);
  optimizer.optim_sess.RegisterExecutionProvider(provider);
*/
void SetExecutionProvider(const Module& /*module*/, const Optimizer& /*optimizer*/, IExecutionProvider* /*provider*/) {
  ORT_NOT_IMPLEMENTED("Not implemented.");
}
}  // namespace api
}  // namespace training
}  // namespace onnxruntime
